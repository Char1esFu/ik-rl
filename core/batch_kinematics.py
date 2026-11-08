"""Batched forward kinematics + collision checks for thousands of configurations per step.

Two backends with the same results (TCP pose to float32 precision, identical collision flags; checked
against RobotKinematics.fk, i.e. the kinematics, TCP and collision model of the SAC pipeline):

- "warp" (default on CUDA): MuJoCo Warp on the GPU. kinematics + collision + a per-world contact reduction
  run as one CUDA graph per batch-size bucket; inputs / outputs are torch tensors that never leave the GPU.
  Batches of <= CPU_MAX_BATCH rows (e.g. a single solve()) use the CPU path, which is faster there.
- "cpu": MuJoCo on the CPU, split over worker processes that share arrays through shared memory (MuJoCo's
  Python calls hold the GIL, so threads do not scale); num_workers=0 runs in-process.

fk_torch(q) -> (p, R, J | None, collided) on q's device; fk(q) is the numpy interface (always with J).
The Jacobian of the warp backend is built from the joint anchors / axes (all arm joints are hinges).
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory

import mujoco
import numpy as np
import torch

from core.kinematics import RobotKinematics
from core.robot import RobotSpec

CPU_MAX_BATCH = 32
WARP_BUCKETS = (256, 1024, 4096, 16384, 65536)
CONTACTS_PER_WORLD = 32  # contact buffer size per world (Franka: ~4 on average, all colliding pairs counted)


def _shapes(max_batch: int, n: int) -> dict[str, tuple[tuple[int, ...], type]]:
    return {
        "q": ((max_batch, n), np.float32),
        "pos": ((max_batch, 3), np.float32),
        "rot": ((max_batch, 3, 3), np.float32),
        "jac": ((max_batch, 6, n), np.float32),
        "col": ((max_batch,), np.bool_),
    }


def _compute_rows(kin: RobotKinematics, data: mujoco.MjData, arr: dict[str, np.ndarray], lo: int, hi: int) -> None:
    for i in range(lo, hi):
        arr["pos"][i], arr["rot"][i], arr["jac"][i], arr["col"][i] = kin.fk(data, arr["q"][i])


def _worker(conn, robot_dict: dict, shm_names: dict[str, str], max_batch: int, n: int) -> None:
    kin = RobotKinematics(RobotSpec.from_dict(robot_dict))
    data = mujoco.MjData(kin.model)
    shms = {k: shared_memory.SharedMemory(name=v) for k, v in shm_names.items()}
    arr = {k: np.ndarray(s, dtype=t, buffer=shms[k].buf) for k, (s, t) in _shapes(max_batch, n).items()}
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            _compute_rows(kin, data, arr, *msg)
            conn.send(True)
    finally:
        for s in shms.values():
            s.close()


class _CPUPool:
    """RobotKinematics.fk over rows, in worker processes (num_workers > 0) or in-process."""

    def __init__(self, robot: RobotSpec, kin: RobotKinematics, num_workers: int, max_batch: int) -> None:
        self.kin = kin
        self.n = kin.n
        self.max_batch = max_batch
        self.num_workers = num_workers
        self.data = mujoco.MjData(kin.model)
        self._procs: list = []
        self._conns: list = []
        self._shms: dict[str, shared_memory.SharedMemory] = {}
        if num_workers > 0:
            ctx = mp.get_context("spawn")  # CUDA may already be initialized in the parent
            for k, (s, t) in _shapes(max_batch, self.n).items():
                size = int(np.prod(s)) * np.dtype(t).itemsize
                self._shms[k] = shared_memory.SharedMemory(create=True, size=max(size, 1))
            self.arr = {
                k: np.ndarray(s, dtype=t, buffer=self._shms[k].buf) for k, (s, t) in _shapes(max_batch, self.n).items()
            }
            names = {k: v.name for k, v in self._shms.items()}
            for _ in range(num_workers):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, robot.to_dict(), names, max_batch, self.n), daemon=True)
                p.start()
                self._procs.append(p)
                self._conns.append(parent)
        else:
            self.arr = {k: np.zeros(s, dtype=t) for k, (s, t) in _shapes(max_batch, self.n).items()}

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        B = q.shape[0]
        if B > self.max_batch:
            parts = [self.fk(q[i : i + self.max_batch]) for i in range(0, B, self.max_batch)]
            return tuple(np.concatenate(x) for x in zip(*parts))  # type: ignore[return-value]
        arr = self.arr
        arr["q"][:B] = q
        if self.num_workers == 0 or B < 64:
            _compute_rows(self.kin, self.data, arr, 0, B)
        else:
            bounds = np.linspace(0, B, self.num_workers + 1).astype(int)
            for c, lo, hi in zip(self._conns, bounds[:-1], bounds[1:]):
                c.send((int(lo), int(hi)))
            for c in self._conns:
                c.recv()
        return arr["pos"][:B].copy(), arr["rot"][:B].copy(), arr["jac"][:B].copy(), arr["col"][:B].copy()

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        self._procs, self._conns = [], []
        for s in self._shms.values():
            s.close()
            s.unlink()
        self._shms = {}


class _WarpBucket:
    """MuJoCo Warp data for `nworld` worlds and the captured graph: kinematics, collision, contact flags."""

    def __init__(self, wp, mjw, model, mjm: mujoco.MjModel, nworld: int, qpos0: np.ndarray, stream, flag_kernel):
        self.nworld = nworld
        self.naconmax = nworld * CONTACTS_PER_WORLD
        self.d = mjw.make_data(mjm, nworld=nworld, naconmax=self.naconmax)
        self.d.qpos.assign(np.tile(qpos0, (nworld, 1)).astype(np.float32))
        self.flags = wp.zeros(nworld, dtype=wp.int32)

        def run():
            self.flags.zero_()
            mjw.kinematics(model, self.d)
            mjw.collision(model, self.d)
            wp.launch(flag_kernel, dim=self.naconmax, inputs=[self.d.nacon, self.d.contact.worldid, self.flags])

        with wp.ScopedStream(stream):
            run()  # compile / warm up outside the capture
            with wp.ScopedCapture() as cap:
                run()
        self.graph = cap.graph
        self.qpos = wp.to_torch(self.d.qpos)  # (nworld, nq) views on the Warp buffers
        self.site_xpos = wp.to_torch(self.d.site_xpos)  # (nworld, nsite, 3)
        self.site_xmat = wp.to_torch(self.d.site_xmat)  # (nworld, nsite, 3, 3)
        self.xanchor = wp.to_torch(self.d.xanchor)  # (nworld, njnt, 3)
        self.xaxis = wp.to_torch(self.d.xaxis)
        self.col = wp.to_torch(self.flags)
        self.nacon = wp.to_torch(self.d.nacon)


class BatchKinematics:
    """fk for a batch (B, n): TCP position (B, 3), rotation (B, 3, 3), Jacobian (B, 6, n), collision (B,)."""

    def __init__(
        self,
        robot: RobotSpec,
        backend: str | None = None,
        num_workers: int = 0,
        max_batch: int = 8192,
    ) -> None:
        if backend is None:
            backend = "warp" if torch.cuda.is_available() else "cpu"
        if backend not in ("warp", "cpu"):
            raise ValueError(f"backend must be 'warp' or 'cpu', got {backend!r}")
        self.backend = backend
        self.kin = RobotKinematics(robot)
        self.n = self.kin.n
        self.max_batch = max_batch
        self.device = torch.device("cuda" if backend == "warp" else "cpu")
        # CPU path: all rows of the cpu backend, small batches of the warp backend
        self._cpu = _CPUPool(robot, self.kin, num_workers if backend == "cpu" else 0,
                             max_batch if backend == "cpu" else CPU_MAX_BATCH)
        self._data = self._cpu.data  # MjData for single-configuration queries (solver, DLS refinement)
        self._buckets: dict[int, _WarpBucket] = {}
        self._calls = 0
        if backend == "warp":
            self._init_warp()

    # ---------------------------------------------------------------- warp backend
    def _init_warp(self) -> None:
        import warp as wp

        wp.config.quiet = True
        import mujoco_warp as mjw

        m = self.kin.model
        hinge = m.jnt_type[self._arm_jnt_ids()] == mujoco.mjtJoint.mjJNT_HINGE
        if not hinge.all():
            raise ValueError("warp backend: the Jacobian assumes hinge arm joints")
        self._wp, self._mjw = wp, mjw
        self._model = mjw.put_model(m)
        qpos0 = m.qpos0.copy()
        qpos0[self.kin.fixed_qadr] = self.kin.fixed_val
        self._qpos0 = qpos0
        self._torch_stream = torch.cuda.Stream()
        self._stream = wp.stream_from_torch(self._torch_stream)

        @wp.kernel
        def flag_kernel(nacon: wp.array(dtype=wp.int32), worldid: wp.array(dtype=wp.int32),
                        flags: wp.array(dtype=wp.int32)):
            i = wp.tid()
            if i < nacon[0]:
                flags[worldid[i]] = 1

        self._flag_kernel = flag_kernel
        dev = self.device
        self._arm_qadr_t = torch.as_tensor(self.kin.arm_qadr, device=dev)
        self._arm_jnt_t = torch.as_tensor(self._arm_jnt_ids(), device=dev)

    def _arm_jnt_ids(self) -> np.ndarray:
        m = self.kin.model
        return np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.kin.robot.arm_joints])

    def _bucket(self, B: int) -> _WarpBucket:
        """Smallest existing bucket that fits B rows, else a new one (GPU memory grows with the world count:
        ~0.8 GB for 4096 Franka worlds, so buckets are reused rather than created per batch size)."""
        fits = [s for s in self._buckets if s >= B]
        if fits:
            return self._buckets[min(fits)]
        size = next((s for s in WARP_BUCKETS if s >= B), None)
        if size is None:
            raise ValueError(f"batch of {B} rows exceeds the largest Warp bucket {WARP_BUCKETS[-1]}")
        if size not in self._buckets:
            self._buckets[size] = _WarpBucket(self._wp, self._mjw, self._model, self.kin.model, size, self._qpos0,
                                              self._stream, self._flag_kernel)
        return self._buckets[size]

    def _fk_warp(self, q: torch.Tensor, jac: bool):
        B = q.shape[0]
        b = self._bucket(B)
        cur = torch.cuda.current_stream()
        b.qpos[:B, self._arm_qadr_t] = q.to(torch.float32)
        self._torch_stream.wait_stream(cur)
        self._wp.capture_launch(b.graph, stream=self._stream)
        cur.wait_stream(self._torch_stream)
        sid = self.kin.ee_site_id
        p = b.site_xpos[:B, sid].clone()
        R = b.site_xmat[:B, sid].clone()
        col = b.col[:B].bool()
        J = None
        if jac:
            anchor = b.xanchor[:B, self._arm_jnt_t]  # (B, n, 3)
            axis = b.xaxis[:B, self._arm_jnt_t]
            jp = torch.linalg.cross(axis, p[:, None, :] - anchor, dim=-1)
            J = torch.cat([jp.transpose(1, 2), axis.transpose(1, 2)], dim=1)  # (B, 6, n)
        self._calls += 1
        if self._calls % 50 == 0 and int(b.nacon[0].item()) >= b.naconmax:
            raise RuntimeError(f"MuJoCo Warp contact buffer full ({b.naconmax}); raise CONTACTS_PER_WORLD")
        return p, R, J, col

    # ---------------------------------------------------------------- interface
    def fk_torch(self, q: torch.Tensor, jac: bool = False):
        """(p, R, J or None, collided) as tensors on q's device."""
        B = q.shape[0]
        if self.backend == "warp" and B > CPU_MAX_BATCH:
            if B > WARP_BUCKETS[-1]:
                parts = [self.fk_torch(q[i : i + WARP_BUCKETS[-1]], jac) for i in range(0, B, WARP_BUCKETS[-1])]
                return tuple(None if x[0] is None else torch.cat(x) for x in zip(*parts))
            p, R, J, col = self._fk_warp(q.to(self.device), jac)
            out = (p, R, J, col)
            return tuple(None if x is None else x.to(q.device) for x in out)
        p, R, J, col = self._cpu.fk(q.detach().cpu().numpy().astype(np.float32))
        dev = q.device
        return (torch.as_tensor(p, device=dev), torch.as_tensor(R, device=dev),
                torch.as_tensor(J, device=dev) if jac else None, torch.as_tensor(col, device=dev))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """numpy interface: (p, R, J, collided)."""
        q = np.asarray(q, dtype=np.float32)
        if self.backend == "cpu" or q.shape[0] <= CPU_MAX_BATCH:
            return self._cpu.fk(q)
        out = self.fk_torch(torch.as_tensor(q, device=self.device), jac=True)
        return tuple(x.cpu().numpy() for x in out)  # type: ignore[return-value]

    def close(self) -> None:
        self._cpu.close()
        self._buckets = {}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def sample_valid_configs(
    bk: BatchKinematics,
    count: int,
    gen: torch.Generator,
    home_prob: float = 0.0,
    workspace: bool = False,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`count` collision-free configurations (q, TCP pos, rot) as tensors, as RobotKinematics' samplers accept them.

    workspace=True: targets (uniform in the joint box, TCP inside the workspace box);
    otherwise start configurations (home_prob near the home pose, else uniform; TCP above start_pos_min).
    `gen` is a torch.Generator on `device` (default: the backend's device).
    """
    kin = bk.kin
    dev = torch.device(device) if device is not None else bk.device
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)  # noqa: E731
    lo, hi, qmin, qmax, home = t(kin.sample_low), t(kin.sample_high), t(kin.q_min), t(kin.q_max), t(kin.home_q)
    ws_lo, ws_hi, smin = t(kin.ws_low), t(kin.ws_high), t(kin.start_min)
    out_q, out_p, out_R = [], [], []
    have = 0
    chunk = min(bk.max_batch, 16384) if bk.backend == "warp" else 4096  # reuses the env's Warp bucket
    while have < count:
        m = chunk
        q = lo + (hi - lo) * torch.rand((m, kin.n), generator=gen, device=dev)
        near = torch.rand(m, generator=gen, device=dev) < home_prob
        noise = kin.robot.home_noise_std * torch.randn((m, kin.n), generator=gen, device=dev)
        q = torch.where(near[:, None], torch.clamp(home + noise, qmin, qmax), q)
        p, R, _, col = bk.fk_torch(q)
        if workspace:
            ok = ~col & ((p >= ws_lo) & (p <= ws_hi)).all(dim=1)
        else:
            ok = ~col & (p >= smin).all(dim=1)
        idx = torch.nonzero(ok).squeeze(1)[: count - have]
        out_q.append(q[idx])
        out_p.append(p[idx])
        out_R.append(R[idx])
        have += len(idx)
    return torch.cat(out_q), torch.cat(out_p), torch.cat(out_R)
