"""PolicyIK step, observation and reward for a batch of arms (kinematic transition), in torch.

`PolicyIKBatch` is the single implementation used by training (`env.py`) and inference (`controllers.py`,
`solver.py`); all state lives on the kinematics backend's device (GPU with the Warp backend). Per row it keeps
the joint configuration, the TCP pose, the goal, the previous action and joint velocity, and the
tolerance-normalized error state of PolicyIK:

    rho_t = max(e_p / tol_pos, e_R / tol_rot)         (both errors in tolerance units)
    b_t   = min(b_{t-1}, rho_t)                       (running best, reset whenever the goal changes)

Differences from the Isaac Lab task (see README.md): joints move kinematically (q_{t+1} = z_t unless the new
configuration collides, then the arm stays and z is re-anchored at q), so the command-state channel
d_t = z_{t-1} - q_t, which PolicyIK uses to keep the PD controller's steady-state offset, is identically zero
and omitted; joint velocity is the executed step / dt.
"""

from __future__ import annotations

import numpy as np
import torch

from core.batch_kinematics import BatchKinematics
from policy.policyik.config import PolicyIKConfig

Tensor = torch.Tensor


def so3_log(R_cur: Tensor, R_goal: Tensor) -> tuple[Tensor, Tensor]:
    """World-frame rotation vector w with Exp(w) R_cur = R_goal, and its angle in [0, pi] (cf. so3_error_vec)."""
    E = R_goal @ R_cur.transpose(-1, -2)
    v = torch.stack([E[..., 2, 1] - E[..., 1, 2], E[..., 0, 2] - E[..., 2, 0], E[..., 1, 0] - E[..., 0, 1]], -1)
    nv = v.norm(dim=-1)
    tr1 = E[..., 0, 0] + E[..., 1, 1] + E[..., 2, 2] - 1.0
    ang = torch.atan2(nv, tr1)
    scale = torch.where(nv > 1e-6, ang / nv.clamp(min=1e-6), torch.full_like(nv, 0.5))
    w = v * scale[..., None]
    # sin(angle) ~ 0 at angle ~ pi: axis from (E + I) = 2 u u^T (computed for all rows: no host sync)
    near_pi = (nv <= 1e-4) & (tr1 < 0.0)
    Epi = E + torch.eye(3, device=E.device, dtype=E.dtype)
    k = torch.diagonal(Epi, dim1=-2, dim2=-1).argmax(-1)
    col = torch.gather(Epi, -1, k[..., None, None].expand(*k.shape, 3, 1))[..., 0]
    w_pi = torch.pi * col / col.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # the column gives the axis up to sign; take the sign of v (the general formula's direction) where defined
    sgn = torch.where((w_pi * v).sum(-1) < 0, -1.0, 1.0)
    w_pi = w_pi * sgn[..., None]
    w = torch.where(near_pi[..., None], w_pi, w)
    ang = torch.where(near_pi, torch.full_like(ang, torch.pi), ang)
    return w, ang


def pose_errors(p: Tensor, R: Tensor, p_g: Tensor, R_g: Tensor):
    """Position error (B,), rotation angle (B,), and both errors in the TCP frame (B, 3) each.

    The rotation error in the TCP frame is Log(R^T R_goal) = R^T Log(R_goal R^T).
    """
    dp = p_g - p
    w, ang = so3_log(R, R_g)
    Rt = R.transpose(-1, -2)
    return dp.norm(dim=-1), ang, (Rt @ dp[..., None])[..., 0], (Rt @ w[..., None])[..., 0]


def keypoint_offsets(p, R, p_g, R_g, size: float, centered: bool) -> Tensor:
    """Goal-minus-current positions (B, 3, 3) of three points at size * e_i (optionally centered) on the TCP."""
    c = torch.eye(3, device=p.device, dtype=p.dtype) * size
    if centered:
        c = c - c.mean(dim=0, keepdim=True)
    # point i: p + R @ c_i
    cur = p[:, None, :] + torch.einsum("bjk,ik->bij", R, c)
    goal = p_g[:, None, :] + torch.einsum("bjk,ik->bij", R_g, c)
    return goal - cur


def ramp(e: Tensor, lo: float, hi: float, g_min: float) -> Tensor:
    return g_min + (1.0 - g_min) * ((e - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0)


class PolicyIKBatch:
    """State of B arms tracking B goals under the PolicyIK action / observation / reward."""

    def __init__(self, bk: BatchKinematics, cfg: PolicyIKConfig, device: torch.device | str | None = None) -> None:
        self.bk = bk
        self.kin = bk.kin
        self.cfg = cfg
        self.n = self.kin.n
        self.device = torch.device(device) if device is not None else bk.device
        f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)  # noqa: E731
        self.q_min, self.q_max, self.home_q = f(self.kin.q_min), f(self.kin.q_max), f(self.kin.home_q)
        self._q_range = (self.q_max - self.q_min).clamp(min=1e-6)
        self.B = 0

    def _t(self, x) -> Tensor:
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    # ---------------------------------------------------------------- state
    def reset(self, q, p_goal, R_goal) -> None:
        """Start all rows at q (B, n) (assumed collision-free) with goals (B, 3) / (B, 3, 3)."""
        q = torch.clamp(self._t(q), self.q_min, self.q_max)
        self.B = q.shape[0]
        self._err = None
        self.q = q.clone()
        self.p, self.R, _, _ = self.bk.fk_torch(self.q)
        z = lambda *s: torch.zeros(*s, device=self.device)  # noqa: E731
        self.prev_action = z(self.B, self.n)
        self.qdot = z(self.B, self.n)
        self.collided = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.p_goal, self.R_goal = z(self.B, 3), z(self.B, 3, 3)
        self.best, self.phi_prev, self.rho = z(self.B), z(self.B), z(self.B)
        self.set_goal(torch.arange(self.B, device=self.device), p_goal, R_goal)

    def reset_rows(self, rows: Tensor, q: Tensor, p: Tensor, R: Tensor) -> None:
        """Put rows at new start configurations with known FK (goals are set separately with set_goal)."""
        self.q[rows], self.p[rows], self.R[rows] = q, p, R
        self.prev_action[rows] = 0.0
        self.qdot[rows] = 0.0
        self.collided[rows] = False
        self._err = None

    def set_goal(self, rows, p_goal, R_goal) -> None:
        """New goals for `rows`; their error state restarts (b = rho, Phi_prev = Phi) without moving the arm."""
        rows = torch.as_tensor(rows, device=self.device)
        self.p_goal[rows] = self._t(p_goal)
        self.R_goal[rows] = self._t(R_goal)
        self._err = None
        e_p, e_r, _, _ = pose_errors(self.p[rows], self.R[rows], self.p_goal[rows], self.R_goal[rows])
        rho = torch.maximum(e_p / self.cfg.tol_pos, e_r / self.cfg.tol_rot)
        self.rho[rows] = rho
        self.best[rows] = rho
        self.phi_prev[rows] = -rho.clamp(max=self.cfg.potential_cap)

    # ---------------------------------------------------------------- observation
    def obs(self, gen: torch.Generator | None = None) -> Tensor:
        """o_t = [q - q_home, qdot, p_goal, keypoint offsets, coarse / fine pose channels, h_t, a_{t-1}]."""
        c = self.cfg
        _, _, e_pb, e_rb = self._errors()
        kp = keypoint_offsets(self.p, self.R, self.p_goal, self.R_goal, c.obs_kp_size, centered=False)
        q_rel = self.q - self.home_q
        qdot = self.qdot
        if gen is not None and c.obs_noise > 0:  # training only
            u = lambda x: (2 * torch.rand(x.shape, generator=gen, device=self.device) - 1) * c.obs_noise  # noqa: E731
            q_rel, qdot = q_rel + u(q_rel), qdot + u(qdot)
        hist = torch.stack(
            [(self.best / c.hist_scale).clamp(0, 1), ((self.rho - self.best) / c.hist_scale).clamp(0, 1)], dim=1
        )
        return torch.cat(
            [
                q_rel,
                qdot,
                self.p_goal,
                kp.reshape(self.B, 9),
                (e_pb / c.coarse_pos).clamp(-1, 1),
                (e_rb / c.coarse_rot).clamp(-1, 1),
                (e_pb / c.fine_pos).clamp(-1, 1),
                (e_rb / c.fine_rot).clamp(-1, 1),
                hist,
                self.prev_action,
            ],
            dim=1,
        )

    # ---------------------------------------------------------------- action
    def _errors(self):
        """pose_errors of the current state, computed once per state (invalidated by every state change)."""
        if self._err is None:
            self._err = pose_errors(self.p, self.R, self.p_goal, self.R_goal)
        return self._err

    def errors(self) -> tuple[Tensor, Tensor]:
        e_p, e_r, _, _ = self._errors()
        return e_p, e_r

    def gain(self) -> Tensor:
        """beta(e_p, e_R) = max(G(e_p), G(e_R)) in [beta_min, 1] from the current pose error."""
        c = self.cfg
        e_p, e_r = self.errors()
        return torch.maximum(
            ramp(e_p, c.gain_pos_lo, c.gain_pos_hi, c.gain_min), ramp(e_r, c.gain_rot_lo, c.gain_rot_hi, c.gain_min)
        )

    def step(self, action, motion_weights: tuple[float, float] | None = None) -> dict[str, Tensor]:
        """Apply actions (B, n); with motion_weights also returns the reward (B,) and its terms."""
        c = self.cfg
        a = self._t(action).clamp(-c.clip_actions, c.clip_actions)
        z = torch.clamp(self.q + (c.action_scale * self.gain())[:, None] * a, self.q_min, self.q_max)
        p, R, _, col = self.bk.fk_torch(z)
        ok = ~col
        q_old = self.q
        self.q = torch.where(ok[:, None], z, self.q)
        self.p = torch.where(ok[:, None], p, self.p)
        self.R = torch.where(ok[:, None, None], R, self.R)
        self.qdot = (self.q - q_old) / c.dt
        prev_action, self.prev_action = self.prev_action, a
        self.collided = col
        self._err = None

        e_p, e_r, _, _ = self._errors()
        rho = torch.maximum(e_p / c.tol_pos, e_r / c.tol_rot)
        best_before = self.best
        self.rho = rho
        self.best = torch.minimum(best_before, rho)
        phi = -rho.clamp(max=c.potential_cap)
        phi_prev, self.phi_prev = self.phi_prev, phi
        out = {"e_p": e_p, "e_r": e_r, "rho": rho, "collided": col}
        if motion_weights is None:
            return out

        U = lambda r: 0.5 * (1.0 + torch.tanh((1.0 - r) / c.improve_beta))  # noqa: E731
        kp_err = keypoint_offsets(self.p, self.R, self.p_goal, self.R_goal, c.reward_kp_size, centered=True)
        kp_err = kp_err.norm(dim=-1).mean(dim=1)
        margin = (torch.minimum(self.q - self.q_min, self.q_max - self.q) / self._q_range).min(dim=1).values
        margin_pen = (1.0 - margin / c.joint_margin_eps).clamp(0.0, 1.0) * (rho > c.joint_margin_gate)
        w_rate, w_vel = motion_weights
        terms = {
            "pos_l2": c.w_pos_l2 * e_p,
            "pos_fine": c.w_pos_fine * (1.0 - torch.tanh(e_p / c.pos_fine_std)),
            "kp_wide": c.w_kp_wide * (1.0 - torch.tanh(kp_err / c.kp_wide_std)),
            "kp_sharp": c.w_kp_sharp * (1.0 - torch.tanh(kp_err / c.kp_sharp_std)),
            "improve": c.w_improve * (U(rho) - U(best_before)).clamp(min=0.0),
            "potential": c.w_potential * (c.potential_gamma * phi - phi_prev),
            "backslide": -c.w_backslide * (rho - best_before).clamp(min=0.0).clamp(max=c.backslide_cap),
            "dwell": c.w_dwell * (rho <= 1.0).float(),
            "joint_margin": -c.w_joint_margin * margin_pen,
            "action_rate": -w_rate * (a - prev_action).square().sum(dim=1),
            "joint_vel": -w_vel * self.qdot.square().sum(dim=1),
            "collision": -c.w_collision * col.float(),
        }
        out["reward"] = c.dt * sum(terms.values())
        out["terms"] = terms
        return out
