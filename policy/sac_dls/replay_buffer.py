"""GPU-Resident Replay Buffer for SAC."""

from __future__ import annotations

import numpy as np
import torch


class GPUReplayBuffer:
    """High-throughput ring buffer stored directly in GPU VRAM for zero-copy SAC sampling."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = 2_000_000,
        device: str | torch.device = "cuda",
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = capacity
        self.device = torch.device(device)

        self.obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, act_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        if n == 0:
            return

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(n, 1)
        nobs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(terminated, dtype=torch.float32, device=self.device).reshape(n, 1)

        if self.ptr + n <= self.capacity:
            sl = slice(self.ptr, self.ptr + n)
            self.obs[sl] = obs_t
            self.actions[sl] = act_t
            self.rewards[sl] = rew_t
            self.next_obs[sl] = nobs_t
            self.dones[sl] = done_t
        else:
            first = self.capacity - self.ptr
            second = n - first
            self.obs[self.ptr :] = obs_t[:first]
            self.actions[self.ptr :] = act_t[:first]
            self.rewards[self.ptr :] = rew_t[:first]
            self.next_obs[self.ptr :] = nobs_t[:first]
            self.dones[self.ptr :] = done_t[:first]

            self.obs[:second] = obs_t[first:]
            self.actions[:second] = act_t[first:]
            self.rewards[:second] = rew_t[first:]
            self.next_obs[:second] = nobs_t[first:]
            self.dones[:second] = done_t[first:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def state_dict(self) -> dict:
        """Filled part of the buffer plus ring pointer (for resuming training)."""
        n = self.size
        return {
            "capacity": self.capacity,
            "ptr": self.ptr,
            "size": n,
            # .cpu() copies only the filled rows (saving a GPU slice would serialize the whole storage)
            "obs": self.obs[:n].cpu(),
            "actions": self.actions[:n].cpu(),
            "rewards": self.rewards[:n].cpu(),
            "next_obs": self.next_obs[:n].cpu(),
            "dones": self.dones[:n].cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity or state["obs"].shape[1] != self.obs_dim:
            raise ValueError(
                f"Replay buffer mismatch: saved capacity {state['capacity']} / obs_dim {state['obs'].shape[1]}, "
                f"current {self.capacity} / {self.obs_dim}"
            )
        n = int(state["size"])
        for name in ("obs", "actions", "rewards", "next_obs", "dones"):
            getattr(self, name)[:n] = state[name].to(self.device)
        self.ptr = int(state["ptr"])
        self.size = n

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )
