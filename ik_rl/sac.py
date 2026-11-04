"""Soft Actor-Critic (SAC) with Twin LayerNorm Critics & Automatic Entropy Tuning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ik_rl.config import SACConfig

LOG_STD_MIN = -5.5
LOG_STD_MAX = 2.0


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Construct MLP with LayerNorm + SiLU hidden blocks for stable high-UTD value learning."""
    layers: list[nn.Module] = []
    in_d = input_dim
    for h_d in hidden_dims:
        layers.append(nn.Linear(in_d, h_d))
        layers.append(nn.LayerNorm(h_d))
        layers.append(nn.SiLU())
        in_d = h_d
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Stochastic policy pi(a|s) with tanh squashing and exact log-prob Jacobian correction."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = obs_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.LayerNorm(h_d))
            layers.append(nn.SiLU())
            in_d = h_d
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_d, act_dim)
        self.log_std_head = nn.Linear(in_d, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(obs)
        mu = self.mu_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (squashed_action, log_prob, deterministic_action)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        action = torch.tanh(u)

        # Exact Gaussian log-prob minus tanh Jacobian: 2 * (log(2) - u - softplus(-2u))
        log_norm = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        log_det_jacobian = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        log_prob = (log_norm - log_det_jacobian).sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mu)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        return torch.tanh(u)


class TwinQCritic(nn.Module):
    """Clipped Double Q-Network Q1(s, a), Q2(s, a) with LayerNorm."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q1 = build_mlp(obs_dim + act_dim, hidden_dims, 1)
        self.q2 = build_mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACAgent:
    """Soft Actor-Critic (SAC) Agent optimized for GPU execution (supports torch.compile)."""

    def __init__(
        self,
        cfg: SACConfig | None = None,
        device: str | torch.device = "cuda",
        use_compile: bool = False,
    ) -> None:
        self.cfg = cfg or SACConfig()
        self.device = torch.device(device)

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        self.actor = SquashedGaussianActor(
            self.cfg.obs_dim, self.cfg.act_dim, self.cfg.hidden_dims
        ).to(self.device)
        self.critic = TwinQCritic(
            self.cfg.obs_dim, self.cfg.act_dim, self.cfg.hidden_dims
        ).to(self.device)
        self.critic_target = TwinQCritic(
            self.cfg.obs_dim, self.cfg.act_dim, self.cfg.hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(
            math.log(self.cfg.init_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self._min_log_alpha = math.log(max(self.cfg.min_alpha, 1e-6))

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        if use_compile and self.device.type == "cuda":
            self._compiled_critic = torch.compile(self._critic_loss_fn)
            self._compiled_actor = torch.compile(self._actor_loss_fn)
        else:
            self._compiled_critic = self._critic_loss_fn
            self._compiled_actor = self._actor_loss_fn

    def _critic_loss_fn(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * min_q_next
        q1, q2 = self.critic(obs, actions)
        return F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    def _actor_loss_fn(
        self, obs: torch.Tensor, alpha: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_actions, log_prob, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, pi_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_prob - min_q_pi).mean()
        return actor_loss, log_prob.detach(), min_q_pi.detach()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)
        act_t = self.actor.act(obs_t, deterministic=deterministic)
        if single:
            return act_t[0].cpu().numpy()
        return act_t.cpu().numpy()

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """Perform one SAC update step (Critic + Actor + Entropy Temperature + Target Polyak)."""
        alpha = self.log_alpha.exp().detach()

        # 1. Critic Update
        critic_loss = self._compiled_critic(obs, actions, rewards, next_obs, dones, alpha)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # 2. Actor Update
        for p in self.critic.parameters():
            p.requires_grad_(False)

        actor_loss, log_prob_det, min_q_det = self._compiled_actor(obs, alpha)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        # 3. Automatic Entropy Temperature Update (with exploration floor)
        alpha_loss = -(self.log_alpha * (log_prob_det + self.cfg.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=self._min_log_alpha, max=1.0)
            tau = self.cfg.tau
            torch._foreach_mul_(list(self.critic_target.parameters()), 1.0 - tau)
            torch._foreach_add_(
                list(self.critic_target.parameters()), list(self.critic.parameters()), alpha=tau
            )

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "q_mean": float(min_q_det.mean().item()),
            "entropy": float(-log_prob_det.mean().item()),
        }

    def training_state(self) -> dict[str, Any]:
        """Networks, entropy temperature and optimizer states (for resuming training)."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_training_state(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])

    def save(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> None:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "sac_config": self.cfg.to_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": float(self.log_alpha.item()),
        }
        if extra_meta is not None:
            payload["meta"] = extra_meta
        torch.save(payload, ckpt_path)

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cuda"
    ) -> tuple[SACAgent, dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = SACConfig.from_dict(payload["sac_config"])
        agent = cls(cfg=cfg, device=device, use_compile=False)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.critic_target.load_state_dict(payload["critic_target"])
        with torch.no_grad():
            agent.log_alpha.fill_(float(payload["log_alpha"]))
        return agent, payload.get("meta", {})
