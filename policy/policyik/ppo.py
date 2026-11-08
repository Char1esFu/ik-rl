"""PPO as configured in PolicyIK (rl_games a2c_continuous): shared ELU trunk, state-independent log std,
input and value normalization, clipped value loss, GAE, adaptive learning rate (KL), bounds loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from policy.policyik.config import PPOConfig


class RunningMeanStd(nn.Module):
    """Running mean / variance (parallel Welford), normalization clipped to +-5 as in rl_games."""

    def __init__(self, shape: tuple[int, ...], clip: float = 5.0, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))
        self.clip = clip
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, *self.mean.shape).double()
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta.square() * self.count * b_n / tot) / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self.mean.float()) / torch.sqrt(self.var.float() + self.eps)
        return y.clamp(-self.clip, self.clip) if self.clip else y

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.sqrt(self.var.float() + self.eps) + self.mean.float()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(d, act_dim)
        self.value = nn.Linear(d, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.value_rms = RunningMeanStd((1,), clip=0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized obs -> (mu, normalized value)."""
        h = self.trunk(self.obs_rms.normalize(obs))
        return self.mu(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mu, v = self(obs)
        if deterministic:
            return mu
        std = self.log_std.exp().expand_as(mu)
        a = mu + std * torch.randn_like(mu)
        logp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)
        return a, logp, mu, v


class PPOAgent:
    """Policy for inference (deterministic mean action) + PPO update."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.net = ActorCritic(obs_dim, act_dim, cfg).to(self.device)
        self.lr = cfg.learning_rate
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-8)

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def act_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic (mean) action for a tensor observation, on obs's device."""
        return self.net.act(obs.to(self.device, torch.float32), deterministic=True).to(obs.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:
            return self.net.act(o, deterministic=True).cpu().numpy()
        return self.net.act(o)[0].cpu().numpy()

    # ---------------------------------------------------------------- training
    def update(self, buf: dict[str, torch.Tensor]) -> dict[str, float]:
        """One PPO epoch on a rollout: buf has obs (T,N,D), act, logp, mu, value (normalized), reward, last_value."""
        c = self.cfg
        net = self.net
        T, N = buf["reward"].shape
        values = net.value_rms.denormalize(buf["value"].unsqueeze(-1)).squeeze(-1)
        last_value = net.value_rms.denormalize(buf["last_value"].unsqueeze(-1)).squeeze(-1)
        adv = torch.zeros_like(buf["reward"])
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            delta = buf["reward"][t] + c.gamma * next_v - values[t]
            gae = delta + c.gamma * c.gae_lambda * gae
            adv[t] = gae
        returns = adv + values
        net.value_rms.update(returns.reshape(-1, 1))
        ret_n = net.value_rms.normalize(returns.unsqueeze(-1)).squeeze(-1).reshape(-1)
        old_v = net.value_rms.normalize(values.unsqueeze(-1)).squeeze(-1).reshape(-1)
        adv = adv.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs = buf["obs"].reshape(T * N, -1)
        act = buf["act"].reshape(T * N, -1)
        old_logp = buf["logp"].reshape(-1)
        old_mu = buf["mu"].reshape(T * N, -1)
        old_std = buf["std"]

        M = T * N
        mb = min(c.minibatch_size, M)
        stats = {"a_loss": 0.0, "c_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        n_mb = 0
        for _ in range(c.mini_epochs):
            perm = torch.randperm(M, device=self.device)
            for s in range(0, M - mb + 1, mb):
                idx = perm[s : s + mb]
                mu, v = net(obs[idx])
                log_std = net.log_std.expand_as(mu)
                std = log_std.exp()
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(act[idx]).sum(-1)
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                a_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - c.e_clip, 1 + c.e_clip)).mean()
                if c.clip_value:
                    v_clip = old_v[idx] + (v - old_v[idx]).clamp(-c.e_clip, c.e_clip)
                    c_loss = torch.max((v - ret_n[idx]).square(), (v_clip - ret_n[idx]).square()).mean()
                else:
                    c_loss = (v - ret_n[idx]).square().mean()
                entropy = dist.entropy().sum(-1).mean()
                b_loss = ((mu - 1.1).clamp(min=0).square() + (-mu - 1.1).clamp(min=0).square()).sum(-1).mean()
                loss = a_loss + 0.5 * c.critic_coef * c_loss - c.entropy_coef * entropy + c.bounds_loss_coef * b_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), c.grad_norm)
                self.opt.step()
                with torch.no_grad():  # KL(old || new) of diagonal Gaussians, adaptive LR per minibatch
                    o_mu, o_std = old_mu[idx], old_std.expand_as(mu)
                    kl = (
                        torch.log(std / o_std) + (o_std.square() + (o_mu - mu).square()) / (2 * std.square()) - 0.5
                    ).sum(-1).mean().item()
                if kl > 2.0 * c.kl_threshold:
                    self.lr = max(self.lr / 1.5, c.lr_min)
                elif kl < 0.5 * c.kl_threshold:
                    self.lr = min(self.lr * 1.5, c.lr_max)
                for g in self.opt.param_groups:
                    g["lr"] = self.lr
                stats["a_loss"] += a_loss.item()
                stats["c_loss"] += c_loss.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += kl
                stats["clip_frac"] += ((ratio - 1).abs() > c.e_clip).float().mean().item()
                n_mb += 1
        net.obs_rms.update(obs)  # after the epoch: old log-probs and the update use the same normalization
        out = {k: v / max(n_mb, 1) for k, v in stats.items()}
        out["lr"] = self.lr
        out["std"] = float(net.log_std.exp().mean().item())
        return out

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path, meta: dict[str, Any], state_dict: dict | None = None) -> None:
        torch.save(
            {
                "method": "policyik",
                "model": state_dict if state_dict is not None else self.net.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "ppo_config": self.cfg.to_dict(),
                **meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> tuple[PPOAgent, dict]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PPOConfig.from_dict(ckpt["ppo_config"])
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], cfg, device=device)
        agent.net.load_state_dict(ckpt["model"])
        agent.net.eval()
        return agent, ckpt


def average_state_dicts(states: list[dict], weights: list[float]) -> dict:
    """Weighted average of floating-point tensors (integer buffers must agree), as PolicyIK's averaging."""
    w = torch.tensor(weights, dtype=torch.float64)
    w = (w / w.sum()).tolist()
    out = {}
    for k, v0 in states[0].items():
        if v0.dtype.is_floating_point:
            acc = sum(wi * s[k].double() for wi, s in zip(w, states))
            out[k] = acc.to(v0.dtype)
        else:
            out[k] = v0.clone()
    return out
