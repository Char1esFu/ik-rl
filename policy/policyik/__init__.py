"""PolicyIK: PPO with tolerance-normalized progress / dwell rewards, after Liu et al. 2026 (README.md)."""

from policy.policyik.config import PolicyIKConfig, PPOConfig
from policy.policyik.solver import PolicyIKSolver

METHOD = "policyik"

__all__ = ["PolicyIKConfig", "PPOConfig", "PolicyIKSolver", "METHOD"]
