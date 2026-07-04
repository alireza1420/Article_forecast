"""PG-S2S input-selection agent (feature 007): PolicyNet MLP (paper eq. 19),
reward (eq. 3–5, log1p space, shared-better-rank ties), discounted returns,
and the REINFORCE update (FR-4/5/6; research D-006/D-007).

Every numerical component is gated by hand-computed toy tests (FR-14) before
any demand-data training.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ── RL constants (research D-008; single source, imported by pgs2s_train) ─────

ALPHA: float = 0.5        # reward mix: α·Rank + (1−α)·Accuracy (eq. 3)
BETA: float = 1.0         # accuracy reward scale β/(β+|err|) (eq. 5)
GAMMA: float = 0.9        # discount factor
EPS_START: float = 0.5    # ε-greedy exploration start
EPS_DECAY: float = 0.9    # per-round multiplicative ε decay
EPS_MIN: float = 0.05     # ε floor
N_P: int = 64             # policy hidden width (eq. 19)
N_ACTIONS: int = 3        # {0: decoder, 1: xgb, 2: lgbm}
STATE_DIM: int = 128      # decoder hidden size = MDP state dim

SEED: int = 42


class PolicyNet(nn.Module):
    """1-hidden-layer MLP: softmax(W₂ ReLU(W₁ s + b₁) + b₂) over 3 actions (eq. 19)."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = N_P,
                 n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        raise NotImplementedError("T019")


def compute_rewards(dec_preds_log1p: torch.Tensor, aux_log1p: torch.Tensor,
                    y_log1p: torch.Tensor, actions: torch.Tensor,
                    alpha: float = ALPHA, beta: float = BETA) -> torch.Tensor:
    """Paper eq. 3–5 in log1p space, (B,10); ties share the better rank; Accuracy_r_H = 0."""
    raise NotImplementedError("T024")


def discounted_returns(rewards: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """G_k = Σ_{k'≥k} γ^(k'−k) r_k', (B,10)."""
    raise NotImplementedError("T024")


def reinforce_update(policy: "PolicyNet", opt: torch.optim.Optimizer,
                     states: torch.Tensor, actions: torch.Tensor,
                     returns: torch.Tensor, gamma: float = GAMMA) -> float:
    """One ascent step on Σ_k γ^k G_k log π(a_k|s_k); returns loss; states must be detached (FR-6)."""
    raise NotImplementedError("T024")


if __name__ == "__main__":
    print("[pgs2s_agent] smoke OK (implementation lands in T019/T024)")
