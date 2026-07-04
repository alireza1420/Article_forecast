"""PG-S2S input-selection agent (feature 007): PolicyNet MLP (paper eq. 19),
reward (eq. 3–5, log1p space, shared-better-rank ties), discounted returns,
and the REINFORCE update (FR-4/5/6; research D-006/D-007).

Every numerical component is gated by hand-computed toy tests (FR-14) before
any demand-data training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(state_dim, hidden)     # W₁ ∈ R^{N_p×state_dim}
        self.fc2 = nn.Linear(hidden, n_actions)     # W₂ ∈ R^{n_actions×N_p}
        self.n_actions = n_actions

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """(B, state_dim) → (B, n_actions) softmax action probabilities."""
        return F.softmax(self.fc2(F.relu(self.fc1(s))), dim=-1)

    def act(self, s: torch.Tensor, epsilon: float,
            generator: torch.Generator | None = None) -> torch.Tensor:
        """ε-greedy argmax action selection (eq. 20); ε=0 ⇒ pure greedy."""
        with torch.no_grad():
            greedy = self.forward(s).argmax(dim=-1)
        if epsilon <= 0.0:
            return greedy
        batch = s.shape[0]
        explore = torch.rand(batch, generator=generator, device=s.device) < epsilon
        random_actions = torch.randint(0, self.n_actions, (batch,),
                                       generator=generator, device=s.device)
        return torch.where(explore, random_actions, greedy)


def compute_rewards(dec_preds_log1p: torch.Tensor, aux_log1p: torch.Tensor,
                    y_log1p: torch.Tensor, actions: torch.Tensor,
                    alpha: float = ALPHA, beta: float = BETA) -> torch.Tensor:
    """Paper eq. 3–5 in log1p space, (B,10); ties share the better rank; Accuracy_r_H = 0.

    Candidate m's error at step k (D-006): |est_m(y_{t+k}) − y_{t+k}| where the
    decoder candidate is its own ŷ_{t+k} and xgb/lgbm come from cache column k−1.
    """
    batch, horizon = dec_preds_log1p.shape
    # candidate errors (B, 3, H): order matches action ids {0: dec, 1: xgb, 2: lgbm}
    cand = torch.stack([dec_preds_log1p, aux_log1p[:, 0, :], aux_log1p[:, 1, :]], dim=1)
    errors = (cand - y_log1p.unsqueeze(1)).abs()

    # rank of the TAKEN action: 1 + count(strictly smaller errors) → ties share
    # the better (smaller) rank (FR-4): (0.2, 0.2, 0.5) → ranks (1, 1, 3)
    taken_err = errors.gather(1, actions.unsqueeze(1)).squeeze(1)        # (B, H)
    rank = 1 + (errors < taken_err.unsqueeze(1)).sum(dim=1)              # (B, H)
    rank_r = 1.0 - rank.to(errors.dtype) / N_ACTIONS                     # eq. 4

    # Accuracy_r_k = β/(β+|y_{t+k+1} − ŷ_{t+k+1}|) — on-policy decoder error at
    # the NEXT step, log1p space; Accuracy_r_H = 0 (eq. 5, D-005/D-006)
    acc_r = torch.zeros_like(rank_r)
    next_err = (y_log1p[:, 1:] - dec_preds_log1p[:, 1:]).abs()           # steps 2..H
    acc_r[:, :-1] = beta / (beta + next_err)

    return alpha * rank_r + (1.0 - alpha) * acc_r                        # eq. 3


def discounted_returns(rewards: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """G_k = Σ_{k'≥k} γ^(k'−k) r_k', computed back-to-front; (B, H)."""
    returns = torch.zeros_like(rewards)
    returns[:, -1] = rewards[:, -1]
    for j in range(rewards.shape[1] - 2, -1, -1):
        returns[:, j] = rewards[:, j] + gamma * returns[:, j + 1]
    return returns


def step_discounts(horizon: int, gamma: float = GAMMA) -> torch.Tensor:
    """γ^k for steps k = 1..H (paper Alg. 1 line 9, 1-based k) — pinned by toy test."""
    return gamma ** torch.arange(1, horizon + 1, dtype=torch.float32)


def reinforce_update(policy: PolicyNet, opt: torch.optim.Optimizer,
                     states: torch.Tensor, actions: torch.Tensor,
                     returns: torch.Tensor, gamma: float = GAMMA) -> float:
    """One ascent step on Σ_k γ^k G_k log π(a_k|s_k) (paper Alg. 1 line 9, k = 1..H).

    Implemented as minimising the negative objective with the supplied optimizer.
    states MUST be detached (FR-6) — refused otherwise.
    """
    if states.requires_grad or states.grad_fn is not None:
        raise ValueError(
            "[pgs2s] reinforce_update: states carry gradient history — they must be "
            "detached from the seq2seq graph before policy updates (FR-6)")

    batch, horizon, state_dim = states.shape
    probs = policy(states.reshape(batch * horizon, state_dim))
    log_pi = torch.log(probs.gather(1, actions.reshape(-1, 1)).clamp_min(1e-12))
    log_pi = log_pi.reshape(batch, horizon)

    discount = step_discounts(horizon, gamma).to(dtype=states.dtype, device=states.device)
    loss = -(discount * returns * log_pi).sum(dim=1).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


if __name__ == "__main__":
    torch.manual_seed(SEED)
    policy = PolicyNet()
    s = torch.randn(4, STATE_DIM)
    probs = policy(s)
    assert probs.shape == (4, N_ACTIONS)
    a = policy.act(s, epsilon=0.0)
    assert torch.equal(a, probs.argmax(dim=-1))
    r = compute_rewards(torch.rand(4, 10) + 2, torch.rand(4, 2, 10) + 2,
                        torch.rand(4, 10) + 2, torch.zeros(4, 10, dtype=torch.int64))
    g = discounted_returns(r)
    assert torch.equal(g[:, -1], r[:, -1])
    print("[pgs2s_agent] smoke OK")
