"""PG-S2S seq2seq forecaster (feature 007): LSTM encoder + LSTMCell decoder with
three decoding modes (free_running / teacher_forcing / policy) and per-step
hidden-state exposure (FR-3; research D-003/D-004/D-005).

Device-agnostic nn.Module — the hard CUDA gate lives in the pgs2s_train.py /
pgs2s_eval.py entry points only (research D-015).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pgs2s_data import to_feedback_scale
    from features import SEQUENCE_TEMPORAL_COLS
except ModuleNotFoundError:
    from src.pgs2s_data import to_feedback_scale
    from src.features import SEQUENCE_TEMPORAL_COLS

# ── Architecture constants (research D-003/D-004) ─────────────────────────────

N_FEATURES: int = 23      # Cycle-5 selected temporal features (best_config_c)
HIDDEN: int = 128
ENC_LAYERS: int = 2
ENC_DROPOUT: float = 0.3
HORIZON: int = 10
LOOKBACK: int = 20        # PAPER_LOOKBACK window served to the encoder

SEED: int = 42

# Cycle-5 feature selection recorded in results/models/lstm/best_config_c.pt
# (temporal_cols; static_cols = []). Pinned here so the encoder input is
# reproducible from code alone — the checkpoint is a gitignored artifact.
TEMPORAL_COLS: list[str] = [
    "num_orders", "checkout_price", "discount_rate", "log_checkout_price",
    "week_number", "week_of_year_sin", "week_of_year_cos",
    "center_week_count", "center_cat_week_count", "city_meal_week_count",
    "meal_week_count", "region_meal_week_count", "type_meal_week_count",
    "center_week_price_rank", "meal_week_price_rank",
    "rolling_mean_4w", "rolling_std_4w", "rolling_min_4w", "rolling_max_4w",
    "rolling_mean_8w", "rolling_std_8w", "rolling_min_8w", "rolling_max_8w",
]

TEMPORAL_IDX: list[int] = [SEQUENCE_TEMPORAL_COLS.index(c) for c in TEMPORAL_COLS]
NUM_ORDERS_POS: int = TEMPORAL_COLS.index("num_orders")

_ROOT: Path = Path(__file__).resolve().parent.parent
SEQ_DIR: Path = _ROOT / "data" / "processed" / "sequences"


def encoder_input(x_temporal_full: np.ndarray) -> np.ndarray:
    """Slice a stored (N, 26, 32) X_temporal array to the encoder view (N, 20, 23)."""
    t_full = x_temporal_full.shape[1]
    window = x_temporal_full[:, t_full - LOOKBACK:, :]
    return np.ascontiguousarray(window[:, :, TEMPORAL_IDX], dtype=np.float32)


def seed_from_window(x_encoder: np.ndarray) -> np.ndarray:
    """y_seed = standardized log1p demand at the anchor week (last window position)."""
    return np.ascontiguousarray(x_encoder[:, -1, NUM_ORDERS_POS], dtype=np.float32)


class S2SOutput:
    """Decode pass result: preds (B,10) log1p; states (B,10,H) detached; actions
    (B,10) int64 or None; inputs (B,10) — the scalar injected at each step."""

    def __init__(self, preds: torch.Tensor, states: torch.Tensor,
                 actions: torch.Tensor | None = None,
                 inputs: torch.Tensor | None = None) -> None:
        self.preds = preds
        self.states = states
        self.actions = actions
        self.inputs = inputs


class Seq2SeqLSTM(nn.Module):
    """Encoder–decoder LSTM whose per-step decoder input source is controllable (FR-3)."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN,
                 enc_layers: int = ENC_LAYERS, enc_dropout: float = ENC_DROPOUT) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, num_layers=enc_layers,
                               dropout=enc_dropout, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=1, hidden_size=hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(
        self,
        x_temporal: torch.Tensor,            # (B, 20, 23) standardized features
        y_seed: torch.Tensor,                # (B,) standardized log1p y_t (k=1 input)
        *,
        mode: str,
        teacher: torch.Tensor | None = None,  # (B, 10) log1p ground truth
        aux: torch.Tensor | None = None,      # (B, 2, 10) standardized aux cache slice
        actions: torch.Tensor | None = None,  # (B, 10) fixed action sequence
        policy: nn.Module | None = None,      # live policy (agent-driven)
        epsilon: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> S2SOutput:
        """Decode H steps; per-step input source per D-005; return S2SOutput."""
        if mode not in ("free_running", "teacher_forcing", "policy"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "teacher_forcing" and teacher is None:
            raise ValueError("teacher_forcing mode requires teacher (B,10) log1p targets")
        if mode == "policy":
            if aux is None:
                raise ValueError("policy mode requires aux (B,2,10) cache slice")
            if actions is None and policy is None:
                raise ValueError("policy mode requires fixed actions or a live policy")

        batch = x_temporal.shape[0]
        _, (h_n, c_n) = self.encoder(x_temporal)
        h, c = h_n[-1], c_n[-1]                      # bridge: encoder top layer (D-004)

        preds: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        inputs: list[torch.Tensor] = []
        taken: list[torch.Tensor] = []               # live-policy a_k per step
        live = mode == "policy" and actions is None

        for j in range(HORIZON):                     # j = k−1 (0-based step)
            if j == 0:
                inp = y_seed                          # k=1: ground-truth seed, ALL modes
            elif mode == "free_running":
                inp = to_feedback_scale(preds[j - 1])
            elif mode == "teacher_forcing":
                inp = to_feedback_scale(teacher[:, j - 1])
            else:                                    # policy: a_{k=j} picks input for step j+1
                a = taken[j - 1] if live else actions[:, j - 1]
                inp = torch.where(
                    a == 0, to_feedback_scale(preds[j - 1]),
                    torch.where(a == 1, aux[:, 0, j - 1], aux[:, 1, j - 1]))

            h, c = self.decoder_cell(inp.view(batch, 1), (h, c))
            pred = F.softplus(self.head(h)).squeeze(-1)   # log1p ≥ 0
            preds.append(pred)
            states.append(h.detach())                # s_k = decoder hidden state (FR-3/FR-6)
            inputs.append(inp)
            if live:
                taken.append(policy.act(h.detach(), epsilon, generator))

        out_actions: torch.Tensor | None = None
        if mode == "policy":
            out_actions = torch.stack(taken, dim=1) if live else actions
        return S2SOutput(
            preds=torch.stack(preds, dim=1),
            states=torch.stack(states, dim=1),
            actions=out_actions,
            inputs=torch.stack(inputs, dim=1),
        )


# ── T018: teacher-forcing training smoke (Principle VI) ───────────────────────

def tf_training_smoke(n_samples: int = 2048, epochs: int = 3,
                      out_path: Path | None = None) -> dict:
    """Short real-data teacher-forcing run: assert val error decreases, save checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    xt = encoder_input(np.load(SEQ_DIR / "train_X_temporal.npy", mmap_mode="r")[:n_samples])
    y = np.load(SEQ_DIR / "train_y.npy", mmap_mode="r")[:n_samples]
    xv = encoder_input(np.load(SEQ_DIR / "val_X_temporal.npy", mmap_mode="r")[:n_samples])
    yv = np.load(SEQ_DIR / "val_y.npy", mmap_mode="r")[:n_samples]

    x_t = torch.from_numpy(xt).to(device)
    seed_t = torch.from_numpy(seed_from_window(xt)).to(device)
    y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(device)
    x_v = torch.from_numpy(xv).to(device)
    seed_v = torch.from_numpy(seed_from_window(xv)).to(device)
    y_v = torch.from_numpy(np.ascontiguousarray(yv, dtype=np.float32)).to(device)

    model = Seq2SeqLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()                              # y is log1p → MSE == RMSLE² (DEC-5)

    def _val_loss() -> float:
        model.eval()
        with torch.no_grad():
            out = model(x_v, seed_v, mode="teacher_forcing", teacher=y_v)
            return float(crit(out.preds, y_v))

    val_before = _val_loss()
    for epoch in range(epochs):
        model.train()
        for lo in range(0, len(x_t), 256):
            sl = slice(lo, lo + 256)
            opt.zero_grad()
            out = model(x_t[sl], seed_t[sl], mode="teacher_forcing", teacher=y_t[sl])
            loss = crit(out.preds, y_t[sl])
            loss.backward()
            opt.step()
        print(f"[tf-smoke] epoch {epoch + 1}/{epochs} val_mse={_val_loss():.4f}")
    val_after = _val_loss()

    assert val_after < val_before, \
        f"TF smoke: val error did not decrease ({val_before:.4f} → {val_after:.4f})"
    if out_path is None:
        out_path = _ROOT / "results" / "models" / "pgs2s" / "tf_smoke.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "val_mse": val_after}, out_path)
    print(f"[tf-smoke] val {val_before:.4f} → {val_after:.4f}; checkpoint → {out_path}")
    return {"val_before": val_before, "val_after": val_after, "checkpoint": str(out_path)}


if __name__ == "__main__":
    torch.manual_seed(SEED)
    model = Seq2SeqLSTM()
    x = torch.randn(4, LOOKBACK, N_FEATURES)
    seed = torch.randn(4)
    out = model(x, seed, mode="free_running")
    assert out.preds.shape == (4, HORIZON) and out.states.shape == (4, HORIZON, HIDDEN)
    print(f"[pgs2s_model] synthetic forward OK: preds {tuple(out.preds.shape)}")
    if (SEQ_DIR / "train_X_temporal.npy").exists():
        tf_training_smoke()
    else:
        print("[pgs2s_model] real sequences absent — skipping TF smoke")
    print("[pgs2s_model] smoke OK")
