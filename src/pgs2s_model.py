"""PG-S2S seq2seq forecaster (feature 007): LSTM encoder + LSTMCell decoder with
three decoding modes (free_running / teacher_forcing / policy) and per-step
hidden-state exposure (FR-3; research D-003/D-004/D-005).

Device-agnostic nn.Module — the hard CUDA gate lives in the pgs2s_train.py /
pgs2s_eval.py entry points only (research D-015).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# ── Architecture constants (research D-003/D-004) ─────────────────────────────

N_FEATURES: int = 23      # Cycle-5 selected temporal features (best_config_c)
HIDDEN: int = 128
ENC_LAYERS: int = 2
ENC_DROPOUT: float = 0.3
HORIZON: int = 10
LOOKBACK: int = 20        # PAPER_LOOKBACK window served to the encoder

SEED: int = 42


class S2SOutput:
    """Container for a decode pass: preds (B,10) log1p, states (B,10,H) detached, actions (B,10) or None."""

    def __init__(self, preds: torch.Tensor, states: torch.Tensor,
                 actions: torch.Tensor | None = None) -> None:
        self.preds = preds
        self.states = states
        self.actions = actions


class Seq2SeqLSTM(nn.Module):
    """Encoder–decoder LSTM whose per-step decoder input source is controllable (FR-3)."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN,
                 enc_layers: int = ENC_LAYERS, enc_dropout: float = ENC_DROPOUT) -> None:
        super().__init__()
        raise NotImplementedError("T017")


if __name__ == "__main__":
    print("[pgs2s_model] smoke OK (implementation lands in T017/T018)")
