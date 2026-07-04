"""PG-S2S asynchronous alternation trainer (feature 007): paper Alg. 1 with the
DEC-2 split restriction, teacher-forcing warm-up, per-round checkpoints, dual
val/cal diagnostics, resume, and the DEC-6 pilot gate (FR-7/12).

REQUIRES CUDA — raises RuntimeError at entry when no GPU is available (D-015).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

try:
    from pgs2s_agent import (ALPHA, BETA, GAMMA, EPS_START, EPS_DECAY, EPS_MIN, N_P)
except ModuleNotFoundError:
    from src.pgs2s_agent import (ALPHA, BETA, GAMMA, EPS_START, EPS_DECAY, EPS_MIN, N_P)

# ── Training constants (research D-008) ───────────────────────────────────────

LR_POLICY: float = 1e-3   # l₁ — policy Adam LR
LR_RNN: float = 1e-4      # l₂ — seq2seq Adam LR (repo convention)
BATCH_SIZE: int = 64
ROUNDS: int = 10          # full-scale alternation rounds
AGENT_EPOCHS: int = 5     # policy passes per round (full scale)
RNN_EPOCHS: int = 10      # seq2seq epochs per round (full scale)
# Pilot overrides (spec §Pilot Configuration)
PILOT_ROUNDS: int = 3
PILOT_AGENT_EPOCHS: int = 3
PILOT_RNN_EPOCHS: int = 5

SEED: int = 42
RUN_SEEDS: tuple[int, ...] = (42, 43, 44)

_ROOT: Path = Path(__file__).resolve().parent.parent
RUNS_DIR: Path = _ROOT / "results" / "models" / "pgs2s"


def require_cuda() -> torch.device:
    """Return cuda device or raise RuntimeError('[pgs2s] CUDA is required…')."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[pgs2s] CUDA is required — run on a GPU machine/Colab "
            "(data/cache building via pgs2s_data.py is CPU-safe)."
        )
    return torch.device("cuda")


def set_determinism(seed: int) -> None:
    """Seed python/numpy/torch and enable full CUDA determinism (research D-008, SC-006)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def code_hash() -> str:
    """sha256 over sorted contents of src/pgs2s_{data,model,agent,train}.py — the DEC-6 gate key (D-012)."""
    src_dir = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in sorted(["pgs2s_data.py", "pgs2s_model.py", "pgs2s_agent.py", "pgs2s_train.py"]):
        h.update((src_dir / name).read_bytes())
    return h.hexdigest()


def run_training(seed: int = SEED, pilot: bool = True,
                 run_dir: Path | None = None, resume: bool = False) -> dict:
    """Full Alg.-1 alternation with DEC-2 split restriction; per-round checkpoints + dual diagnostics; returns run summary."""
    raise NotImplementedError("T026")


if __name__ == "__main__":
    print(f"[pgs2s_train] code_hash={code_hash()[:12]}…")
    print("[pgs2s_train] smoke OK (implementation lands in T026–T028)")
