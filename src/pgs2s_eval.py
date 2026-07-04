"""PG-S2S canonical-protocol evaluation (feature 007): ε=0 eval-split decode of
the 7 FR-8 systems over ≥3 seeds, rolling-origin tree regeneration (DEC-3),
contract prediction CSVs, comparison tables and figures (FR-8/12/13).

REQUIRES CUDA for model inference (D-015). Metrics are imported from
src/evaluate.py (Constitution V) — never reimplemented.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = _ROOT / "results"
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"

SEED: int = 42
HORIZON: int = 10
# FR-8 comparison systems (registry cycle-8 names + existing baselines)
SYSTEMS: tuple[str, ...] = (
    "pgs2s", "s2s_free", "s2s_tf", "s2s_teach_xgb", "s2s_teach_lgbm",
    "demandrnn", "lstm_ims",
)


def evaluate_all_systems(run_ids: list[str], out_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """ε=0 eval-split evaluation of the 7 FR-8 systems over ≥3 seeds; writes comparison tables, figures, prediction CSVs."""
    raise NotImplementedError("T031")


def regenerate_rolling_origin_trees(out_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """DEC-3 supplementary table: xgb/lgbm rolling-origin eval metrics from the eval aux caches."""
    raise NotImplementedError("T032")


if __name__ == "__main__":
    print("[pgs2s_eval] smoke OK (implementation lands in T031–T033)")
