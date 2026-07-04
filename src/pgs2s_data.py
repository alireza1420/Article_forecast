"""PG-S2S data foundation (feature 007): meta emission, aux prediction caches,
pair-disjoint policy masks, pilot pair sample, and the artifact manifest.

CPU-safe and torch-free by design (research D-015) so every artifact can be
built on any machine and shipped to Colab, verified via the manifest (FR-10).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from features import FEATURE_COLS, build_dl_sequences
except ModuleNotFoundError:
    from src.features import FEATURE_COLS, build_dl_sequences

# ── Paths & constants ─────────────────────────────────────────────────────────

_ROOT: Path = Path(__file__).resolve().parent.parent
PROCESSED_DIR: Path = _ROOT / "data" / "processed"
SEQ_DIR: Path = PROCESSED_DIR / "sequences"
PGS2S_DIR: Path = PROCESSED_DIR / "pgs2s"
MODELS_DIR: Path = _ROOT / "results" / "models"

SEED: int = 42
HORIZON: int = 10
SPLITS: tuple[str, ...] = ("train", "val", "cal", "eval")
# Anchor-week bounds per split (constitution v1.2.0 anchor-bounded interpretation)
ANCHOR_BOUNDS: dict[str, tuple[int, int]] = {
    "train": (26, 105),
    "val": (116, 125),
    "cal": (116, 125),
    "eval": (126, 135),
}
HOLDOUT_FRAC: float = 0.2
N_PILOT_PAIRS: int = 100
SPOT_CHECK_RTOL: float = 1e-5
SPOT_CHECK_ATOL: float = 1e-6


class MissingAnchorRowsError(RuntimeError):
    """Raised when any sample's anchor-week feature row is missing (FR-11)."""

    def __init__(self, missing: list[tuple[int, int, int]]) -> None:
        self.missing = missing
        lines = "\n".join(f"  (center_id={c}, meal_id={m}, W={w})" for c, m, w in missing)
        super().__init__(
            f"[pgs2s] {len(missing)} anchor feature row(s) missing from "
            f"feature_matrix.parquet — no silent fill (FR-11):\n{lines}"
        )


class ManifestMismatchError(RuntimeError):
    """Raised when manifest verification finds drifted/missing files (SC-010)."""

    def __init__(self, diffs: dict) -> None:
        self.diffs = diffs
        lines = "\n".join(f"  {path}: {why}" for path, why in diffs.items())
        super().__init__(
            f"[pgs2s] manifest mismatch — {len(diffs)} file(s) drifted (SC-010):\n{lines}"
        )


# ── DEC-4 feedback scaling (T004) ─────────────────────────────────────────────

_NUM_ORDERS_STATS: tuple[float, float] | None = None


def _num_orders_stats() -> tuple[float, float]:
    """Load (μ, σ) of the log1p num_orders channel from scaler_params.json (cached)."""
    global _NUM_ORDERS_STATS
    if _NUM_ORDERS_STATS is None:
        with open(PROCESSED_DIR / "scaler_params.json") as fh:
            mu, sigma = json.load(fh)["num_orders"]
        _NUM_ORDERS_STATS = (float(mu), float(sigma))
    return _NUM_ORDERS_STATS


def to_feedback_scale(log1p_values):
    """DEC-4: (log1p − μ)/σ with μ,σ from scaler_params.json['num_orders']. Single shared implementation."""
    mu, sigma = _num_orders_stats()
    return (log1p_values - mu) / sigma


def from_feedback_scale(scaled):
    """Inverse of to_feedback_scale, back to log1p space."""
    mu, sigma = _num_orders_stats()
    return scaled * sigma + mu


# ── Meta emission (T009) ──────────────────────────────────────────────────────

def emit_all_meta(processed_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """Re-run build_dl_sequences with all-split meta emission; verify regenerated X/y byte-identity; return {split: meta_path}."""
    seq_dir = processed_dir / "sequences"
    guarded = [seq_dir / f"{s}_{kind}.npy"
               for s in SPLITS for kind in ("X_temporal", "X_static", "y")]
    guarded.append(processed_dir / "scaler_params.json")
    pre_hashes = {p: _sha256_of(p) for p in guarded if p.exists()}

    build_dl_sequences(processed_dir=processed_dir)

    drifted = [str(p) for p, h in pre_hashes.items()
               if not p.exists() or _sha256_of(p) != h]
    if drifted:
        raise RuntimeError(
            "[pgs2s] emit_all_meta: regenerated arrays are NOT byte-identical to the "
            "pre-existing ones — build_dl_sequences output drifted since the stored "
            "artifacts were built (D-001). Stop and reconcile:\n"
            + "\n".join(f"  {p}" for p in drifted)
        )
    return {split: seq_dir / f"{split}_meta.npy" for split in SPLITS}


# ── Aux prediction cache (T011) ───────────────────────────────────────────────

def _load_booster(model: str, h: int):
    """Load the pre-trained h-step booster (xgb JSON / lgbm joblib pkl)."""
    if model == "xgb":
        import xgboost as xgb

        m = xgb.XGBRegressor()
        m.load_model(str(MODELS_DIR / "xgboost" / f"xgb_h{h:02d}.json"))
        return m
    if model == "lgbm":
        import joblib

        return joblib.load(MODELS_DIR / "lightgbm" / f"lgbm_h{h:02d}.pkl")
    raise ValueError(f"unknown booster model {model!r} — expected 'xgb' or 'lgbm'")


def _anchor_features(meta: np.ndarray, fm) -> np.ndarray:
    """Join meta rows on (center_id, meal_id, week==W); return (N, n_features) float32 in meta order; raise on misses."""
    import pandas as pd

    meta_df = pd.DataFrame({
        "center_id": meta[:, 0].astype(np.int64),
        "meal_id": meta[:, 1].astype(np.int64),
        "week": meta[:, 2].astype(np.int64),
    })
    keyed = fm[["center_id", "meal_id", "week"] + FEATURE_COLS].copy()
    for key in ("center_id", "meal_id", "week"):
        keyed[key] = keyed[key].astype(np.int64)
    merged = meta_df.merge(keyed, on=["center_id", "meal_id", "week"],
                           how="left", indicator=True)
    if len(merged) != len(meta_df):
        raise RuntimeError(
            f"[pgs2s] duplicate (center_id, meal_id, week) rows in feature matrix: "
            f"join produced {len(merged)} rows for {len(meta_df)} meta rows"
        )
    miss = merged["_merge"] == "left_only"
    if miss.any():
        missing = [(int(c), int(m), int(w)) for c, m, w in
                   merged.loc[miss, ["center_id", "meal_id", "week"]].itertuples(index=False)]
        raise MissingAnchorRowsError(missing)
    return merged[FEATURE_COLS].to_numpy(dtype=np.float32)


def build_aux_cache(split: str, model: Literal["xgb", "lgbm"],
                    out_dir: Path = PGS2S_DIR,
                    meta_override: np.ndarray | None = None,
                    fm_override=None) -> Path:
    """Score the 10 boosters over the split's anchor rows; save (N,10) standardized-log1p .npy; raise MissingAnchorRowsError with itemised report."""
    import pandas as pd

    meta = meta_override if meta_override is not None \
        else np.load(SEQ_DIR / f"{split}_meta.npy")
    fm = fm_override if fm_override is not None \
        else pd.read_parquet(PROCESSED_DIR / "feature_matrix.parquet", engine="pyarrow")

    X = _anchor_features(meta, fm)
    cache = np.empty((len(meta), HORIZON), dtype=np.float32)
    for h in range(1, HORIZON + 1):
        booster = _load_booster(model, h)
        raw = np.asarray(booster.predict(X), dtype=np.float64)
        # Boosters natively emit log1p(orders) (D-002) — clip at 0 in log space
        # (ml_models._postprocess convention), then apply ONLY the z-score step of DEC-4.
        cache[:, h - 1] = to_feedback_scale(np.clip(raw, 0.0, None)).astype(np.float32)
    if not np.isfinite(cache).all():
        raise RuntimeError(f"[pgs2s] non-finite values in {split}_aux_{model} cache")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_aux_{model}.npy"
    np.save(out_path, cache)
    return out_path


def spot_check_cache(split: str, model: str, n_cells: int = 100, seed: int = SEED) -> None:
    """Re-score n random cells and assert float32-tolerance match vs the cache (SC-002)."""
    import pandas as pd

    cache = np.load(PGS2S_DIR / f"{split}_aux_{model}.npy")
    meta = np.load(SEQ_DIR / f"{split}_meta.npy")
    fm = pd.read_parquet(PROCESSED_DIR / "feature_matrix.parquet", engine="pyarrow")

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(meta), size=n_cells)
    cols = rng.integers(0, HORIZON, size=n_cells)

    uniq_rows = np.unique(rows)
    X = _anchor_features(meta[uniq_rows], fm)
    row_pos = {int(r): i for i, r in enumerate(uniq_rows)}
    for k in np.unique(cols):
        booster = _load_booster(model, int(k) + 1)
        sel = rows[cols == k]
        raw = np.asarray(booster.predict(X[[row_pos[int(r)] for r in sel]]), dtype=np.float64)
        expected = to_feedback_scale(np.clip(raw, 0.0, None)).astype(np.float32)
        np.testing.assert_allclose(
            cache[sel, k], expected,
            rtol=SPOT_CHECK_RTOL, atol=SPOT_CHECK_ATOL,
            err_msg=f"[pgs2s] spot check failed: {split}_aux_{model} column {int(k)} (SC-002)",
        )


# ── Policy masks & pilot pairs (T014) ─────────────────────────────────────────

def _pair_universe(meta: np.ndarray) -> list[tuple[int, int]]:
    """Sorted unique (center_id, meal_id) pairs of a meta array."""
    return sorted({(int(c), int(m)) for c, m in meta[:, :2]})


def build_policy_masks(out_dir: Path = PGS2S_DIR, holdout_frac: float = HOLDOUT_FRAC) -> tuple[Path, Path]:
    """Pair-disjoint 80/20 masks over val/cal rows (rng(42)); assert disjointness + full week coverage."""
    val_meta = np.load(SEQ_DIR / "val_meta.npy")
    cal_meta = np.load(SEQ_DIR / "cal_meta.npy")

    pairs = _pair_universe(val_meta)
    perm = np.random.default_rng(SEED).permutation(len(pairs))
    n_holdout = int(round(holdout_frac * len(pairs)))
    holdout_pairs = {pairs[i] for i in perm[:n_holdout]}
    train_pairs = {pairs[i] for i in perm[n_holdout:]}
    assert not (train_pairs & holdout_pairs), "policy pair sets overlap (US4-AS4)"

    train_mask = np.fromiter(
        ((int(c), int(m)) in train_pairs for c, m in val_meta[:, :2]),
        dtype=bool, count=len(val_meta))
    holdout_mask = np.fromiter(
        ((int(c), int(m)) in holdout_pairs for c, m in cal_meta[:, :2]),
        dtype=bool, count=len(cal_meta))

    lo, hi = ANCHOR_BOUNDS["val"]
    for mask, meta, view in ((train_mask, val_meta, "policy_train"),
                             (holdout_mask, cal_meta, "policy_holdout")):
        weeks = set(np.unique(meta[mask][:, 2]).astype(int))
        assert weeks == set(range(lo, hi + 1)), (
            f"[pgs2s] {view} view missing anchor weeks: {sorted(set(range(lo, hi + 1)) - weeks)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "policy_train_mask.npy"
    holdout_path = out_dir / "policy_holdout_mask.npy"
    np.save(train_path, train_mask)
    np.save(holdout_path, holdout_mask)
    return train_path, holdout_path


def sample_pilot_pairs(n_pairs: int = N_PILOT_PAIRS, out_dir: Path = PGS2S_DIR) -> Path:
    """Deterministically sample pilot pairs from the eval pair universe; save (100,2) int64."""
    pairs = _pair_universe(np.load(SEQ_DIR / "eval_meta.npy"))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(pairs), size=n_pairs, replace=False)
    pilot = np.array([pairs[i] for i in chosen], dtype=np.int64)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pilot_pairs.npy"
    np.save(out_path, pilot)
    return out_path


# ── Manifest (T005) ───────────────────────────────────────────────────────────

def _sha256_of(path: Path) -> str:
    """Streaming sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """Current git HEAD sha, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def manifest_inventory() -> list[Path]:
    """Canonical feature-007 artifact inventory (FR-10): sequences+meta, caches, masks, pilot, scaler, feature matrix, 20 boosters."""
    files: list[Path] = []
    for split in SPLITS:
        for kind in ("X_temporal", "X_static", "y", "meta"):
            files.append(SEQ_DIR / f"{split}_{kind}.npy")
        for model in ("xgb", "lgbm"):
            files.append(PGS2S_DIR / f"{split}_aux_{model}.npy")
    files += [PGS2S_DIR / "policy_train_mask.npy",
              PGS2S_DIR / "policy_holdout_mask.npy",
              PGS2S_DIR / "pilot_pairs.npy",
              PROCESSED_DIR / "scaler_params.json",
              PROCESSED_DIR / "feature_matrix.parquet"]
    files += [MODELS_DIR / "xgboost" / f"xgb_h{h:02d}.json" for h in range(1, HORIZON + 1)]
    files += [MODELS_DIR / "lightgbm" / f"lgbm_h{h:02d}.pkl" for h in range(1, HORIZON + 1)]
    return files


def _relpath(path: Path) -> str:
    """Repo-relative posix path when under _ROOT, else absolute posix path."""
    path = Path(path).resolve()
    try:
        return path.relative_to(_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_manifest(out_path: Path = PGS2S_DIR / "manifest.json",
                   files: list[Path] | None = None) -> Path:
    """Hash every feature-007 input artifact (sha256+bytes) into manifest.json."""
    from datetime import datetime, timezone

    inventory = manifest_inventory() if files is None else [Path(f) for f in files]
    missing = [str(f) for f in inventory if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"[pgs2s] write_manifest: {len(missing)} inventory file(s) missing "
            f"— build them first:\n" + "\n".join(f"  {m}" for m in missing)
        )
    entries = {
        _relpath(f): {"sha256": _sha256_of(f), "bytes": f.stat().st_size}
        for f in sorted(inventory)
    }
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "files": entries,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return out_path


def verify_manifest(manifest_path: Path = PGS2S_DIR / "manifest.json") -> None:
    """Recompute hashes and raise ManifestMismatchError listing every drifted file (SC-010)."""
    manifest_path = Path(manifest_path)
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    diffs: dict[str, str] = {}
    for rel, expected in manifest["files"].items():
        path = Path(rel)
        if not path.is_absolute():
            path = _ROOT / rel
        if not path.exists():
            diffs[rel] = "missing"
            continue
        actual_sha = _sha256_of(path)
        actual_bytes = path.stat().st_size
        if actual_sha != expected["sha256"] or actual_bytes != expected["bytes"]:
            diffs[rel] = (
                f"drift: sha256 {expected['sha256'][:12]}…→{actual_sha[:12]}…, "
                f"bytes {expected['bytes']}→{actual_bytes}"
            )
    if diffs:
        raise ManifestMismatchError(diffs)


def build_all() -> None:
    """Quickstart §1: meta → aux caches (+spot checks) → masks → pilot pairs → manifest."""
    print("[pgs2s_data] emit_all_meta …")
    meta_paths = emit_all_meta()
    for split, path in meta_paths.items():
        meta = np.load(path)
        n_rows = np.load(SEQ_DIR / f"{split}_y.npy", mmap_mode="r").shape[0]
        assert meta.shape == (n_rows, 3), f"{split}_meta misaligned: {meta.shape} vs {n_rows} rows"
        print(f"  {split}: {n_rows} rows, meta OK")

    for split in SPLITS:
        for model in ("xgb", "lgbm"):
            path = build_aux_cache(split, model)
            cache = np.load(path)
            n_nan = int(np.isnan(cache).sum())
            spot_check_cache(split, model)
            print(f"  {split}_aux_{model}: {cache.shape} {n_nan} NaN, spot-check OK")

    train_path, holdout_path = build_policy_masks()
    train_mask, holdout_mask = np.load(train_path), np.load(holdout_path)
    val_meta = np.load(SEQ_DIR / "val_meta.npy")
    cal_meta = np.load(SEQ_DIR / "cal_meta.npy")
    train_pairs = {(int(c), int(m)) for c, m in val_meta[train_mask][:, :2]}
    holdout_pairs = {(int(c), int(m)) for c, m in cal_meta[holdout_mask][:, :2]}
    print(f"  masks disjoint: {not (train_pairs & holdout_pairs)} "
          f"({len(train_pairs)} train / {len(holdout_pairs)} holdout pairs)")

    pilot_path = sample_pilot_pairs()
    print(f"  pilot pairs: {np.load(pilot_path).shape}")

    manifest_path = write_manifest()
    with open(manifest_path) as fh:
        n_files = len(json.load(fh)["files"])
    print(f"  manifest: {n_files} files → {manifest_path}")
    print("[pgs2s_data] build-all complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PG-S2S data foundation (CPU-safe)")
    parser.add_argument("command", nargs="?", choices=["build-all"],
                        help="build-all: run the full quickstart §1 pipeline")
    args = parser.parse_args()
    if args.command == "build-all":
        build_all()
    else:
        print(f"[pgs2s_data] PGS2S_DIR={PGS2S_DIR}")
        print("[pgs2s_data] smoke OK")
