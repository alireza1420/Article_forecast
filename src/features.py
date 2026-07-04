"""Feature engineering: tabular matrix and sequence tensors for the LSTM."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR: Path = _ROOT / "data" / "processed"
TABLES_DIR: Path = _ROOT / "results" / "tables"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_MAX_WEEK: int = 115
VAL_WEEKS: tuple[int, int] = (116, 125)
CAL_WEEKS: tuple[int, int] = (116, 125)  # same as VAL; kept for compatibility
EVAL_WEEKS: tuple[int, int] = (126, 145)
LOOKBACK: int = 26
HORIZON: int = 10
LAG_SHIFTS: list[int] = [10, 11, 12, 13]
ROLLING_WINDOWS: list[int] = [4, 8]
EWM_SPANS: list[int] = [10, 13, 26]
SEED: int = 42

FEATURE_COLS: list[str] = [
    # Lags
    "lag_10", "lag_11", "lag_12", "lag_13",
    # Rolling 4-week
    "rolling_mean_4w", "rolling_std_4w", "rolling_min_4w", "rolling_max_4w",
    # Rolling 8-week
    "rolling_mean_8w", "rolling_std_8w", "rolling_min_8w", "rolling_max_8w",
    # EWM
    "ewm_span10", "ewm_span13", "ewm_span26", "ewm_alpha05",
    # Price
    "discount_rate", "log_checkout_price",
    # Promotion
    "emailer_for_promotion", "homepage_featured",
    "email_x_discount", "homepage_x_discount",
    # Calendar
    "week_number", "week_of_year_sin", "week_of_year_cos",
    # Center aggregates
    "center_mean_orders", "center_std_orders",
    # Meal aggregates
    "meal_mean_orders", "meal_std_orders",
    # Categorical (encoded)
    "center_type_enc", "category_enc", "cuisine_enc",
    # Base-price aggregates per meal (train only)
    "base_price_max", "base_price_mean", "base_price_min",
    # Checkout-price aggregates per meal (train only)
    "meal_price_max", "meal_price_mean", "meal_price_min",
    # Count aggregates (train only)
    "meal_count", "center_cat_count", "center_cui_count", "region_meal_count",
    # Static price ranks (train only)
    "center_price_rank", "center_cat_price_rank",
    "meal_city_price_rank", "meal_price_rank", "meal_region_price_rank",
    # Time-varying weekly counts
    "center_week_count", "center_cat_week_count", "city_meal_week_count",
    "meal_week_count", "region_meal_week_count", "type_meal_week_count",
    # Time-varying price ranks
    "center_week_price_rank", "meal_week_price_rank",
]

# DL sequence column sets (additive — FEATURE_COLS not modified)
SEQUENCE_TEMPORAL_COLS: list[str] = [
    "num_orders",
    "checkout_price",
    "base_price",
    "discount_rate",
    "log_checkout_price",
    "emailer_for_promotion",
    "homepage_featured",
    "email_x_discount",
    "homepage_x_discount",
    "week_number",
    "week_of_year_sin",
    "week_of_year_cos",
    # Time-varying weekly counts
    "center_week_count",
    "center_cat_week_count",
    "city_meal_week_count",
    "meal_week_count",
    "region_meal_week_count",
    "type_meal_week_count",
    # Time-varying price ranks
    "center_week_price_rank",
    "meal_week_price_rank",
    # EWM features (α=0.5 per paper; span variants for multi-scale smoothing)
    "ewm_alpha05",
    "ewm_span10",
    "ewm_span13",
    "ewm_span26",
    # Rolling statistics (4-week and 8-week windows)
    "rolling_mean_4w", "rolling_std_4w", "rolling_min_4w", "rolling_max_4w",
    "rolling_mean_8w", "rolling_std_8w", "rolling_min_8w", "rolling_max_8w",
]

SEQUENCE_STATIC_COLS: list[str] = [
    "center_type_enc",
    "category_enc",
    "cuisine_enc",
    "op_area",
    "center_mean_orders",
    "meal_mean_orders",
    # Base-price aggregates per meal
    "base_price_max", "base_price_mean", "base_price_min",
    # Checkout-price aggregates per meal
    "meal_price_max", "meal_price_mean", "meal_price_min",
    # Count aggregates
    "meal_count", "center_cat_count", "center_cui_count", "region_meal_count",
    # Static price ranks
    "center_price_rank", "center_cat_price_rank",
    "meal_city_price_rank", "meal_price_rank", "meal_region_price_rank",
]

# Demand-derived temporal columns that must be log1p-transformed before standardisation.
# Raw demand is right-skewed (max ~13 000 orders) which, after z-score scaling, produces
# values up to 33 std devs — saturating LSTM tanh/sigmoid gates and causing systematic
# underprediction. log1p compresses the scale to match the log1p target (y = log1p(orders)).
SEQUENCE_LOG1P_COLS: set[str] = {
    "num_orders",
    "ewm_alpha05", "ewm_span10", "ewm_span13", "ewm_span26",
    "rolling_mean_4w", "rolling_std_4w", "rolling_min_4w", "rolling_max_4w",
    "rolling_mean_8w", "rolling_std_8w", "rolling_min_8w", "rolling_max_8w",
    "center_week_count", "center_cat_week_count", "city_meal_week_count",
    "meal_week_count", "region_meal_week_count", "type_meal_week_count",
}

SEQUENCE_STATIC_CONTINUOUS_COLS: list[str] = [
    "op_area",
    "center_mean_orders",
    "meal_mean_orders",
    # New continuous static features
    "base_price_max", "base_price_mean", "base_price_min",
    "meal_price_max", "meal_price_mean", "meal_price_min",
    "meal_count", "center_cat_count", "center_cui_count", "region_meal_count",
    "center_price_rank", "center_cat_price_rank",
    "meal_city_price_rank", "meal_price_rank", "meal_region_price_rank",
]

SPLIT_CONFIG: dict[str, tuple[int, int]] = {
    "train": (1, TRAIN_MAX_WEEK),
    "val":   VAL_WEEKS,
    "cal":   CAL_WEEKS,
    "eval":  EVAL_WEEKS,
}

# ---------------------------------------------------------------------------
# Module-level mutable state (populated by build_* calls or JSON load)
# ---------------------------------------------------------------------------
ENCODERS: dict[str, dict] = {}
SCALER_PARAMS: dict[str, list] = {}

# Try loading persisted artefacts at import time (silent if absent)
_enc_path = PROCESSED_DIR / "encoders.json"
_scaler_path = PROCESSED_DIR / "scaler_params.json"
if _enc_path.exists():
    with open(_enc_path, encoding="utf-8") as _f:
        ENCODERS.update(json.load(_f))
if _scaler_path.exists():
    with open(_scaler_path, encoding="utf-8") as _f:
        SCALER_PARAMS.update(json.load(_f))


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _add_all_features(
    df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Compute all feature columns on the full merged DataFrame; return enriched copy."""
    out = df.copy().sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)
    grp = out.groupby(["center_id", "meal_id"], sort=False)

    # --- US1: Lag features ---
    for shift in LAG_SHIFTS:
        out[f"lag_{shift}"] = grp["num_orders"].shift(shift)

    # --- US2: Rolling statistics ---
    for w in ROLLING_WINDOWS:
        rolled = grp["num_orders"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w)
        )
        # We need mean, std, min, max separately
        out[f"rolling_mean_{w}w"] = grp["num_orders"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).mean()
        )
        out[f"rolling_std_{w}w"] = grp["num_orders"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).std()
        )
        out[f"rolling_min_{w}w"] = grp["num_orders"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).min()
        )
        out[f"rolling_max_{w}w"] = grp["num_orders"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).max()
        )

    # --- US3: EWM features ---
    for span in EWM_SPANS:
        out[f"ewm_span{span}"] = grp["num_orders"].transform(
            lambda s, _sp=span: s.ewm(span=_sp, adjust=False).mean()
        )
    out["ewm_alpha05"] = grp["num_orders"].transform(
        lambda s: s.ewm(alpha=0.5, adjust=False).mean()
    )

    # --- US4: Price features ---
    out["discount_rate"] = np.where(
        out["base_price"] > 0,
        (out["base_price"] - out["checkout_price"]) / out["base_price"],
        np.nan,
    )
    out["log_checkout_price"] = np.log(out["checkout_price"] + 1)

    # --- US5: Promotion interaction features ---
    out["email_x_discount"] = out["emailer_for_promotion"] * out["discount_rate"]
    out["homepage_x_discount"] = out["homepage_featured"] * out["discount_rate"]

    # --- US6: Calendar features ---
    out["week_number"] = out["week"]
    woy = ((out["week"] - 1) % 52) + 1
    out["week_of_year_sin"] = np.sin(2 * np.pi * woy / 52)
    out["week_of_year_cos"] = np.cos(2 * np.pi * woy / 52)

    # --- US7: Center-level aggregates (training weeks only) ---
    train_mask = out["week"] <= TRAIN_MAX_WEEK
    center_agg = (
        out.loc[train_mask]
        .groupby("center_id")["num_orders"]
        .agg(center_mean_orders="mean", center_std_orders="std")
        .reset_index()
    )
    out = out.merge(center_agg, on="center_id", how="left")

    # --- US8: Meal-level aggregates (training weeks only) ---
    meal_agg = (
        out.loc[out["week"] <= TRAIN_MAX_WEEK]
        .groupby("meal_id")["num_orders"]
        .agg(meal_mean_orders="mean", meal_std_orders="std")
        .reset_index()
    )
    out = out.merge(meal_agg, on="meal_id", how="left")

    # --- US9: Categorical encoding ---
    cat_cols = {"center_type": "center_type_enc", "category": "category_enc", "cuisine": "cuisine_enc"}
    train_rows = out[out["week"] <= TRAIN_MAX_WEEK]
    enc_store: dict[str, dict] = {}
    for src_col, enc_col in cat_cols.items():
        le = LabelEncoder()
        le.fit(train_rows[src_col].astype(str))
        label_map = {label: int(code) for code, label in enumerate(le.classes_)}
        enc_store[src_col] = label_map
        out[enc_col] = out[src_col].astype(str).map(label_map).fillna(-1).astype(int)

    # Persist encoders
    ENCODERS.clear()
    ENCODERS.update(enc_store)
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "encoders.json", "w", encoding="utf-8") as f:
        json.dump(enc_store, f, indent=2)

    # --- US10: Base-price aggregates per meal (train only) ---
    train_mask = out["week"] <= TRAIN_MAX_WEEK
    meal_bp = (
        out[train_mask].groupby("meal_id")["base_price"]
        .agg(base_price_max="max", base_price_mean="mean", base_price_min="min")
        .reset_index()
    )
    out = out.merge(meal_bp, on="meal_id", how="left")

    # --- US11: Checkout-price aggregates per meal (train only) ---
    meal_cp = (
        out[train_mask].groupby("meal_id")["checkout_price"]
        .agg(meal_price_max="max", meal_price_mean="mean", meal_price_min="min")
        .reset_index()
    )
    out = out.merge(meal_cp, on="meal_id", how="left")

    # --- US12: Count aggregates (train only) ---
    meal_cnt = (
        out[train_mask].groupby("meal_id")["num_orders"]
        .sum().rename("meal_count").reset_index()
    )
    out = out.merge(meal_cnt, on="meal_id", how="left")

    center_cat_cnt = (
        out[train_mask].groupby(["center_id", "category"])["num_orders"]
        .sum().rename("center_cat_count").reset_index()
    )
    out = out.merge(center_cat_cnt, on=["center_id", "category"], how="left")

    center_cui_cnt = (
        out[train_mask].groupby(["center_id", "cuisine"])["num_orders"]
        .sum().rename("center_cui_count").reset_index()
    )
    out = out.merge(center_cui_cnt, on=["center_id", "cuisine"], how="left")

    region_meal_cnt = (
        out[train_mask].groupby(["region_code", "meal_id"])["num_orders"]
        .sum().rename("region_meal_count").reset_index()
    )
    out = out.merge(region_meal_cnt, on=["region_code", "meal_id"], how="left")

    # --- US13: Static price ranks (train only) ---
    _mp = (
        out[train_mask].groupby(["center_id", "meal_id"])["checkout_price"]
        .mean().reset_index(name="_p")
    )
    _mp["center_price_rank"] = _mp.groupby("center_id")["_p"].rank(method="dense")
    out = out.merge(_mp[["center_id", "meal_id", "center_price_rank"]], on=["center_id", "meal_id"], how="left")

    _cp = (
        out[train_mask].groupby(["center_id", "category"])["checkout_price"]
        .mean().reset_index(name="_p")
    )
    _cp["center_cat_price_rank"] = _cp.groupby("center_id")["_p"].rank(method="dense")
    out = out.merge(_cp[["center_id", "category", "center_cat_price_rank"]], on=["center_id", "category"], how="left")

    _cityp = (
        out[train_mask].groupby(["city_code", "meal_id"])["checkout_price"]
        .mean().reset_index(name="_p")
    )
    _cityp["meal_city_price_rank"] = _cityp.groupby("city_code")["_p"].rank(method="dense")
    out = out.merge(_cityp[["city_code", "meal_id", "meal_city_price_rank"]], on=["city_code", "meal_id"], how="left")

    _gp = (
        out[train_mask].groupby("meal_id")["checkout_price"]
        .mean().reset_index(name="_p")
    )
    _gp["meal_price_rank"] = _gp["_p"].rank(method="dense")
    out = out.merge(_gp[["meal_id", "meal_price_rank"]], on="meal_id", how="left")

    _rp = (
        out[train_mask].groupby(["region_code", "meal_id"])["checkout_price"]
        .mean().reset_index(name="_p")
    )
    _rp["meal_region_price_rank"] = _rp.groupby("region_code")["_p"].rank(method="dense")
    out = out.merge(_rp[["region_code", "meal_id", "meal_region_price_rank"]], on=["region_code", "meal_id"], how="left")

    # --- US14: Time-varying weekly counts (all weeks — current week is always known) ---
    cwc = out.groupby(["center_id", "week"])["num_orders"].sum().rename("center_week_count").reset_index()
    out = out.merge(cwc, on=["center_id", "week"], how="left")

    ccwc = out.groupby(["center_id", "category", "week"])["num_orders"].sum().rename("center_cat_week_count").reset_index()
    out = out.merge(ccwc, on=["center_id", "category", "week"], how="left")

    cmwc = out.groupby(["city_code", "meal_id", "week"])["num_orders"].sum().rename("city_meal_week_count").reset_index()
    out = out.merge(cmwc, on=["city_code", "meal_id", "week"], how="left")

    mwc = out.groupby(["meal_id", "week"])["num_orders"].sum().rename("meal_week_count").reset_index()
    out = out.merge(mwc, on=["meal_id", "week"], how="left")

    rmwc = out.groupby(["region_code", "meal_id", "week"])["num_orders"].sum().rename("region_meal_week_count").reset_index()
    out = out.merge(rmwc, on=["region_code", "meal_id", "week"], how="left")

    tmwc = out.groupby(["center_type", "meal_id", "week"])["num_orders"].sum().rename("type_meal_week_count").reset_index()
    out = out.merge(tmwc, on=["center_type", "meal_id", "week"], how="left")

    # --- US15: Time-varying price ranks (all weeks) ---
    out["center_week_price_rank"] = out.groupby(["center_id", "week"])["checkout_price"].rank(method="dense")
    out["meal_week_price_rank"] = out.groupby(["meal_id", "week"])["checkout_price"].rank(method="dense")

    return out


def _standardise_temporal(
    df: pd.DataFrame,
    train_end: int = TRAIN_MAX_WEEK,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """log1p-transform demand cols, then z-score standardise all temporal cols; persist stats."""
    out = df.copy()

    # Apply log1p to demand-derived columns before standardising so that right-skewed
    # demand values (max ~13 000 raw orders → up to 33 std devs after plain z-score)
    # are compressed to a range that won't saturate LSTM tanh/sigmoid gates.
    for col in SEQUENCE_LOG1P_COLS:
        if col in out.columns:
            out[col] = np.log1p(np.clip(out[col], 0.0, None))

    train_mask = out["week"] <= train_end
    sp: dict[str, list] = {}
    for col in SEQUENCE_TEMPORAL_COLS:
        col_vals = out.loc[train_mask, col].values
        col_mean = float(np.nanmean(col_vals))
        col_std = float(np.nanstd(col_vals, ddof=1))
        if col_std < 1e-12:
            col_std = 1.0
        sp[col] = [col_mean, col_std]
        out[col] = (out[col] - col_mean) / col_std
    SCALER_PARAMS.update(sp)
    processed_dir.mkdir(parents=True, exist_ok=True)
    sp_path = processed_dir / "scaler_params.json"
    existing: dict = {}
    if sp_path.exists():
        with open(sp_path, encoding="utf-8") as _f:
            try:
                existing = json.load(_f)
            except json.JSONDecodeError:
                existing = {}
    existing.update(sp)
    with open(sp_path, "w", encoding="utf-8") as _f:
        json.dump(existing, _f, indent=2)
    return out


def _standardise_static(
    df: pd.DataFrame,
    train_end: int = TRAIN_MAX_WEEK,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Standardise continuous static cols; persist stats; return modified copy."""
    out = df.copy()
    train_mask = out["week"] <= train_end
    sp: dict[str, list] = {}
    for col in SEQUENCE_STATIC_CONTINUOUS_COLS:
        if col not in out.columns:
            continue
        col_vals = out.loc[train_mask, col].values
        col_mean = float(np.nanmean(col_vals))
        col_std = float(np.nanstd(col_vals, ddof=1))
        if col_std < 1e-12:
            col_std = 1.0
        sp[col] = [col_mean, col_std]
        out[col] = (out[col] - col_mean) / col_std
    SCALER_PARAMS.update(sp)
    sp_path = processed_dir / "scaler_params.json"
    existing: dict = {}
    if sp_path.exists():
        with open(sp_path, encoding="utf-8") as _f:
            try:
                existing = json.load(_f)
            except json.JSONDecodeError:
                existing = {}
    existing.update(sp)
    with open(sp_path, "w", encoding="utf-8") as _f:
        json.dump(existing, _f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def make_horizon_target(fm: pd.DataFrame, h: int) -> pd.DataFrame:
    """Return feature rows paired with the h-step-ahead log1p target.

    Shifts num_orders forward by h within each (center_id, meal_id) group,
    applies log1p, and drops rows where the shifted target is NaN.
    """
    if h < 1 or h > HORIZON:
        raise ValueError(f"h must be between 1 and {HORIZON}, got {h}")
    out = fm.sort_values(["center_id", "meal_id", "week"]).copy()
    out["y"] = out.groupby(["center_id", "meal_id"])["num_orders"].shift(-h)
    out = out.dropna(subset=["y"]).copy()
    out["y"] = np.log1p(out["y"])
    return out.reset_index(drop=True)


def build_feature_matrix(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Assemble tabular feature matrix, drop NaN rows, save parquet; return DataFrame."""
    merged_path = processed_dir / "merged.parquet"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged.parquet not found at {merged_path}")
    df = pd.read_parquet(merged_path, engine="pyarrow")
    enriched = _add_all_features(df, processed_dir=processed_dir)
    enriched = enriched.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    keep_cols = ["center_id", "meal_id", "week", "num_orders"] + FEATURE_COLS
    enriched = enriched[keep_cols]
    out_path = processed_dir / "feature_matrix.parquet"
    enriched.to_parquet(out_path, engine="pyarrow", index=False)
    return enriched


def build_sequences(
    processed_dir: Path = PROCESSED_DIR,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build (X, y) sequence arrays for train/val/cal/eval splits; save as .npy."""
    fm_path = processed_dir / "feature_matrix.parquet"
    if fm_path.exists():
        fm = pd.read_parquet(fm_path, engine="pyarrow")
    else:
        fm = build_feature_matrix(processed_dir=processed_dir)

    # --- Fit scaler on training rows ---
    train_rows = fm[fm["week"] <= TRAIN_MAX_WEEK]
    sp: dict[str, list] = {}
    for col in FEATURE_COLS:
        col_mean = float(np.nanmean(train_rows[col].values))
        col_std = float(np.nanstd(train_rows[col].values, ddof=1))
        if col_std < 1e-12:
            col_std = 1.0
        sp[col] = [col_mean, col_std]

    SCALER_PARAMS.clear()
    SCALER_PARAMS.update(sp)
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(sp, f, indent=2)

    # --- Scale feature values ---
    means = np.array([sp[c][0] for c in FEATURE_COLS], dtype=np.float64)
    stds = np.array([sp[c][1] for c in FEATURE_COLS], dtype=np.float64)
    fm_scaled = fm.copy()
    fm_scaled[FEATURE_COLS] = (fm[FEATURE_COLS].values - means) / stds

    # --- Sliding window construction ---
    split_Xs: dict[str, list] = {s: [] for s in ("train", "val", "cal", "eval")}
    split_ys: dict[str, list] = {s: [] for s in ("train", "val", "cal", "eval")}

    # Define split boundaries by first-target-week (t+1)
    # train:  27 ≤ t+1 ≤ 106  →  t ∈ [26, 96]   (using first valid which needs lookback rows)
    # val:   107 ≤ t+1 ≤ 116  →  t ∈ [106, 115]
    # cal:   117 ≤ t+1 ≤ 126  →  t ∈ [116, 125]
    # eval:  127 ≤ t+1 ≤ 136  →  t ∈ [126, 135]
    def _split_name(t: int) -> str | None:
        if 26 <= t <= 96:
            return "train"
        if 106 <= t <= 115:
            return "val"
        if 116 <= t <= 125:
            return "cal"
        if 126 <= t <= 135:
            return "eval"
        return None

    # For each pair, slide windows along the feature matrix rows
    for (center_id, meal_id), pair_fm in fm_scaled.groupby(["center_id", "meal_id"], sort=False):
        pair_fm = pair_fm.sort_values("week").reset_index(drop=True)
        n_rows = len(pair_fm)
        if n_rows < lookback:
            continue
        weeks = pair_fm["week"].values
        feat_vals = pair_fm[FEATURE_COLS].values.astype(np.float32)
        orders_vals = pair_fm["num_orders"].values.astype(np.float32)

        # Build a lookup from week → row index and orders for y
        week_to_idx = {int(w): i for i, w in enumerate(weeks)}

        for end in range(lookback - 1, n_rows):
            # X window: rows [end-lookback+1 .. end]
            t = int(weeks[end])
            split = _split_name(t)
            if split is None:
                continue
            # y: we need num_orders at weeks t+1..t+horizon for this pair
            y_weeks = [t + h for h in range(1, horizon + 1)]
            if not all(w in week_to_idx for w in y_weeks):
                continue
            X_window = feat_vals[end - lookback + 1: end + 1]  # (lookback, n_feat)
            y_vals = np.array([orders_vals[week_to_idx[w]] for w in y_weeks], dtype=np.float32)
            split_Xs[split].append(X_window)
            split_ys[split].append(y_vals)

    # --- Save .npy files ---
    seq_dir = processed_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "cal", "eval"):
        if split_Xs[split]:
            X = np.stack(split_Xs[split], axis=0).astype(np.float32)
            y = np.stack(split_ys[split], axis=0).astype(np.float32)
        else:
            X = np.zeros((0, lookback, len(FEATURE_COLS)), dtype=np.float32)
            y = np.zeros((0, horizon), dtype=np.float32)
        np.save(seq_dir / f"{split}_X.npy", X)
        np.save(seq_dir / f"{split}_y.npy", y)
        result[split] = (X, y)

    return result


def load_feature_matrix(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load feature_matrix.parquet; raise FileNotFoundError if absent."""
    path = processed_dir / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(f"feature_matrix.parquet not found at {path}")
    return pd.read_parquet(path, engine="pyarrow")


def build_dl_sequences(
    df: pd.DataFrame | None = None,
    processed_dir: Path = PROCESSED_DIR,
    lookback: int = LOOKBACK,
    horizons: int = HORIZON,
    split_config: dict | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build and save DL sequence arrays (X_temporal, X_static, y) for all splits."""
    if split_config is None:
        split_config = SPLIT_CONFIG

    if df is None:
        merged_path = processed_dir / "merged.parquet"
        if not merged_path.exists():
            raise FileNotFoundError(f"merged.parquet not found at {merged_path}")
        raw = pd.read_parquet(merged_path, engine="pyarrow")
        df = _add_all_features(raw, processed_dir=processed_dir)

    all_dl_cols = list(dict.fromkeys(
        SEQUENCE_TEMPORAL_COLS + SEQUENCE_STATIC_COLS + ["week", "center_id", "meal_id"]
    ))
    missing = [c for c in all_dl_cols if c not in df.columns]
    if missing:
        raise ValueError(f"build_dl_sequences: missing columns in df: {missing}")

    # Stash raw num_orders before scaling — y target must remain un-scaled for RMSLE
    df = df.copy()
    df["num_orders_raw"] = df["num_orders"].copy()
    df = _standardise_temporal(df, train_end=TRAIN_MAX_WEEK, processed_dir=processed_dir)
    df = _standardise_static(df, train_end=TRAIN_MAX_WEEK, processed_dir=processed_dir)

    # Fill NaN from early-series rolling/EWM values (min_periods not met at start of each series)
    fill_cols = [c for c in SEQUENCE_TEMPORAL_COLS + SEQUENCE_STATIC_COLS if c in df.columns and df[c].isna().any()]
    if fill_cols:
        df[fill_cols] = (
            df.groupby(["center_id", "meal_id"])[fill_cols]
            .transform(lambda s: s.ffill().bfill())
        )
        df[fill_cols] = df[fill_cols].fillna(0.0)

    seq_dir = processed_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    Xt_lists: dict[str, list] = {s: [] for s in split_config}
    Xs_lists: dict[str, list] = {s: [] for s in split_config}
    y_lists: dict[str, list] = {s: [] for s in split_config}
    meta_lists: dict[str, list] = {s: [] for s in split_config}
    excluded_count = 0

    for (cid, mid), group in df.groupby(["center_id", "meal_id"], sort=True):
        group = group.sort_values("week").reset_index(drop=True)
        week_set = set(int(w) for w in group["week"].values)
        if len(week_set) < lookback:
            excluded_count += 1
            continue
        week_to_row: dict[int, int] = {int(w): i for i, w in enumerate(group["week"].values)}

        for split_name, (start, end) in split_config.items():
            is_train = split_name == "train"
            for W in range(start, end + 1):
                # Anchor week W is the LAST observed week: the lookback window ends at W
                # (inclusive) so the most recent demand (week W) is available to the model,
                # and h=1 is a true 1-step-ahead forecast of W+1 — matching the ML setup
                # (make_horizon_target pairs features@W with target W+h). Previously the
                # window ended at W-1, silently discarding week W and making h=1 a 2-step
                # forecast, which crippled short-horizon accuracy.
                lb_weeks = list(range(W - lookback + 1, W + 1))
                tgt_weeks = list(range(W + 1, W + horizons + 1))
                if not all(w in week_set for w in lb_weeks):
                    continue
                if not all(w in week_set for w in tgt_weeks):
                    continue
                if is_train and max(tgt_weeks) > TRAIN_MAX_WEEK:
                    continue
                if W not in week_set:
                    continue
                lb_idx = [week_to_row[w] for w in lb_weeks]
                tgt_idx = [week_to_row[w] for w in tgt_weeks]
                X_t = group.iloc[lb_idx][SEQUENCE_TEMPORAL_COLS].to_numpy(dtype=np.float32)
                X_s = group.iloc[week_to_row[W]][SEQUENCE_STATIC_COLS].to_numpy(dtype=np.float32)
                y_v = np.log1p(group.iloc[tgt_idx]["num_orders_raw"].to_numpy(dtype=np.float32))
                Xt_lists[split_name].append(X_t)
                Xs_lists[split_name].append(X_s)
                y_lists[split_name].append(y_v)
                meta_lists[split_name].append([float(cid), float(mid), float(W)])

    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name in split_config:
        if Xt_lists[split_name]:
            Xt = np.stack(Xt_lists[split_name], axis=0)
            Xs = np.stack(Xs_lists[split_name], axis=0)
            y = np.stack(y_lists[split_name], axis=0)
        else:
            Xt = np.empty((0, lookback, len(SEQUENCE_TEMPORAL_COLS)), dtype=np.float32)
            Xs = np.empty((0, len(SEQUENCE_STATIC_COLS)), dtype=np.float32)
            y = np.empty((0, horizons), dtype=np.float32)

        for arr_name, arr in [("X_temporal", Xt), ("X_static", Xs), ("y", y)]:
            if np.isnan(arr).any():
                raise ValueError(f"NaN detected in {split_name}_{arr_name}")

        np.save(seq_dir / f"{split_name}_X_temporal.npy", Xt)
        np.save(seq_dir / f"{split_name}_X_static.npy", Xs)
        np.save(seq_dir / f"{split_name}_y.npy", y)

        # Persist per-sample identity (center_id, meal_id, anchor W) for every split —
        # row-aligned with X/y by construction (feature 007 FR-1; additive per contract).
        meta_arr = (
            np.array(meta_lists[split_name], dtype=np.float32)
            if meta_lists[split_name]
            else np.empty((0, 3), dtype=np.float32)
        )
        np.save(seq_dir / f"{split_name}_meta.npy", meta_arr)

        print(f"{split_name}: X_temporal={Xt.shape} X_static={Xs.shape} y={y.shape}")
        result[split_name] = (Xt, Xs, y)

    print(f"Excluded pairs (insufficient history): {excluded_count}")
    return result


def save_feature_table_tex(out_dir: Path = TABLES_DIR) -> Path:
    """Write feature_table.tex booktabs tabular; return path."""
    # Metadata: (type, description, model_scope)
    feature_meta: dict[str, tuple[str, str, str]] = {
        "lag_10":              ("float", "num\\_orders lagged 10 weeks per pair",          "Both"),
        "lag_11":              ("float", "num\\_orders lagged 11 weeks per pair",          "Both"),
        "lag_12":              ("float", "num\\_orders lagged 12 weeks per pair",          "Both"),
        "lag_13":              ("float", "num\\_orders lagged 13 weeks per pair",          "Both"),
        "rolling_mean_4w":     ("float", "4-week rolling mean of num\\_orders",            "Both"),
        "rolling_std_4w":      ("float", "4-week rolling std of num\\_orders",             "Both"),
        "rolling_min_4w":      ("float", "4-week rolling minimum of num\\_orders",         "Both"),
        "rolling_max_4w":      ("float", "4-week rolling maximum of num\\_orders",         "Both"),
        "rolling_mean_8w":     ("float", "8-week rolling mean of num\\_orders",            "Both"),
        "rolling_std_8w":      ("float", "8-week rolling std of num\\_orders",             "Both"),
        "rolling_min_8w":      ("float", "8-week rolling minimum of num\\_orders",         "Both"),
        "rolling_max_8w":      ("float", "8-week rolling maximum of num\\_orders",         "Both"),
        "ewm_span10":          ("float", "Exponentially weighted mean, span=10",           "Both"),
        "ewm_span13":          ("float", "Exponentially weighted mean, span=13",           "Both"),
        "ewm_span26":          ("float", "Exponentially weighted mean, span=26",           "Both"),
        "ewm_alpha05":         ("float", "Exponentially weighted mean, $\\alpha=0.5$",     "Both"),
        "discount_rate":       ("float", "(base\\_price$-$checkout\\_price)/base\\_price", "Both"),
        "log_checkout_price":  ("float", "$\\log$(checkout\\_price$+1$)",                  "Both"),
        "emailer_for_promotion": ("int", "Email promotion flag (0/1)",                    "Both"),
        "homepage_featured":   ("int",   "Homepage feature flag (0/1)",                   "Both"),
        "email_x_discount":    ("float", "emailer\\_for\\_promotion $\\times$ discount\\_rate", "Both"),
        "homepage_x_discount": ("float", "homepage\\_featured $\\times$ discount\\_rate", "Both"),
        "week_number":         ("int",   "Raw week index (1--145)",                       "Tabular"),
        "week_of_year_sin":    ("float", "$\\sin(2\\pi\\cdot$woy$/52)$",                  "Both"),
        "week_of_year_cos":    ("float", "$\\cos(2\\pi\\cdot$woy$/52)$",                  "Both"),
        "center_mean_orders":  ("float", "Mean demand per centre (train weeks only)",      "Both"),
        "center_std_orders":   ("float", "Std demand per centre (train weeks only)",       "Both"),
        "meal_mean_orders":    ("float", "Mean demand per meal (train weeks only)",        "Both"),
        "meal_std_orders":     ("float", "Std demand per meal (train weeks only)",         "Both"),
        "center_type_enc":          ("int",   "Label-encoded centre type",                               "Both"),
        "category_enc":             ("int",   "Label-encoded meal category",                            "Both"),
        "cuisine_enc":              ("int",   "Label-encoded cuisine",                                  "Both"),
        "base_price_max":           ("float", "Max base price for meal (train weeks)",                  "Both"),
        "base_price_mean":          ("float", "Mean base price for meal (train weeks)",                 "Both"),
        "base_price_min":           ("float", "Min base price for meal (train weeks)",                  "Both"),
        "meal_price_max":           ("float", "Max checkout price for meal (train weeks)",              "Both"),
        "meal_price_mean":          ("float", "Mean checkout price for meal (train weeks)",             "Both"),
        "meal_price_min":           ("float", "Min checkout price for meal (train weeks)",              "Both"),
        "meal_count":               ("int",   "Total orders for meal across all centres (train)",       "Both"),
        "center_cat_count":         ("int",   "Total orders for category in centre (train)",            "Both"),
        "center_cui_count":         ("int",   "Total orders for cuisine in centre (train)",             "Both"),
        "region_meal_count":        ("int",   "Total orders for meal in region (train)",                "Both"),
        "center_price_rank":        ("float", "Dense rank of meal price within centre (train)",         "Both"),
        "center_cat_price_rank":    ("float", "Dense rank of category price within centre (train)",     "Both"),
        "meal_city_price_rank":     ("float", "Dense rank of meal price within city (train)",           "Both"),
        "meal_price_rank":          ("float", "Dense rank of meal price globally (train)",              "Both"),
        "meal_region_price_rank":   ("float", "Dense rank of meal price within region (train)",         "Both"),
        "center_week_count":        ("int",   "Total orders in centre for given week",                  "Both"),
        "center_cat_week_count":    ("int",   "Total orders for category in centre for given week",     "Both"),
        "city_meal_week_count":     ("int",   "Total orders for meal in city for given week",           "Both"),
        "meal_week_count":          ("int",   "Total orders for meal across all centres for given week","Both"),
        "region_meal_week_count":   ("int",   "Total orders for meal in region for given week",         "Both"),
        "type_meal_week_count":     ("int",   "Total orders for meal in centre type for given week",    "Both"),
        "center_week_price_rank":   ("float", "Dense rank of meal price in centre for given week",      "Both"),
        "meal_week_price_rank":     ("float", "Dense rank of meal price globally for given week",       "Both"),
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Feature Engineering: 54 Model-Input Columns}",
        r"\label{tab:features}",
        r"\begin{tabular}{llp{5.5cm}l}",
        r"\toprule",
        r"\textbf{Feature} & \textbf{Type} & \textbf{Description} & \textbf{Scope} \\",
        r"\midrule",
    ]
    for col in FEATURE_COLS:
        ftype, desc, scope = feature_meta[col]
        safe_col = col.replace("_", r"\_")
        lines.append(f"\\texttt{{{safe_col}}} & {ftype} & {desc} & {scope} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Smoke test entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_ROOT))

    print("Building feature matrix…")
    fm = build_feature_matrix()
    print(f"  Feature matrix: {len(fm):,} rows × {len(fm.columns)} columns")

    print("Building sequences…")
    splits = build_sequences()
    for name, (X, y) in splits.items():
        print(f"  {name:5s}  X: {X.shape}  y: {y.shape}")

    print("Building DL sequences…")
    dl_splits = build_dl_sequences()
    for name, (Xt, Xs, y_dl) in dl_splits.items():
        print(f"  {name:5s}  X_temporal={Xt.shape}  X_static={Xs.shape}  y={y_dl.shape}")

    tex_path = save_feature_table_tex()
    print(f"Saved {tex_path}")
    print("Feature engineering complete.")
