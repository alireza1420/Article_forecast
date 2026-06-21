"""TDD tests for src/features.py — all stories US1–US12."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLS,
    HORIZON,
    LOOKBACK,
    TRAIN_MAX_WEEK,
    _add_all_features,
    build_feature_matrix,
    build_sequences,
    load_feature_matrix,
    save_feature_table_tex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pair(df: pd.DataFrame, center: int, meal: int, week: int) -> pd.Series:
    mask = (df["center_id"] == center) & (df["meal_id"] == meal) & (df["week"] == week)
    return df.loc[mask].iloc[0]


def _enriched(features_df: pd.DataFrame, tmp_path: Path | None = None) -> pd.DataFrame:
    """Call _add_all_features with a tmp processed_dir."""
    processed_dir = tmp_path if tmp_path is not None else Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    return _add_all_features(features_df, processed_dir=processed_dir)


# ---------------------------------------------------------------------------
# US1 — Lag features  (T009, T010)
# ---------------------------------------------------------------------------

def test_lag_values_correct(features_df, tmp_path):
    """lag_10 at week 15 == num_orders at week 5 == 10; lag_13 at week 20 == 14."""
    df = _enriched(features_df, tmp_path)
    # pair A: num_orders = week*2, so week 5 → 10, week 7 → 14
    row15 = _pair(df, 1, 10, 15)
    assert row15["lag_10"] == pytest.approx(10.0), "lag_10 at week 15 should equal num_orders at week 5 = 10"
    row20 = _pair(df, 1, 10, 20)
    assert row20["lag_13"] == pytest.approx(14.0), "lag_13 at week 20 should equal num_orders at week 7 = 14"


def test_lag_nans_for_early_rows(features_df, tmp_path):
    """lag_10 NaN for weeks 1–10; lag_13 NaN for weeks 1–13 of pair A."""
    df = _enriched(features_df, tmp_path)
    pair_a = df[(df["center_id"] == 1) & (df["meal_id"] == 10)]
    for week in range(1, 11):
        val = pair_a.loc[pair_a["week"] == week, "lag_10"].values[0]
        assert np.isnan(val), f"lag_10 should be NaN at week {week}"
    for week in range(1, 14):
        val = pair_a.loc[pair_a["week"] == week, "lag_13"].values[0]
        assert np.isnan(val), f"lag_13 should be NaN at week {week}"


# ---------------------------------------------------------------------------
# US2 — Rolling statistics  (T012, T013)
# ---------------------------------------------------------------------------

def test_rolling_mean_4w_correct(features_df, tmp_path):
    """rolling_mean_4w at week 10 == mean(w7..w10) = 17.0; min=14, max=20."""
    df = _enriched(features_df, tmp_path)
    row = _pair(df, 1, 10, 10)
    # pair A weeks 7,8,9,10 → orders 14,16,18,20
    assert row["rolling_mean_4w"] == pytest.approx(17.0)
    assert row["rolling_min_4w"] == pytest.approx(14.0)
    assert row["rolling_max_4w"] == pytest.approx(20.0)


def test_rolling_nans_for_partial_windows(features_df, tmp_path):
    """rolling_mean_4w is NaN for week 1; rolling_mean_8w is NaN for week 7."""
    df = _enriched(features_df, tmp_path)
    pair_a = df[(df["center_id"] == 1) & (df["meal_id"] == 10)]
    val_4w = pair_a.loc[pair_a["week"] == 1, "rolling_mean_4w"].values[0]
    assert np.isnan(val_4w), "rolling_mean_4w at week 1 should be NaN"
    val_8w = pair_a.loc[pair_a["week"] == 7, "rolling_mean_8w"].values[0]
    assert np.isnan(val_8w), "rolling_mean_8w at week 7 should be NaN (only 7 rows, min_periods=8)"


# ---------------------------------------------------------------------------
# US3 — EWM  (T015)
# ---------------------------------------------------------------------------

def test_ewm_recency_ordering(features_df, tmp_path):
    """ewm_span10 > ewm_span26 for pair A (increasing series); no NaN anywhere."""
    df = _enriched(features_df, tmp_path)
    pair_a = df[(df["center_id"] == 1) & (df["meal_id"] == 10)]
    row50 = pair_a[pair_a["week"] == 50].iloc[0]
    assert row50["ewm_span10"] > row50["ewm_span26"], (
        "Shorter EWM span should react faster to rising trend"
    )
    for col in ["ewm_span10", "ewm_span13", "ewm_span26"]:
        assert not pair_a[col].isna().any(), f"{col} should have no NaN values"


# ---------------------------------------------------------------------------
# US4 — Price features  (T017, T018, T019)
# ---------------------------------------------------------------------------

def test_discount_rate_correct(features_df, tmp_path):
    """discount_rate == (10-8)/10 == 0.2 for rows with base_price=10."""
    df = _enriched(features_df, tmp_path)
    non_zero_base = df[df["base_price"] > 0]
    assert np.allclose(non_zero_base["discount_rate"].values, 0.2, atol=1e-9)


def test_log_checkout_price_correct(features_df, tmp_path):
    """log_checkout_price == log(checkout+1) == log(9) for checkout=8 rows."""
    df = _enriched(features_df, tmp_path)
    expected = np.log(8.0 + 1.0)
    rows = df[df["checkout_price"] == 8.0]
    assert np.allclose(rows["log_checkout_price"].values, expected, atol=1e-9)


def test_zero_base_price_nan(features_df, tmp_path):
    """Row with base_price==0 must have discount_rate=NaN (not 0, not inf)."""
    df = _enriched(features_df, tmp_path)
    zero_base_row = df[df["base_price"] == 0.0]
    assert len(zero_base_row) >= 1, "Expected at least one zero-base-price row in fixture"
    assert zero_base_row["discount_rate"].isna().all()


# ---------------------------------------------------------------------------
# US5 — Promotion interaction features  (T021, T022)
# ---------------------------------------------------------------------------

def test_interaction_terms_correct(features_df, tmp_path):
    """email_x_discount == emailer * discount_rate; homepage_x_discount similarly."""
    df = _enriched(features_df, tmp_path)
    non_zero = df[df["base_price"] > 0]
    email1 = non_zero[non_zero["emailer_for_promotion"] == 1]
    assert np.allclose(email1["email_x_discount"].values, 0.2, atol=1e-9)
    email0 = non_zero[non_zero["emailer_for_promotion"] == 0]
    assert np.allclose(email0["email_x_discount"].values, 0.0, atol=1e-12)


def test_interaction_zero_base_price(features_df, tmp_path):
    """email_x_discount is NaN for zero-base-price row (NaN propagates)."""
    df = _enriched(features_df, tmp_path)
    zero_row = df[df["base_price"] == 0.0]
    assert zero_row["email_x_discount"].isna().all()
    assert zero_row["homepage_x_discount"].isna().all()


# ---------------------------------------------------------------------------
# US6 — Calendar features  (T024, T025)
# ---------------------------------------------------------------------------

def test_week_of_year_cyclic(features_df, tmp_path):
    """week 1 and week 53 produce identical sin/cos; week 14 sin matches formula."""
    df = _enriched(features_df, tmp_path)
    # week 1 and week 53 both map to woy=1
    row1 = _pair(df, 1, 10, 1)
    row53 = _pair(df, 1, 10, 53)
    assert row1["week_of_year_sin"] == pytest.approx(row53["week_of_year_sin"], abs=1e-9)
    assert row1["week_of_year_cos"] == pytest.approx(row53["week_of_year_cos"], abs=1e-9)
    # week 14 → woy = ((14-1)%52)+1 = 14
    row14 = _pair(df, 1, 10, 14)
    expected_sin = np.sin(2 * np.pi * 14 / 52)
    assert row14["week_of_year_sin"] == pytest.approx(expected_sin, abs=1e-9)


def test_week_number_passthrough(features_df, tmp_path):
    """week_number equals the week column value for every row."""
    df = _enriched(features_df, tmp_path)
    assert (df["week_number"] == df["week"]).all()


# ---------------------------------------------------------------------------
# US7 — Center aggregates  (T027, T028)
# ---------------------------------------------------------------------------

def test_center_aggregates_training_only(features_df, tmp_path):
    """center_mean_orders for center=1 uses only training weeks."""
    df = _enriched(features_df, tmp_path)
    train_center1 = features_df[(features_df["center_id"] == 1) & (features_df["week"] <= TRAIN_MAX_WEEK)]
    expected_mean = train_center1["num_orders"].mean()
    c1_rows = df[df["center_id"] == 1]
    assert np.allclose(c1_rows["center_mean_orders"].values, expected_mean, rtol=1e-6)


def test_center_aggregate_all_rows_joined(features_df, tmp_path):
    """Training centers have non-NaN aggregates; unseen center gets NaN."""
    # Inject center=99 that appears only in weeks > TRAIN_MAX_WEEK
    extra = pd.DataFrame([{
        "center_id": 99, "meal_id": 10, "week": TRAIN_MAX_WEEK + 5,
        "num_orders": 7, "base_price": 10.0, "checkout_price": 8.0,
        "emailer_for_promotion": 0, "homepage_featured": 1,
        "center_type": "TYPE_X", "category": "Biryani", "cuisine": "Indian",
        "city_code": 990, "region_code": 99, "op_area": 1.0,
        "discount_rate": 0.2,
    }])
    combined = pd.concat([features_df, extra], ignore_index=True)
    df = _add_all_features(combined, processed_dir=tmp_path)
    # center=1 (training) should have finite aggregates
    assert not df[df["center_id"] == 1]["center_mean_orders"].isna().any()
    # center=99 (never in training) should have NaN aggregates
    assert df[df["center_id"] == 99]["center_mean_orders"].isna().all()


# ---------------------------------------------------------------------------
# US8 — Meal aggregates  (T030)
# ---------------------------------------------------------------------------

def test_meal_aggregates_training_only(features_df, tmp_path):
    """meal_mean_orders for meal=10 uses only training weeks; all rows get the value."""
    df = _enriched(features_df, tmp_path)
    train_meal10 = features_df[(features_df["meal_id"] == 10) & (features_df["week"] <= TRAIN_MAX_WEEK)]
    expected_mean = train_meal10["num_orders"].mean()
    meal10_rows = df[df["meal_id"] == 10]
    assert np.allclose(meal10_rows["meal_mean_orders"].values, expected_mean, rtol=1e-6)


# ---------------------------------------------------------------------------
# US9 — Categorical encoding  (T032, T033, T034)
# ---------------------------------------------------------------------------

def test_encoder_fit_on_training_only(features_df, tmp_path):
    """TYPE_A maps to same integer in training and evaluation rows."""
    df = _enriched(features_df, tmp_path)
    train_enc = df[(df["center_id"] == 1) & (df["week"] <= TRAIN_MAX_WEEK)]["center_type_enc"].iloc[0]
    eval_enc = df[(df["center_id"] == 1) & (df["week"] > TRAIN_MAX_WEEK)]["center_type_enc"].iloc[0]
    assert train_enc == eval_enc
    assert isinstance(int(train_enc), int)
    assert train_enc >= 0


def test_unseen_label_sentinel(features_df, tmp_path):
    """Row with center_type not in training gets center_type_enc == -1."""
    extra = pd.DataFrame([{
        "center_id": 1, "meal_id": 10, "week": TRAIN_MAX_WEEK + 3,
        "num_orders": 5, "base_price": 10.0, "checkout_price": 8.0,
        "emailer_for_promotion": 0, "homepage_featured": 0,
        "center_type": "TYPE_Z", "category": "Biryani", "cuisine": "Indian",
        "city_code": 100, "region_code": 10, "op_area": 2.0, "discount_rate": 0.2,
    }])
    combined = pd.concat([features_df, extra], ignore_index=True)
    df = _add_all_features(combined, processed_dir=tmp_path)
    unseen = df[df["center_type"] == "TYPE_Z"]
    assert len(unseen) == 1
    assert int(unseen["center_type_enc"].iloc[0]) == -1


def test_encoders_json_roundtrip(features_df, tmp_path):
    """encoders.json written by _add_all_features is valid JSON with expected keys."""
    _add_all_features(features_df, processed_dir=tmp_path)
    enc_path = tmp_path / "encoders.json"
    assert enc_path.exists()
    with open(enc_path, encoding="utf-8") as f:
        enc = json.load(f)
    assert "center_type" in enc
    assert "category" in enc
    assert "cuisine" in enc
    # All values should be dicts mapping strings to ints
    for key in ("center_type", "category", "cuisine"):
        for label, code in enc[key].items():
            assert isinstance(label, str)
            assert isinstance(code, int)


# ---------------------------------------------------------------------------
# US10 — Tabular feature matrix  (T036, T037, T038)
# ---------------------------------------------------------------------------

def test_feature_cols_excludes_identifiers_and_target():
    """FEATURE_COLS must not contain target or identifier columns; length == 31."""
    for col in ("num_orders", "week", "center_id", "meal_id"):
        assert col not in FEATURE_COLS, f"{col} must not be in FEATURE_COLS"
    assert len(FEATURE_COLS) == 31


def test_build_feature_matrix_no_nans(features_df, tmp_path):
    """build_feature_matrix returns DataFrame with zero NaN across all FEATURE_COLS."""
    features_df.to_parquet(tmp_path / "merged.parquet", engine="pyarrow")
    fm = build_feature_matrix(processed_dir=tmp_path)
    assert len(fm) > 0
    assert fm[FEATURE_COLS].isna().sum().sum() == 0


def test_lag_leakage_check(features_df, tmp_path):
    """lag_10 at (pair A, week 20) == num_orders at (pair A, week 10) == 20.0."""
    features_df.to_parquet(tmp_path / "merged.parquet", engine="pyarrow")
    fm = build_feature_matrix(processed_dir=tmp_path)
    row = fm[(fm["center_id"] == 1) & (fm["meal_id"] == 10) & (fm["week"] == 20)]
    assert len(row) == 1
    # num_orders at week 10 = 10*2 = 20
    assert row["lag_10"].values[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# US11 — Sequence tensors  (T041, T042, T043, T044, T045)
# ---------------------------------------------------------------------------

def _build_splits(features_df: pd.DataFrame, tmp_path: Path):
    """Write merged.parquet and return build_sequences output."""
    features_df.to_parquet(tmp_path / "merged.parquet", engine="pyarrow")
    return build_sequences(processed_dir=tmp_path)


def test_sequence_shapes_correct(features_df, tmp_path):
    """X.shape == (n, 26, 31) and y.shape == (n, 10) for every non-empty split."""
    splits = _build_splits(features_df, tmp_path)
    assert set(splits.keys()) == {"train", "val", "cal", "eval"}
    for name, (X, y) in splits.items():
        if X.shape[0] == 0:
            continue
        assert X.shape[1] == LOOKBACK, f"{name} X.shape[1] != LOOKBACK"
        assert X.shape[2] == len(FEATURE_COLS), f"{name} X.shape[2] != len(FEATURE_COLS)"
        assert y.shape[1] == HORIZON, f"{name} y.shape[1] != HORIZON"
        assert X.dtype == np.float32
        assert y.dtype == np.float32


def test_sequence_X_standardised(features_df, tmp_path):
    """train_X values are finite and feature-wise means are close to 0 (z-score applied)."""
    splits = _build_splits(features_df, tmp_path)
    X_train = splits["train"][0]
    assert X_train.shape[0] > 0, "train split must have at least one sample"
    assert np.isfinite(X_train).all(), "train_X contains non-finite values"
    # Flatten over samples and time: (n*26, 31)
    flat = X_train.reshape(-1, X_train.shape[2])
    col_means = flat.mean(axis=0)
    # For varying features means should be near 0; constant features are exactly 0 after scaling
    assert np.all(np.abs(col_means) < 5), (
        f"Feature means too far from 0 after z-scoring: "
        f"{[(FEATURE_COLS[i], col_means[i]) for i in np.where(np.abs(col_means) >= 5)[0]]}"
    )


def test_sequence_y_raw_counts(features_df, tmp_path):
    """y values are non-negative and integer-valued (raw demand counts)."""
    splits = _build_splits(features_df, tmp_path)
    for name, (_, y) in splits.items():
        if y.shape[0] == 0:
            continue
        assert (y >= 0).all(), f"{name}: y has negative values"
        assert np.allclose(y, np.round(y), atol=1e-4), f"{name}: y has non-integer values"


def test_short_pair_excluded_from_sequences(features_df, tmp_path):
    """Pair B (10 weeks) produces 0 sequence samples; only pair A contributes."""
    splits = _build_splits(features_df, tmp_path)
    train_n = splits["train"][0].shape[0]
    # Pair A: feature matrix weeks 14-145 (132 rows); first valid t=39, last train t=96 → 58 samples
    assert train_n == 58, f"Expected 58 train samples from pair A only, got {train_n}"
    assert splits["val"][0].shape[0] == 10
    assert splits["cal"][0].shape[0] == 10
    assert splits["eval"][0].shape[0] == 10


def test_scaler_params_training_only(features_df, tmp_path):
    """scaler_params.json has 31 keys; lag_10 mean matches training feature matrix."""
    features_df.to_parquet(tmp_path / "merged.parquet", engine="pyarrow")
    fm = build_feature_matrix(processed_dir=tmp_path)
    build_sequences(processed_dir=tmp_path)
    sp_path = tmp_path / "scaler_params.json"
    assert sp_path.exists()
    with open(sp_path, encoding="utf-8") as f:
        sp = json.load(f)
    assert len(sp) == 31
    expected_mean = float(np.nanmean(fm.loc[fm["week"] <= TRAIN_MAX_WEEK, "lag_10"].values))
    assert sp["lag_10"][0] == pytest.approx(expected_mean, rel=1e-5)


# ---------------------------------------------------------------------------
# US12 — Feature documentation table  (T048, T049)
# ---------------------------------------------------------------------------

def test_feature_table_tex_exists(tmp_path):
    """save_feature_table_tex writes non-empty feature_table.tex."""
    path = save_feature_table_tex(out_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_feature_table_tex_content(tmp_path):
    """feature_table.tex contains booktabs macros and all 31 feature rows."""
    save_feature_table_tex(out_dir=tmp_path)
    content = (tmp_path / "feature_table.tex").read_text(encoding="utf-8")
    assert r"\toprule" in content
    assert r"\bottomrule" in content
    assert r"lag\_10" in content
    # Count data rows (lines with \\ that are not header/footer lines)
    data_rows = [ln for ln in content.splitlines()
                 if "\\\\" in ln
                 and r"\toprule" not in ln
                 and r"\bottomrule" not in ln
                 and r"\midrule" not in ln
                 and r"\textbf" not in ln]
    assert len(data_rows) == 31, f"Expected 31 data rows, found {len(data_rows)}"
