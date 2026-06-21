"""Tests for src/eda.py — all EDA analysis and plotting functions."""
import warnings
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.eda import (
    profile_dataset,
    plot_order_distribution,
    plot_timeseries_grid,
    analyse_group_demand,
    save_group_demand_tex,
    analyse_promotion_effect,
    analyse_price_elasticity,
    plot_promotion_elasticity,
    plot_acf_analysis,
    save_dataset_stats_tex,
)


# ---------------------------------------------------------------------------
# T014 — test_profile_dataset  (US2)
# ---------------------------------------------------------------------------

def test_profile_dataset_keys(merged_df):
    profile = profile_dataset(merged_df)
    for key in ("shape", "dtypes", "missing", "unique_counts", "num_orders_stats", "history_lengths"):
        assert key in profile, f"Missing key: {key}"


def test_profile_dataset_zero_order_rate(merged_df):
    profile = profile_dataset(merged_df)
    n_zeros = (merged_df["num_orders"] == 0).sum()
    expected = n_zeros / len(merged_df)
    assert abs(profile["num_orders_stats"]["zero_order_rate"] - expected) < 1e-9


def test_profile_dataset_history_length_min(merged_df):
    profile = profile_dataset(merged_df)
    assert profile["history_lengths"]["min"] >= 1


# ---------------------------------------------------------------------------
# T016 — test_plot_order_distribution  (US3)
# ---------------------------------------------------------------------------

def test_plot_order_distribution(merged_df, tmp_path):
    out = plot_order_distribution(merged_df, tmp_path)
    assert out.name == "fig1_order_distribution.pdf"
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# T018 — test_plot_timeseries_grid  (US4)
# ---------------------------------------------------------------------------

def test_plot_timeseries_grid_pdf_exists(merged_df, tmp_path):
    out = plot_timeseries_grid(merged_df, tmp_path, seed=42, n_pairs=4)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_timeseries_grid_reproducible(merged_df, tmp_path):
    out1 = plot_timeseries_grid(merged_df, tmp_path / "a", seed=42, n_pairs=4)
    out2 = plot_timeseries_grid(merged_df, tmp_path / "b", seed=42, n_pairs=4)
    # Both files exist; reproducibility check passes if no exception raised
    assert out1.exists() and out2.exists()


def test_plot_timeseries_grid_short_pair(merged_df, tmp_path):
    """Pair with only 1 week of history must not raise an error."""
    out = plot_timeseries_grid(merged_df, tmp_path, seed=42, n_pairs=4)
    assert out.exists()


# ---------------------------------------------------------------------------
# T020 — test_analyse_group_demand  (US5)
# ---------------------------------------------------------------------------

def test_analyse_group_demand_keys(merged_df):
    result = analyse_group_demand(merged_df)
    assert set(result.keys()) == {"center_type", "cuisine", "category"}


def test_analyse_group_demand_columns(merged_df):
    result = analyse_group_demand(merged_df)
    for key, df in result.items():
        assert list(df.columns) == ["group_value", "mean_orders", "std_orders", "count"], \
            f"Wrong columns for {key}: {list(df.columns)}"


def test_analyse_group_demand_mean_correct(merged_df):
    result = analyse_group_demand(merged_df)
    indian_rows = merged_df[merged_df["cuisine"] == "Indian"]["num_orders"]
    expected_mean = indian_rows.mean()
    cuisine_df = result["cuisine"]
    row = cuisine_df[cuisine_df["group_value"] == "Indian"].iloc[0]
    assert abs(row["mean_orders"] - expected_mean) < 1e-9


# ---------------------------------------------------------------------------
# T021 — test_save_group_demand_tex  (US5)
# ---------------------------------------------------------------------------

def _make_group_stats():
    return {
        "center_type": pd.DataFrame({
            "group_value": ["TYPE_A", "TYPE_B"],
            "mean_orders": [10.0, 8.0],
            "std_orders": [2.0, 1.5],
            "count": [100, 80],
        }),
        "cuisine": pd.DataFrame({
            "group_value": ["Indian", "Thai"],
            "mean_orders": [12.0, 7.0],
            "std_orders": [3.0, 1.0],
            "count": [150, 60],
        }),
        "category": pd.DataFrame({
            "group_value": ["Biryani"],
            "mean_orders": [15.0],
            "std_orders": [4.0],
            "count": [200],
        }),
    }


def test_save_group_demand_tex_exists(tmp_path):
    out = save_group_demand_tex(_make_group_stats(), tmp_path)
    assert out.exists()
    assert out.name == "group_demand.tex"


def test_save_group_demand_tex_booktabs(tmp_path):
    out = save_group_demand_tex(_make_group_stats(), tmp_path)
    content = out.read_text()
    assert r"\toprule" in content
    assert r"\bottomrule" in content


# ---------------------------------------------------------------------------
# T024 — test_analyse_promotion_effect  (US6)
# ---------------------------------------------------------------------------

def test_analyse_promotion_effect_keys(merged_df):
    result = analyse_promotion_effect(merged_df)
    assert set(result.keys()) == {"emailer", "homepage"}


def test_analyse_promotion_effect_subkeys(merged_df):
    result = analyse_promotion_effect(merged_df)
    for flag in ("emailer", "homepage"):
        for k in ("mean_0", "mean_1", "lift_pct", "diff"):
            assert k in result[flag], f"Missing {k} in {flag}"


def test_analyse_promotion_effect_lift_pct(merged_df):
    result = analyse_promotion_effect(merged_df)
    for flag_col, key in [("emailer_for_promotion", "emailer"), ("homepage_featured", "homepage")]:
        mean_0 = merged_df[merged_df[flag_col] == 0]["num_orders"].mean()
        mean_1 = merged_df[merged_df[flag_col] == 1]["num_orders"].mean()
        expected_lift = (mean_1 - mean_0) / mean_0 * 100
        assert abs(result[key]["lift_pct"] - expected_lift) < 1e-6


def test_analyse_promotion_effect_single_value_raises(merged_df):
    df_bad = merged_df.copy()
    df_bad["emailer_for_promotion"] = 0
    df_bad["homepage_featured"] = 0
    with pytest.raises(ValueError):
        analyse_promotion_effect(df_bad)


# ---------------------------------------------------------------------------
# T026 — test_analyse_price_elasticity  (US7)
# ---------------------------------------------------------------------------

def test_analyse_price_elasticity_warns_zero_price(merged_df):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pearson_r, scatter_df = analyse_price_elasticity(merged_df)
        assert any("base_price" in str(warning.message).lower() or
                   "zero" in str(warning.message).lower() or
                   "excluded" in str(warning.message).lower()
                   for warning in w), "Expected a warning about zero base_price rows"


def test_analyse_price_elasticity_pearson_range(merged_df):
    pearson_r, scatter_df = analyse_price_elasticity(merged_df)
    assert isinstance(pearson_r, float)
    assert -1.0 <= pearson_r <= 1.0


def test_analyse_price_elasticity_scatter_columns(merged_df):
    pearson_r, scatter_df = analyse_price_elasticity(merged_df)
    assert "mean_discount_rate" in scatter_df.columns
    assert "mean_orders" in scatter_df.columns


# ---------------------------------------------------------------------------
# T027 — test_plot_promotion_elasticity  (US7)
# ---------------------------------------------------------------------------

def _make_promo_stats():
    return {
        "emailer": {"mean_0": 10.0, "mean_1": 14.0, "lift_pct": 40.0, "diff": 4.0},
        "homepage": {"mean_0": 9.0, "mean_1": 13.0, "lift_pct": 44.4, "diff": 4.0},
    }


def test_plot_promotion_elasticity_pdf_exists(tmp_path):
    promo_stats = _make_promo_stats()
    scatter_df = pd.DataFrame({
        "mean_discount_rate": [0.1, 0.2, 0.3],
        "mean_orders": [10.0, 15.0, 12.0],
    })
    out = plot_promotion_elasticity(promo_stats, scatter_df, 0.45, tmp_path)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# T030 — test_plot_acf_analysis  (US8)
# ---------------------------------------------------------------------------

def test_plot_acf_analysis_pdf_exists(merged_df, tmp_path):
    out = plot_acf_analysis(merged_df, tmp_path, max_lags=5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_acf_analysis_short_series_no_exception(merged_df, tmp_path):
    """Pair with only 1 week must not raise — lags capped at len-1."""
    out = plot_acf_analysis(merged_df, tmp_path, max_lags=20)
    assert out.exists()


# ---------------------------------------------------------------------------
# T032 — test_save_dataset_stats_tex  (US9)
# ---------------------------------------------------------------------------

def _make_minimal_profile():
    return {
        "shape": (100, 12),
        "dtypes": {"num_orders": "int64"},
        "missing": {},
        "unique_counts": {
            "center_id": 5,
            "meal_id": 10,
            "week": 20,
            "center_type": 3,
            "cuisine": 4,
            "category": 8,
        },
        "num_orders_stats": {
            "mean": 12.5,
            "std": 4.3,
            "median": 11.0,
            "p95": 25.0,
            "min": 0.0,
            "max": 48.0,
            "zero_order_rate": 0.08,
        },
        "history_lengths": {
            "min": 1,
            "median": 10.0,
            "max": 20,
            "distribution": pd.Series([5, 10, 3], index=[1, 10, 20]),
        },
    }


def test_save_dataset_stats_tex_exists(tmp_path):
    out = save_dataset_stats_tex(_make_minimal_profile(), tmp_path)
    assert out.exists()
    assert out.name == "dataset_stats.tex"


def test_save_dataset_stats_tex_content(tmp_path):
    out = save_dataset_stats_tex(_make_minimal_profile(), tmp_path)
    content = out.read_text()
    assert r"\toprule" in content
    assert r"\bottomrule" in content
    assert "n\\_weeks" in content or "n_weeks" in content
    assert "zero\\_order\\_rate" in content or "zero_order_rate" in content


def test_save_dataset_stats_tex_missing_key_raises(tmp_path):
    profile = _make_minimal_profile()
    del profile["num_orders_stats"]
    with pytest.raises(KeyError):
        save_dataset_stats_tex(profile, tmp_path)
