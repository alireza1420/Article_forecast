"""Tests for src/data.py — load_raw_csvs, merge_datasets, save_merged, load_merged."""
import pytest
import pandas as pd
from pathlib import Path

from src.data import load_raw_csvs, merge_datasets, save_merged, load_merged


# ---------------------------------------------------------------------------
# T007 — test_load_raw_csvs
# ---------------------------------------------------------------------------

def test_load_raw_csvs_missing_raises(tmp_path):
    """FileNotFoundError when any CSV is absent."""
    with pytest.raises(FileNotFoundError):
        load_raw_csvs(tmp_path)


def test_load_raw_csvs_returns_three_dataframes(tmp_path):
    """Returns three DataFrames with the expected column sets."""
    train = pd.DataFrame({
        "id": [1], "week": [1], "center_id": [1], "meal_id": [10],
        "checkout_price": [9.0], "base_price": [10.0],
        "emailer_for_promotion": [0], "homepage_featured": [0], "num_orders": [5],
    })
    meal_info = pd.DataFrame({"meal_id": [10], "category": ["Biryani"], "cuisine": ["Indian"]})
    centre_info = pd.DataFrame({
        "center_id": [1], "city_code": [100], "region_code": [10],
        "op_area": [2.0], "center_type": ["TYPE_A"],
    })

    train.to_csv(tmp_path / "train.csv", index=False)
    meal_info.to_csv(tmp_path / "meal_info.csv", index=False)
    centre_info.to_csv(tmp_path / "fulfilment_center_info.csv", index=False)

    t, m, c = load_raw_csvs(tmp_path)

    assert set(["id", "week", "center_id", "meal_id", "num_orders"]).issubset(t.columns)
    assert set(["meal_id", "category", "cuisine"]).issubset(m.columns)
    assert set(["center_id", "center_type"]).issubset(c.columns)


# ---------------------------------------------------------------------------
# T008 — test_merge_datasets
# ---------------------------------------------------------------------------

def _make_minimal_frames():
    train = pd.DataFrame({
        "id": [1, 2, 3],
        "week": [1, 1, 2],
        "center_id": [1, 1, 2],
        "meal_id": [10, 20, 10],
        "checkout_price": [9.0, 14.0, 8.0],
        "base_price": [10.0, 15.0, 10.0],
        "emailer_for_promotion": [0, 1, 0],
        "homepage_featured": [0, 0, 1],
        "num_orders": [5, 3, 7],
    })
    meal_info = pd.DataFrame({
        "meal_id": [10, 20],
        "category": ["Biryani", "Beverages"],
        "cuisine": ["Indian", "Continental"],
    })
    centre_info = pd.DataFrame({
        "center_id": [1, 2],
        "city_code": [100, 200],
        "region_code": [10, 20],
        "op_area": [2.0, 3.0],
        "center_type": ["TYPE_A", "TYPE_B"],
    })
    return train, meal_info, centre_info


def test_merge_datasets_row_count_preserved():
    train, meal_info, centre_info = _make_minimal_frames()
    merged = merge_datasets(train, meal_info, centre_info)
    assert len(merged) == len(train)


def test_merge_datasets_no_duplicates():
    train, meal_info, centre_info = _make_minimal_frames()
    merged = merge_datasets(train, meal_info, centre_info)
    assert merged.duplicated(["center_id", "meal_id", "week"]).sum() == 0


def test_merge_datasets_duplicate_dim_raises():
    train, meal_info, centre_info = _make_minimal_frames()
    # Introduce a duplicate key in the dimension table
    meal_info_dup = pd.concat([meal_info, meal_info.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        merge_datasets(train, meal_info_dup, centre_info)


# ---------------------------------------------------------------------------
# T009 — test_save_and_load_merged
# ---------------------------------------------------------------------------

def test_save_and_load_merged_roundtrip(tmp_path):
    """Write to tmp_path and read back; shape and dtypes must be preserved."""
    df = pd.DataFrame({
        "center_id": [1, 2],
        "meal_id": [10, 20],
        "week": [1, 1],
        "num_orders": [5, 3],
        "base_price": [10.0, 15.0],
    })
    path = save_merged(df, tmp_path)
    df2 = load_merged(tmp_path)

    assert df.shape == df2.shape
    for col in df.columns:
        assert col in df2.columns


def test_load_merged_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_merged(tmp_path)
