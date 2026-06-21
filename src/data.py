"""Data loading and merging utilities for the CDI project."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")
SEED = 42
ACF_MAX_LAGS = 52
GRID_N_PAIRS = 9

_CSV_FILES = {
    "train": "train.csv",
    "meal_info": "meal_info.csv",
    "centre_info": "fulfilment_center_info.csv",
}


def load_raw_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train.csv, meal_info.csv, fulfilment_center_info.csv from data_dir."""
    paths = {k: Path(data_dir) / v for k, v in _CSV_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {missing}")
    return (
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["meal_info"]),
        pd.read_csv(paths["centre_info"]),
    )


def merge_datasets(
    train: pd.DataFrame,
    meal_info: pd.DataFrame,
    centre_info: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join train on meal_info and centre_info; assert row count preserved."""
    n_train = len(train)

    merged = train.merge(meal_info, on="meal_id", how="left")
    merged = merged.merge(centre_info, on="center_id", how="left")

    assert len(merged) == n_train, (
        f"Row count changed after merge: {n_train} → {len(merged)}. "
        "Dimension table likely has duplicate keys."
    )
    assert merged.duplicated(["center_id", "meal_id", "week"]).sum() == 0, (
        "Duplicate (center_id, meal_id, week) triplets found after merge."
    )

    merged["discount_rate"] = np.where(
        merged["base_price"] > 0,
        (merged["base_price"] - merged["checkout_price"]) / merged["base_price"],
        np.nan,
    )
    return merged


def save_merged(df: pd.DataFrame, processed_dir: Path) -> Path:
    """Write merged DataFrame to processed_dir/merged.parquet; return file path."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "merged.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


def load_merged(processed_dir: Path) -> pd.DataFrame:
    """Load merged.parquet from processed_dir."""
    path = Path(processed_dir) / "merged.parquet"
    if not path.exists():
        raise FileNotFoundError(f"merged.parquet not found at {path}")
    return pd.read_parquet(path, engine="pyarrow")


if __name__ == "__main__":
    train, meal_info, centre_info = load_raw_csvs(RAW_DIR)
    merged = merge_datasets(train, meal_info, centre_info)
    save_merged(merged, PROCESSED_DIR)
    print(f"Saved merged.parquet — {len(merged):,} rows.")
