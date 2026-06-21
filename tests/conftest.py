"""Shared pytest fixtures for EDA foundation tests."""
import numpy as np
import pandas as pd
import pytest

SEED = 42


@pytest.fixture
def merged_df() -> pd.DataFrame:
    """Synthetic merged DataFrame covering all edge cases needed by EDA tests."""
    rng = np.random.default_rng(SEED)

    # Four (center_id, meal_id) pairs across different weeks
    # Pair A: center=1, meal=10  — 10 weeks (full-ish history)
    # Pair B: center=1, meal=20  — 8 weeks
    # Pair C: center=2, meal=10  — 6 weeks
    # Pair D: center=2, meal=30  — 1 week (short history edge case)
    records = []

    pair_configs = [
        (1, 10, list(range(1, 11)),  "TYPE_A", "Indian",       "Biryani",    12.0),
        (1, 20, list(range(1, 9)),   "TYPE_A", "Continental",  "Beverages",  15.0),
        (2, 10, list(range(1, 7)),   "TYPE_B", "Indian",       "Rice Bowl",  10.0),
        (2, 30, [1],                 "TYPE_B", "Thai",         "Salads",     8.0),
    ]

    for center_id, meal_id, weeks, center_type, cuisine, category, base_price in pair_configs:
        for week in weeks:
            num_orders = int(rng.integers(0, 50))
            emailer = int(rng.integers(0, 2))
            homepage = int(rng.integers(0, 2))
            checkout_price = base_price * rng.uniform(0.8, 1.0)
            records.append({
                "center_id": center_id,
                "meal_id": meal_id,
                "week": week,
                "num_orders": num_orders,
                "base_price": base_price,
                "checkout_price": round(checkout_price, 2),
                "emailer_for_promotion": emailer,
                "homepage_featured": homepage,
                "center_type": center_type,
                "cuisine": cuisine,
                "category": category,
                "city_code": center_id * 100,
                "region_code": center_id * 10,
                "op_area": float(center_id * 2),
            })

    df = pd.DataFrame(records)

    # Force at least one zero-order row
    df.loc[df.index[0], "num_orders"] = 0

    # Force at least one base_price == 0 row (edge case for discount_rate)
    df.loc[df.index[1], "base_price"] = 0.0

    # Add discount_rate column (NaN where base_price == 0)
    df["discount_rate"] = np.where(
        df["base_price"] > 0,
        (df["base_price"] - df["checkout_price"]) / df["base_price"],
        np.nan,
    )

    # Ensure both emailer values (0 and 1) are present
    df.loc[df.index[2], "emailer_for_promotion"] = 0
    df.loc[df.index[3], "emailer_for_promotion"] = 1
    df.loc[df.index[4], "homepage_featured"] = 0
    df.loc[df.index[5], "homepage_featured"] = 1

    return df.reset_index(drop=True)


@pytest.fixture
def features_df() -> pd.DataFrame:
    """Synthetic merged DataFrame with deterministic num_orders for feature tests.

    Pair A (center=1, meal=10): weeks 1–145, num_orders = week * 2  → exact lag/rolling assertions.
    Pair B (center=2, meal=20): weeks 1–10,  num_orders = 5 (constant).
    One row has base_price=0.0 (discount_rate edge case).
    """
    records = []

    # Pair A — 145 weeks, num_orders = week * 2
    for week in range(1, 146):
        records.append({
            "center_id": 1,
            "meal_id": 10,
            "week": week,
            "num_orders": week * 2,
            "base_price": 10.0,
            "checkout_price": 8.0,
            "emailer_for_promotion": week % 2,
            "homepage_featured": 1 - (week % 2),
            "center_type": "TYPE_A",
            "category": "Biryani",
            "cuisine": "Indian",
            "city_code": 100,
            "region_code": 10,
            "op_area": 2.0,
            "discount_rate": (10.0 - 8.0) / 10.0,
        })

    # Pair B — 10 weeks, num_orders = 5 (constant, to test EWM/rolling stability)
    for week in range(1, 11):
        records.append({
            "center_id": 2,
            "meal_id": 20,
            "week": week,
            "num_orders": 5,
            "base_price": 10.0,
            "checkout_price": 8.0,
            "emailer_for_promotion": week % 2,
            "homepage_featured": 1 - (week % 2),
            "center_type": "TYPE_B",
            "category": "Beverages",
            "cuisine": "Continental",
            "city_code": 200,
            "region_code": 20,
            "op_area": 4.0,
            "discount_rate": (10.0 - 8.0) / 10.0,
        })

    df = pd.DataFrame(records)

    # Edge case: one row with base_price == 0 (discount_rate must become NaN)
    df.loc[df.index[0], "base_price"] = 0.0
    df.loc[df.index[0], "discount_rate"] = float("nan")

    return df.reset_index(drop=True)


@pytest.fixture
def baseline_fm() -> pd.DataFrame:
    """Synthetic feature matrix for baselines tests.

    Pair A (center=1, meal=10): weeks 1–145, num_orders=week*2.
      lag_10 at week w = (w-10)*2 for w>10, else NaN.
    Pair B (center=2, meal=20): weeks 1–30, num_orders=5 (constant).
      lag_10 = 5.0 for weeks > 10, else NaN.
    """
    records = []

    # Pair A — 145 weeks, num_orders = week * 2
    for week in range(1, 146):
        lag_10 = float((week - 10) * 2) if week > 10 else float("nan")
        lag_11 = float((week - 11) * 2) if week > 11 else float("nan")
        lag_12 = float((week - 12) * 2) if week > 12 else float("nan")
        lag_13 = float((week - 13) * 2) if week > 13 else float("nan")
        ewm_span10 = lag_10  # approximate; sufficient for LR fixture tests
        records.append({
            "center_id": 1,
            "meal_id": 10,
            "week": week,
            "num_orders": float(week * 2),
            "lag_10": lag_10,
            "lag_11": lag_11,
            "lag_12": lag_12,
            "lag_13": lag_13,
            "ewm_span10": ewm_span10,
        })

    # Pair B — 30 weeks, num_orders = 5 (constant)
    for week in range(1, 31):
        lag_10 = 5.0 if week > 10 else float("nan")
        lag_11 = 5.0 if week > 11 else float("nan")
        lag_12 = 5.0 if week > 12 else float("nan")
        lag_13 = 5.0 if week > 13 else float("nan")
        ewm_span10 = lag_10
        records.append({
            "center_id": 2,
            "meal_id": 20,
            "week": week,
            "num_orders": 5.0,
            "lag_10": lag_10,
            "lag_11": lag_11,
            "lag_12": lag_12,
            "lag_13": lag_13,
            "ewm_span10": ewm_span10,
        })

    return pd.DataFrame(records).reset_index(drop=True)


@pytest.fixture
def ml_fm() -> pd.DataFrame:
    """Synthetic feature matrix for ml_models tests with all 31 FEATURE_COLS present.

    Pair A (center=1, meal=10): weeks 1–145, num_orders=week*2.
      Lag features NaN for early weeks; rolling/ewm similarly NaN-sparse.
    Pair B (center=2, meal=20): weeks 1–30, num_orders=5 (constant).
    Categorical _enc columns are int (0). All FEATURE_COLS columns present.
    """
    from src.features import FEATURE_COLS  # import here to avoid circular at module load

    records = []

    def _lag(orders_map: dict, w: int, shift: int) -> float:
        v = orders_map.get(w - shift)
        return float(v) if v is not None else float("nan")

    def _rolling_mean(orders_map: dict, w: int, window: int) -> float:
        vals = [orders_map[w - i] for i in range(1, window + 1) if (w - i) in orders_map]
        return float(np.mean(vals)) if len(vals) >= window else float("nan")

    def _rolling_std(orders_map: dict, w: int, window: int) -> float:
        vals = [orders_map[w - i] for i in range(1, window + 1) if (w - i) in orders_map]
        return float(np.std(vals, ddof=1)) if len(vals) >= window else float("nan")

    def _rolling_min(orders_map: dict, w: int, window: int) -> float:
        vals = [orders_map[w - i] for i in range(1, window + 1) if (w - i) in orders_map]
        return float(np.min(vals)) if len(vals) >= window else float("nan")

    def _rolling_max(orders_map: dict, w: int, window: int) -> float:
        vals = [orders_map[w - i] for i in range(1, window + 1) if (w - i) in orders_map]
        return float(np.max(vals)) if len(vals) >= window else float("nan")

    def _ewm(orders_map: dict, w: int, span: int) -> float:
        # Use simple approximation: last known value for weeks with sufficient history
        lag = w - 1
        if lag not in orders_map:
            return float("nan")
        return float(orders_map[lag])

    # Pair A: weeks 1–145, num_orders = week * 2
    orders_a = {w: w * 2 for w in range(1, 146)}
    for week in range(1, 146):
        n = float(week * 2)
        # center/meal aggregates based on all training weeks
        cma = float(np.mean(list(orders_a.values())))
        cstd = float(np.std(list(orders_a.values()), ddof=1))
        mma = cma
        mstd = cstd
        records.append({
            "center_id": 1,
            "meal_id": 10,
            "week": week,
            "num_orders": n,
            "lag_10": _lag(orders_a, week, 10),
            "lag_11": _lag(orders_a, week, 11),
            "lag_12": _lag(orders_a, week, 12),
            "lag_13": _lag(orders_a, week, 13),
            "rolling_mean_4w": _rolling_mean(orders_a, week, 4),
            "rolling_std_4w": _rolling_std(orders_a, week, 4),
            "rolling_min_4w": _rolling_min(orders_a, week, 4),
            "rolling_max_4w": _rolling_max(orders_a, week, 4),
            "rolling_mean_8w": _rolling_mean(orders_a, week, 8),
            "rolling_std_8w": _rolling_std(orders_a, week, 8),
            "rolling_min_8w": _rolling_min(orders_a, week, 8),
            "rolling_max_8w": _rolling_max(orders_a, week, 8),
            "ewm_span10": _ewm(orders_a, week, 10),
            "ewm_span13": _ewm(orders_a, week, 13),
            "ewm_span26": _ewm(orders_a, week, 26),
            "discount_rate": 0.2,
            "log_checkout_price": float(np.log1p(8.0)),
            "emailer_for_promotion": week % 2,
            "homepage_featured": 1 - (week % 2),
            "email_x_discount": (week % 2) * 0.2,
            "homepage_x_discount": (1 - week % 2) * 0.2,
            "week_number": float(week),
            "week_of_year_sin": float(np.sin(2 * np.pi * week / 52)),
            "week_of_year_cos": float(np.cos(2 * np.pi * week / 52)),
            "center_mean_orders": cma,
            "center_std_orders": cstd,
            "meal_mean_orders": mma,
            "meal_std_orders": mstd,
            "center_type_enc": 0,
            "category_enc": 0,
            "cuisine_enc": 0,
        })

    # Pair B: weeks 1–30, num_orders = 5 (constant)
    orders_b = {w: 5 for w in range(1, 31)}
    for week in range(1, 31):
        records.append({
            "center_id": 2,
            "meal_id": 20,
            "week": week,
            "num_orders": 5.0,
            "lag_10": _lag(orders_b, week, 10),
            "lag_11": _lag(orders_b, week, 11),
            "lag_12": _lag(orders_b, week, 12),
            "lag_13": _lag(orders_b, week, 13),
            "rolling_mean_4w": _rolling_mean(orders_b, week, 4),
            "rolling_std_4w": _rolling_std(orders_b, week, 4),
            "rolling_min_4w": _rolling_min(orders_b, week, 4),
            "rolling_max_4w": _rolling_max(orders_b, week, 4),
            "rolling_mean_8w": _rolling_mean(orders_b, week, 8),
            "rolling_std_8w": _rolling_std(orders_b, week, 8),
            "rolling_min_8w": _rolling_min(orders_b, week, 8),
            "rolling_max_8w": _rolling_max(orders_b, week, 8),
            "ewm_span10": _ewm(orders_b, week, 10),
            "ewm_span13": _ewm(orders_b, week, 13),
            "ewm_span26": _ewm(orders_b, week, 26),
            "discount_rate": 0.2,
            "log_checkout_price": float(np.log1p(8.0)),
            "emailer_for_promotion": week % 2,
            "homepage_featured": 1 - (week % 2),
            "email_x_discount": (week % 2) * 0.2,
            "homepage_x_discount": (1 - week % 2) * 0.2,
            "week_number": float(week),
            "week_of_year_sin": float(np.sin(2 * np.pi * week / 52)),
            "week_of_year_cos": float(np.cos(2 * np.pi * week / 52)),
            "center_mean_orders": 5.0,
            "center_std_orders": 0.0,
            "meal_mean_orders": 5.0,
            "meal_std_orders": 0.0,
            "center_type_enc": 1,
            "category_enc": 1,
            "cuisine_enc": 1,
        })

    df = pd.DataFrame(records).reset_index(drop=True)
    # Verify all required columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert not missing, f"ml_fm fixture missing columns: {missing}"
    return df
