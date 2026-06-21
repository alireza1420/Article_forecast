from __future__ import annotations

"""Statistical baseline forecasters: Naive, SeasonalNaive, SMA, SES, LinearReg."""

import matplotlib
matplotlib.use("Agg")

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
PROCESSED_DIR: Path = _ROOT / "data" / "processed"
RESULTS_DIR: Path = _ROOT / "results"
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"
MODELS_DIR: Path = RESULTS_DIR / "models"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Constants (no magic numbers — Principle VI)
# ---------------------------------------------------------------------------
TRAIN_MAX_WEEK: int = 105
PRED_CUTOFF: int = 125
EVAL_MIN_WEEK: int = 126
EVAL_MAX_WEEK: int = 145
HORIZON: int = 10
SMA_WINDOW: int = 4
SES_DEFAULT_ALPHA: float = 0.3
LR_FEATURES: list[str] = ["lag_10", "lag_11", "lag_12", "lag_13", "ewm_span10"]
SEED: int = 42
MODEL_NAMES: list[str] = ["naive", "seasonal_naive", "sma", "ses", "linreg"]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseForecaster(ABC):
    """Shared interface for all statistical baseline forecasters."""

    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train on feature matrix; stateless models treat this as a no-op."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return long-format predictions: [center_id, meal_id, horizon, y_pred]."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error — numpy scratch per Constitution Principle V."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        return np.nan
    clipped = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(clipped) - np.log1p(y_true)) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolue Error — numpy scratch."""
    y_true=np.asarray(y_true, dtype=float)
    y_pred=np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        return np.nan
    clipped = np.clip(y_pred, 0.0, None)
    return float(np.mean(np.abs(clipped - y_true)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — numpy scratch."""
    y_true=np.asarray(y_true, dtype=float)
    y_pred=np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        return np.nan
    clipped = np.clip(y_pred, 0.0, None)
    nonzero_mask = y_true != 0
    if not np.any(nonzero_mask):
        return np.nan
    return float(np.mean(np.abs(clipped[nonzero_mask] - y_true[nonzero_mask]) / np.abs(y_true[nonzero_mask])))

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error — numpy scratch."""
    y_true=np.asarray(y_true, dtype=float)
    y_pred=np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        return np.nan
    clipped = np.clip(y_pred, 0.0, None)
    denominator = np.abs(y_true) + np.abs(clipped)
    nonzero_mask = denominator != 0
    if not np.any(nonzero_mask):
        return np.nan
    return float(np.mean(2 * np.abs(clipped[nonzero_mask] - y_true[nonzero_mask]) / denominator[nonzero_mask]))-1.0



def load_feature_matrix(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load feature_matrix.parquet from processed_dir."""
    return pd.read_parquet(Path(processed_dir) / "feature_matrix.parquet")


def _anchor_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last row per (center_id, meal_id) pair with week ≤ PRED_CUTOFF."""
    sub = df[df["week"] <= PRED_CUTOFF]
    idx = sub.groupby(["center_id", "meal_id"])["week"].idxmax()
    return sub.loc[idx].reset_index(drop=True)


def _expand_horizons(anchors: pd.DataFrame, value_col: str = "y_pred") -> pd.DataFrame:
    """Cross-join anchor rows with horizons 1..HORIZON to produce long-format output."""
    horizons = pd.DataFrame({"horizon": range(1, HORIZON + 1)})
    expanded = anchors.assign(_key=1).merge(horizons.assign(_key=1), on="_key").drop(columns="_key")
    expanded = expanded.rename(columns={value_col: "y_pred"}) if value_col != "y_pred" else expanded
    return expanded[["center_id", "meal_id", "horizon", "y_pred"]].copy()


# ---------------------------------------------------------------------------
# Naive Forecaster
# ---------------------------------------------------------------------------

class NaiveForecaster(BaseForecaster):
    """Flat prediction: last known num_orders for all horizons."""

    name = "naive"

    def fit(self, df: pd.DataFrame) -> None:
        """No-op — Naive requires no training."""

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict last known num_orders (anchor row) for h=1..HORIZON per pair."""
        anchors = _anchor_rows(df)[["center_id", "meal_id", "num_orders"]].copy()
        anchors["y_pred"] = np.clip(anchors["num_orders"], 0.0, None)
        return _expand_horizons(anchors.drop(columns=["num_orders"]))


# ---------------------------------------------------------------------------
# Seasonal Naive Forecaster
# ---------------------------------------------------------------------------

class SeasonalNaiveForecaster(BaseForecaster):
    """Prediction = num_orders at week W−52; fallback to Naive when unavailable."""

    name = "seasonal_naive"

    def fit(self, df: pd.DataFrame) -> None:
        """No-op."""

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict same-week-last-year demand; fall back to Naive on missing history."""
        anchors = _anchor_rows(df)[["center_id", "meal_id", "week", "num_orders"]].copy()
        anchors = anchors.rename(columns={"week": "anchor_week", "num_orders": "naive_val"})

        # Build seasonal lookup: num_orders at week W−52 per pair
        anchors["seasonal_week"] = anchors["anchor_week"] - 52
        lookup = df[["center_id", "meal_id", "week", "num_orders"]].copy()
        merged = anchors.merge(
            lookup.rename(columns={"week": "seasonal_week", "num_orders": "seasonal_val"}),
            on=["center_id", "meal_id", "seasonal_week"],
            how="left",
        )

        # Fallback: seasonal_week < 1 OR no historical row found → use naive_val
        mask_fallback = (merged["seasonal_week"] < 1) | merged["seasonal_val"].isna()
        merged["y_pred"] = np.where(mask_fallback, merged["naive_val"], merged["seasonal_val"])
        merged["y_pred"] = np.clip(merged["y_pred"], 0.0, None)

        return _expand_horizons(merged[["center_id", "meal_id", "y_pred"]])


# ---------------------------------------------------------------------------
# Simple Moving Average Forecaster
# ---------------------------------------------------------------------------

class SMAForecaster(BaseForecaster):
    """Prediction = mean of last SMA_WINDOW known num_orders per pair."""

    name = "sma"

    def fit(self, df: pd.DataFrame) -> None:
        """No-op."""

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict mean of last SMA_WINDOW weeks (or all available if fewer) per pair."""
        sub = df[df["week"] <= PRED_CUTOFF][["center_id", "meal_id", "week", "num_orders"]].copy()
        sub = sub.sort_values(["center_id", "meal_id", "week"])

        def _sma(grp: pd.DataFrame) -> float:
            vals = grp["num_orders"].values[-SMA_WINDOW:]
            return float(np.mean(vals))

        means = (
            sub.groupby(["center_id", "meal_id"])
            .apply(_sma)
            .reset_index(name="y_pred")
        )
        means["y_pred"] = np.clip(means["y_pred"], 0.0, None)
        return _expand_horizons(means)


# ---------------------------------------------------------------------------
# Simple Exponential Smoothing Forecaster
# ---------------------------------------------------------------------------

class SESForecaster(BaseForecaster):
    """SES with per-pair optimised alpha; persists fitted alphas to CSV."""

    name = "ses"

    def __init__(self) -> None:
        self.alpha_df_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> None:
        """Fit alpha per pair on weeks ≤ TRAIN_MAX_WEEK; save to ses_alpha.csv."""
        train = df[df["week"] <= TRAIN_MAX_WEEK][["center_id", "meal_id", "week", "num_orders"]].copy()
        train = train.sort_values(["center_id", "meal_id", "week"])

        rows = []
        for (cid, mid), grp in train.groupby(["center_id", "meal_id"]):
            y = grp["num_orders"].values
            if len(y) < 2:
                alpha = SES_DEFAULT_ALPHA
            else:
                def _ses_mse(alpha: float) -> float:
                    level = y[0]
                    errors = []
                    for obs in y[1:]:
                        pred = level
                        errors.append((pred - obs) ** 2)
                        level = alpha * obs + (1 - alpha) * level
                    return float(np.mean(errors))

                try:
                    res = minimize_scalar(_ses_mse, method="bounded", bounds=(1e-4, 1.0))
                    alpha = float(np.clip(res.x, 1e-4, 1.0))
                except Exception:
                    alpha = SES_DEFAULT_ALPHA

            rows.append({"center_id": cid, "meal_id": mid, "alpha": alpha})

        self.alpha_df_ = pd.DataFrame(rows)
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_df_.to_csv(results_dir / "ses_alpha.csv", index=False)

    def predict(self, df: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
        """Recursively smooth num_orders through anchor week; return flat forecast."""
        if self.alpha_df_ is None:
            alpha_path = Path(results_dir) / "ses_alpha.csv"
            self.alpha_df_ = pd.read_csv(alpha_path)

        alpha_map = {
            (row.center_id, row.meal_id): row.alpha
            for row in self.alpha_df_.itertuples()
        }

        sub = df[df["week"] <= PRED_CUTOFF][["center_id", "meal_id", "week", "num_orders"]].copy()
        sub = sub.sort_values(["center_id", "meal_id", "week"])

        rows = []
        for (cid, mid), grp in sub.groupby(["center_id", "meal_id"]):
            alpha = alpha_map.get((cid, mid), SES_DEFAULT_ALPHA)
            y = grp["num_orders"].values
            level = float(y[0])
            for obs in y[1:]:
                level = alpha * float(obs) + (1 - alpha) * level
            rows.append({"center_id": cid, "meal_id": mid, "y_pred": max(0.0, level)})

        result = pd.DataFrame(rows)
        return _expand_horizons(result)


# ---------------------------------------------------------------------------
# Linear Regression Forecaster
# ---------------------------------------------------------------------------

class LinearRegressionForecaster(BaseForecaster):
    """Global LinearRegression per horizon on lag features; persisted via joblib."""

    name = "linreg"

    def __init__(self) -> None:
        self.models_: list[LinearRegression] = []

    def fit(
        self, df: pd.DataFrame, models_dir: Path = MODELS_DIR
    ) -> None:
        """Fit 10 global LR models (one per horizon) and save as pkl files."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        self.models_ = []

        for h in range(1, HORIZON + 1):
            cutoff = TRAIN_MAX_WEEK - h
            train = df[df["week"] <= cutoff].copy()

            # Drop rows with NaN in LR features
            train = train.dropna(subset=LR_FEATURES)

            # Build target: num_orders at week + h for same pair
            target = (
                df[["center_id", "meal_id", "week", "num_orders"]]
                .rename(columns={"week": "target_week", "num_orders": "y"})
            )
            train["target_week"] = train["week"] + h
            merged = train.merge(target, on=["center_id", "meal_id", "target_week"], how="inner")
            merged = merged.dropna(subset=["y"])

            if len(merged) < 2:
                # Not enough data — fit on zeros as fallback
                model = LinearRegression()
                model.fit(np.zeros((1, len(LR_FEATURES))), [0.0])
            else:
                X = merged[LR_FEATURES].values
                y = merged["y"].values
                model = LinearRegression()
                model.fit(X, y)

            self.models_.append(model)
            joblib.dump(model, models_dir / f"linreg_h{h:02d}.pkl")

    def predict(
        self, df: pd.DataFrame, models_dir: Path = MODELS_DIR
    ) -> pd.DataFrame:
        """Predict using fitted global LR models; NaN-feature pairs get y_pred=0."""
        if not self.models_:
            models_dir = Path(models_dir)
            self.models_ = [
                joblib.load(models_dir / f"linreg_h{h:02d}.pkl") for h in range(1, HORIZON + 1)
            ]

        anchors = _anchor_rows(df)[["center_id", "meal_id"] + LR_FEATURES].copy()

        rows = []
        for _, row in anchors.iterrows():
            cid, mid = row["center_id"], row["meal_id"]
            feat = row[LR_FEATURES].values
            for h_idx, model in enumerate(self.models_):
                horizon = h_idx + 1
                if np.any(np.isnan(feat)):
                    y_pred = 0.0
                else:
                    y_pred = float(model.predict([feat])[0])
                y_pred = max(0.0, y_pred)
                rows.append({"center_id": cid, "meal_id": mid, "horizon": horizon, "y_pred": y_pred})

        return pd.DataFrame(rows)[["center_id", "meal_id", "horizon", "y_pred"]]


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_all(
    models: list[BaseForecaster],
    fm: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute all metric matrices on the eval set.

    Returns a dict with keys "rmsle", "mae", "mape", "smape".
    Each value is a DataFrame of shape (n_models, HORIZON) with
    columns h01..h10 and model names as the index.
    """
    actuals = fm[["center_id", "meal_id", "week", "num_orders"]].copy()
    anchor_wks = (
        fm[fm["week"] <= PRED_CUTOFF]
        .groupby(["center_id", "meal_id"])["week"]
        .max()
        .reset_index()
        .rename(columns={"week": "anchor_week"})
    )

    metric_fns = {"rmsle": rmsle, "mae": mae, "mape": mape, "smape": smape}
    # rows[metric][model_name] = list of HORIZON values
    rows: dict[str, dict[str, list[float]]] = {m: {} for m in metric_fns}

    for model in models:
        preds = model.predict(fm)
        per_h: dict[str, list[float]] = {m: [] for m in metric_fns}

        for h in range(1, HORIZON + 1):
            h_preds = preds[preds["horizon"] == h].copy()
            h_preds = h_preds.merge(anchor_wks, on=["center_id", "meal_id"], how="left")
            h_preds["target_week"] = h_preds["anchor_week"] + h

            joined = h_preds.merge(
                actuals.rename(columns={"week": "target_week", "num_orders": "y_true"}),
                on=["center_id", "meal_id", "target_week"],
                how="inner",
            )

            for metric_name, fn in metric_fns.items():
                if len(joined) == 0:
                    per_h[metric_name].append(np.nan)
                else:
                    per_h[metric_name].append(
                        fn(joined["y_true"].values, joined["y_pred"].values)
                    )

        for metric_name in metric_fns:
            rows[metric_name][model.name] = per_h[metric_name]

    cols = [f"h{h:02d}" for h in range(1, HORIZON + 1)]
    return {
        metric_name: pd.DataFrame(rows[metric_name], index=cols).T
        for metric_name in metric_fns
    }


def save_predictions(
    model: BaseForecaster,
    predictions: pd.DataFrame,
    horizon: int,
    fm: pd.DataFrame,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> Path:
    """Write pred_{model.name}_h{horizon:02d}.csv following the constitution contract."""
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    h_preds = predictions[predictions["horizon"] == horizon].copy()

     # Compute target week per pair (anchor_week + horizon)
    anchor_wks = (
        fm[fm["week"] <= PRED_CUTOFF]
        .groupby(["center_id", "meal_id"])["week"]
        .max()
        .reset_index()
        .rename(columns={"week": "anchor_week"})
    )
    h_preds = h_preds.merge(anchor_wks, on=["center_id", "meal_id"], how="left")
    h_preds["week"] = h_preds["anchor_week"] + horizon
    h_preds["split"] = "eval"

    out = h_preds[["center_id", "meal_id", "week", "y_pred", "split"]].reset_index(drop=True)
    path = predictions_dir / f"pred_{model.name}_h{horizon:02d}.csv"
    out.to_csv(path, index=False)
    return path


def save_metrics_matrices(
    metrics: dict[str, pd.DataFrame],
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Write one CSV per metric to results_dir/baseline_{metric}.csv.

    Returns a dict mapping metric name → written file path.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for metric_name, df in metrics.items():
        path = results_dir / f"baseline_{metric_name}.csv"
        df.to_csv(path, index=True, index_label="model")
        paths[metric_name] = path
    return paths


def save_rmsle_matrix(
    rmsle_df: pd.DataFrame, results_dir: Path = RESULTS_DIR
) -> Path:
    """Write baseline_rmsle.csv with model index and h01..h10 columns."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "baseline_rmsle.csv"
    rmsle_df.to_csv(path, index=True, index_label="model")
    return path


_METRIC_LABELS: dict[str, str] = {
    "rmsle": "RMSLE",
    "mae": "MAE",
    "mape": "MAPE",
    "smape": "sMAPE",
}


def plot_metric_comparison(
    metric_df: pd.DataFrame,
    metric_name: str,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot metric vs horizon for all models; save as baseline_{metric_name}.pdf."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = list(range(1, HORIZON + 1))
    for model_name in metric_df.index:
        vals = metric_df.loc[model_name].values.astype(float)
        ax.plot(horizons, vals, marker="o", label=model_name)

    ylabel = _METRIC_LABELS.get(metric_name, metric_name.upper())
    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons)
    ax.legend()
    sns.despine()

    path = figures_dir / f"baseline_{metric_name}.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all_comparisons(
    metrics: dict[str, pd.DataFrame],
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Generate one comparison PDF per metric; return dict of paths."""
    return {
        metric_name: plot_metric_comparison(df, metric_name, figures_dir)
        for metric_name, df in metrics.items()
    }


def plot_baseline_comparison(
    rmsle_df: pd.DataFrame, figures_dir: Path = FIGURES_DIR
) -> Path:
    """Plot RMSLE vs horizon for all models; save as baseline_comparison.pdf."""
    return plot_metric_comparison(rmsle_df, "rmsle", figures_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("[baselines] Loading feature matrix...")
    fm = load_feature_matrix()
    print(f"[baselines] Feature matrix: {fm.shape[0]:,} rows × {fm.shape[1]} cols")

    models: list[BaseForecaster] = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(),
        SMAForecaster(),
        SESForecaster(),
        LinearRegressionForecaster(),
    ]

    for m in models:
        print(f"[baselines] Fitting {m.name}...")
        m.fit(fm)

    print("[baselines] Generating predictions and saving files...")
    for m in models:
        preds = m.predict(fm)
        for h in range(1, HORIZON + 1):
            save_predictions(m, preds, h, fm)

    print("[baselines] Computing metric matrices (RMSLE, MAE, MAPE, sMAPE)...")
    metrics = evaluate_all(models, fm)
    csv_paths = save_metrics_matrices(metrics)
    for metric_name, path in csv_paths.items():
        print(f"[baselines]   {metric_name}: {path}")

    print("[baselines] Generating comparison charts...")
    fig_paths = plot_all_comparisons(metrics)
    for metric_name, path in fig_paths.items():
        print(f"[baselines]   {metric_name}: {path}")

    print("\n[baselines] RMSLE Summary:")
    print(metrics["rmsle"].round(4).to_string())
    print("\n[baselines] MAE Summary:")
    print(metrics["mae"].round(2).to_string())
    print("\n[baselines] Done.")
    sys.exit(0)
