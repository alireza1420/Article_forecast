from __future__ import annotations

"""ML direct-strategy forecasters: XGBoost, RandomForest, LightGBM (Cycle 4)."""

import matplotlib
matplotlib.use("Agg")

import itertools
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from features import FEATURE_COLS, HORIZON, make_horizon_target  # make_horizon_target added in T006
from baselines import evaluate_all, rmsle, save_predictions  # noqa: F401

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _ROOT / "results" / "models"
RESULTS_DIR: Path = _ROOT / "results"
FIGURES_DIR: Path = _ROOT / "results" / "figures"
PREDICTIONS_DIR: Path = _ROOT / "results" / "predictions"

# ---------------------------------------------------------------------------
# Constants (no magic numbers — Constitution Principle VI)
# ---------------------------------------------------------------------------
TRAIN_MAX_WEEK: int = 105
VAL_MIN_WEEK: int = 106
VAL_MAX_WEEK: int = 125
EVAL_MIN_WEEK: int = 126
EVAL_MAX_WEEK: int = 145
PRED_CUTOFF: int = 125
SEED: int = 42
NAN_FILL_RF: float = -1.0

# Hyperparameter grids (exhaustive; no Optuna — Constitution Principle I)
XGB_HPO_GRID: dict = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6, 8],
}  # 6 combinations

LGB_HPO_GRID: dict = {
    "learning_rate": [0.05, 0.1],
    "num_leaves": [63, 127],
}  # 4 combinations

# Default hyperparameters (may be overridden by HPO winner)
XGB_PARAMS: dict = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "nthread": -1,
    "seed": SEED,
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}

RF_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": SEED,
}

LGB_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": SEED,
    "verbosity": -1,
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMLForecaster(ABC):
    """Shared interface for direct-strategy ML forecasters (Cycle 4)."""

    name: str  # class-level; MUST match Constitution Principle IV approved names

    def __init__(self) -> None:
        self.models_: list = []

    @abstractmethod
    def fit_all(self, fm: pd.DataFrame) -> None:
        """Train 10 horizon-specific models on fm; populates self.models_."""

    @abstractmethod
    def predict_all(self, fm: pd.DataFrame) -> pd.DataFrame:
        """Return long-format predictions [center_id, meal_id, horizon, y_pred]."""

    def predict(self, fm: pd.DataFrame) -> pd.DataFrame:
        """Delegate to predict_all for evaluate_all() harness compatibility."""
        return self.predict_all(fm)

    @abstractmethod
    def save(self, models_dir: Path = MODELS_DIR) -> None:
        """Persist all 10 trained models to models_dir/{self.name}/."""

    @abstractmethod
    def load(self, models_dir: Path = MODELS_DIR) -> None:
        """Load all 10 models from models_dir/{self.name}/ into self.models_."""


# ---------------------------------------------------------------------------
# Shared anchor-row helper
# ---------------------------------------------------------------------------

def _anchor_rows(fm: pd.DataFrame) -> pd.DataFrame:
    """Return last row per (center_id, meal_id) with week <= PRED_CUTOFF."""
    sub = fm[fm["week"] <= PRED_CUTOFF]
    idx = sub.groupby(["center_id", "meal_id"])["week"].idxmax()
    return sub.loc[idx].reset_index(drop=True)


def _postprocess(raw: np.ndarray) -> np.ndarray:
    """Clip to 0 in log-space then expm1 to recover orders scale."""
    return np.expm1(np.clip(raw, 0.0, None))


# ---------------------------------------------------------------------------
# Placeholder classes — to be implemented in subsequent tasks
# ---------------------------------------------------------------------------

class XGBoostForecaster(BaseMLForecaster):
    """Direct multi-step XGBoost forecaster; HPO at h=1 reused for h=2..10."""

    name: str = "xgboost"

    def fit_all(self, fm: pd.DataFrame) -> None:
        """Train 10 XGBRegressors; HPO at h=1 selects best lr×max_depth for all horizons."""
        self.models_ = []
        best_hpo = run_hpo_xgboost(fm, RESULTS_DIR)
        params = {**XGB_PARAMS, **best_hpo}
        for h in range(1, HORIZON + 1):
            hdf = make_horizon_target(fm, h)
            train_df = hdf[hdf["week"] <= TRAIN_MAX_WEEK]
            val_df = hdf[(hdf["week"] >= VAL_MIN_WEEK) & (hdf["week"] <= VAL_MAX_WEEK)]
            X_train = train_df[FEATURE_COLS].values.astype(np.float32)
            y_train = train_df["y"].values.astype(np.float32)
            X_val = val_df[FEATURE_COLS].values.astype(np.float32)
            y_val = val_df["y"].values.astype(np.float32)
            model = xgb.XGBRegressor(**params)
            if len(X_val) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
            self.models_.append(model)

    def predict_all(self, fm: pd.DataFrame) -> pd.DataFrame:
        """Predict for all horizons; clip log-space pred to 0 then expm1."""
        anchors = _anchor_rows(fm)
        X_anchor = anchors[FEATURE_COLS].values.astype(np.float32)
        rows = []
        for h_idx, model in enumerate(self.models_):
            horizon = h_idx + 1
            raw = model.predict(X_anchor)
            y_pred = _postprocess(raw)
            for i, (_, anc) in enumerate(anchors.iterrows()):
                rows.append({
                    "center_id": int(anc["center_id"]),
                    "meal_id": int(anc["meal_id"]),
                    "horizon": horizon,
                    "y_pred": float(y_pred[i]),
                })
        return pd.DataFrame(rows)[["center_id", "meal_id", "horizon", "y_pred"]]

    def save(self, models_dir: Path = MODELS_DIR) -> None:
        """Save each booster as xgb_h{h:02d}.json in models_dir/xgboost/."""
        out_dir = Path(models_dir) / "xgboost"
        out_dir.mkdir(parents=True, exist_ok=True)
        for h_idx, model in enumerate(self.models_):
            model.save_model(str(out_dir / f"xgb_h{h_idx + 1:02d}.json"))

    def load(self, models_dir: Path = MODELS_DIR) -> None:
        """Load 10 boosters from models_dir/xgboost/xgb_h*.json."""
        in_dir = Path(models_dir) / "xgboost"
        self.models_ = []
        for h in range(1, HORIZON + 1):
            m = xgb.XGBRegressor()
            m.load_model(str(in_dir / f"xgb_h{h:02d}.json"))
            self.models_.append(m)


class RandomForestForecaster(BaseMLForecaster):
    """Direct multi-step RandomForest forecaster; fixed params, NaN fill = -1."""

    name: str = "random_forest"

    def fit_all(self, fm: pd.DataFrame) -> None:
        """Train 10 RF regressors; NaN features filled with NAN_FILL_RF sentinel."""
        self.models_ = []
        for h in range(1, HORIZON + 1):
            hdf = make_horizon_target(fm, h)
            train_df = hdf[hdf["week"] <= TRAIN_MAX_WEEK]
            X_train = train_df[FEATURE_COLS].fillna(NAN_FILL_RF).values.astype(np.float32)
            y_train = train_df["y"].values.astype(np.float32)
            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(X_train, y_train)
            self.models_.append(model)

    def predict_all(self, fm: pd.DataFrame) -> pd.DataFrame:
        """Predict for all horizons; fill NaN, clip, expm1."""
        anchors = _anchor_rows(fm)
        X_anchor = anchors[FEATURE_COLS].fillna(NAN_FILL_RF).values.astype(np.float32)
        rows = []
        for h_idx, model in enumerate(self.models_):
            horizon = h_idx + 1
            raw = model.predict(X_anchor)
            y_pred = _postprocess(raw)
            for i, (_, anc) in enumerate(anchors.iterrows()):
                rows.append({
                    "center_id": int(anc["center_id"]),
                    "meal_id": int(anc["meal_id"]),
                    "horizon": horizon,
                    "y_pred": float(y_pred[i]),
                })
        return pd.DataFrame(rows)[["center_id", "meal_id", "horizon", "y_pred"]]

    def save(self, models_dir: Path = MODELS_DIR) -> None:
        """Persist 10 models as rf_h{h:02d}.pkl via joblib."""
        out_dir = Path(models_dir) / "random_forest"
        out_dir.mkdir(parents=True, exist_ok=True)
        for h_idx, model in enumerate(self.models_):
            joblib.dump(model, out_dir / f"rf_h{h_idx + 1:02d}.pkl")

    def load(self, models_dir: Path = MODELS_DIR) -> None:
        """Load 10 joblib pkl files into self.models_."""
        in_dir = Path(models_dir) / "random_forest"
        self.models_ = [joblib.load(in_dir / f"rf_h{h:02d}.pkl") for h in range(1, HORIZON + 1)]


class LightGBMForecaster(BaseMLForecaster):
    """Direct multi-step LightGBM forecaster; early stopping on val weeks 106–125."""

    name: str = "lightgbm"

    def fit_all(self, fm: pd.DataFrame) -> None:
        """Train 10 LGBMRegressors; HPO at h=1 selects best lr×num_leaves for all horizons."""
        self.models_ = []
        best_hpo = run_hpo_lightgbm(fm, RESULTS_DIR)
        params = {**LGB_PARAMS, **best_hpo}
        for h in range(1, HORIZON + 1):
            hdf = make_horizon_target(fm, h)
            train_df = hdf[hdf["week"] <= TRAIN_MAX_WEEK]
            val_df = hdf[(hdf["week"] >= VAL_MIN_WEEK) & (hdf["week"] <= VAL_MAX_WEEK)]
            X_train = train_df[FEATURE_COLS].values.astype(np.float32)
            y_train = train_df["y"].values.astype(np.float32)
            model = lgb.LGBMRegressor(**params)
            if len(val_df) > 0:
                X_val = val_df[FEATURE_COLS].values.astype(np.float32)
                y_val = val_df["y"].values.astype(np.float32)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
                    )
            else:
                model.fit(X_train, y_train)
            self.models_.append(model)

    def predict_all(self, fm: pd.DataFrame) -> pd.DataFrame:
        """Predict for all horizons; clip, expm1."""
        anchors = _anchor_rows(fm)
        X_anchor = anchors[FEATURE_COLS].values.astype(np.float32)
        rows = []
        for h_idx, model in enumerate(self.models_):
            horizon = h_idx + 1
            raw = model.predict(X_anchor)
            y_pred = _postprocess(raw)
            for i, (_, anc) in enumerate(anchors.iterrows()):
                rows.append({
                    "center_id": int(anc["center_id"]),
                    "meal_id": int(anc["meal_id"]),
                    "horizon": horizon,
                    "y_pred": float(y_pred[i]),
                })
        return pd.DataFrame(rows)[["center_id", "meal_id", "horizon", "y_pred"]]

    def save(self, models_dir: Path = MODELS_DIR) -> None:
        """Persist 10 LGBMRegressors as lgbm_h{h:02d}.pkl via joblib."""
        out_dir = Path(models_dir) / "lightgbm"
        out_dir.mkdir(parents=True, exist_ok=True)
        for h_idx, model in enumerate(self.models_):
            joblib.dump(model, out_dir / f"lgbm_h{h_idx + 1:02d}.pkl")

    def load(self, models_dir: Path = MODELS_DIR) -> None:
        """Load 10 joblib pkl files into self.models_."""
        in_dir = Path(models_dir) / "lightgbm"
        self.models_ = [joblib.load(in_dir / f"lgbm_h{h:02d}.pkl") for h in range(1, HORIZON + 1)]


# ---------------------------------------------------------------------------
# Module-level functions — placeholders until implemented in later tasks
# ---------------------------------------------------------------------------

def run_hpo_xgboost(fm: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> dict:
    """Run 6-combo HPO grid for XGBoost at h=1; returns best params dict."""
    hdf = make_horizon_target(fm, h=1)
    train_df = hdf[hdf["week"] <= TRAIN_MAX_WEEK]
    val_df = hdf[(hdf["week"] >= VAL_MIN_WEEK) & (hdf["week"] <= VAL_MAX_WEEK)]
    X_tr, y_tr = train_df[FEATURE_COLS].values, train_df["y"].values
    X_val, y_val = val_df[FEATURE_COLS].values, val_df["y"].values

    rows = []
    for lr, depth in itertools.product(XGB_HPO_GRID["learning_rate"], XGB_HPO_GRID["max_depth"]):
        params = {**XGB_PARAMS, "learning_rate": lr, "max_depth": depth}
        model = xgb.XGBRegressor(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = np.expm1(np.clip(model.predict(X_val), 0, None))
        y_true = np.expm1(y_val)
        val_rmsle = rmsle(y_true, preds)
        rows.append({"learning_rate": lr, "max_depth": depth, "val_rmsle_h01": val_rmsle})

    df_hpo = pd.DataFrame(rows)
    best_idx = df_hpo["val_rmsle_h01"].idxmin()
    df_hpo["selected"] = False
    df_hpo.loc[best_idx, "selected"] = True

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df_hpo.to_csv(results_dir / "hpo_xgboost.csv", index=False)

    best = df_hpo.loc[best_idx]
    return {"learning_rate": float(best["learning_rate"]), "max_depth": int(best["max_depth"])}


def run_hpo_lightgbm(fm: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> dict:
    """Run 4-combo HPO grid for LightGBM at h=1; returns best params dict."""
    hdf = make_horizon_target(fm, h=1)
    train_df = hdf[hdf["week"] <= TRAIN_MAX_WEEK]
    val_df = hdf[(hdf["week"] >= VAL_MIN_WEEK) & (hdf["week"] <= VAL_MAX_WEEK)]
    X_tr, y_tr = train_df[FEATURE_COLS].values, train_df["y"].values
    X_val, y_val = val_df[FEATURE_COLS].values, val_df["y"].values

    rows = []
    for lr, leaves in itertools.product(LGB_HPO_GRID["learning_rate"], LGB_HPO_GRID["num_leaves"]):
        params = {**LGB_PARAMS, "learning_rate": lr, "num_leaves": leaves}
        model = lgb.LGBMRegressor(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
            )
        preds = np.expm1(np.clip(model.predict(X_val), 0, None))
        y_true = np.expm1(y_val)
        val_rmsle = rmsle(y_true, preds)
        rows.append({"learning_rate": lr, "num_leaves": leaves, "val_rmsle_h01": val_rmsle})

    df_hpo = pd.DataFrame(rows)
    best_idx = df_hpo["val_rmsle_h01"].idxmin()
    df_hpo["selected"] = False
    df_hpo.loc[best_idx, "selected"] = True

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df_hpo.to_csv(results_dir / "hpo_lightgbm.csv", index=False)

    best = df_hpo.loc[best_idx]
    return {"learning_rate": float(best["learning_rate"]), "num_leaves": int(best["num_leaves"])}


def plot_feature_importance(
    xgb_model: XGBoostForecaster,
    lgb_model: LightGBMForecaster,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Plot top-20 gain importance for XGBoost and LightGBM; returns PDF paths."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # XGBoost: average gain across 10 horizon boosters
    xgb_scores: dict[str, float] = {}
    for model in xgb_model.models_:
        scores = model.get_booster().get_score(importance_type="gain")
        for feat, val in scores.items():
            xgb_scores[feat] = xgb_scores.get(feat, 0.0) + val
    n_xgb = len(xgb_model.models_) or 1
    xgb_scores = {k: v / n_xgb for k, v in xgb_scores.items()}
    xgb_series = pd.Series(xgb_scores).sort_values(ascending=False).head(20)

    # LightGBM: average feature_importances_ arrays across 10 models
    lgb_arrays = [m.feature_importances_ for m in lgb_model.models_]
    lgb_mean = np.mean(lgb_arrays, axis=0)
    lgb_series = pd.Series(lgb_mean, index=FEATURE_COLS).sort_values(ascending=False).head(20)

    # Top-5 agreement
    top5_xgb = set(xgb_series.head(5).index)
    top5_lgb = set(lgb_series.head(5).index)
    agreement = len(top5_xgb & top5_lgb)
    print(f"Top-5 feature agreement between XGBoost and LightGBM: {agreement}/5")

    # XGBoost plot
    fig_xgb, ax_xgb = plt.subplots(figsize=(8, 6))
    xgb_series[::-1].plot(kind="barh", ax=ax_xgb)
    ax_xgb.set_title("XGBoost — Top-20 Feature Importance (Gain)")
    ax_xgb.set_xlabel("Mean Gain")
    plt.tight_layout()
    xgb_path = figures_dir / "feature_importance_xgb.pdf"
    fig_xgb.savefig(xgb_path)
    plt.close(fig_xgb)

    # LightGBM plot
    fig_lgb, ax_lgb = plt.subplots(figsize=(8, 6))
    lgb_series[::-1].plot(kind="barh", ax=ax_lgb)
    ax_lgb.set_title("LightGBM — Top-20 Feature Importance")
    ax_lgb.set_xlabel("Mean Importance")
    plt.tight_layout()
    lgb_path = figures_dir / "feature_importance_lgb.pdf"
    fig_lgb.savefig(lgb_path)
    plt.close(fig_lgb)

    return {"xgb": xgb_path, "lgb": lgb_path}


def merge_metric_matrices(baseline_path: Path, ml_path: Path) -> pd.DataFrame:
    """Concatenate baseline and ML metric CSVs into a single DataFrame."""
    baseline_df = pd.read_csv(Path(baseline_path), index_col="model")
    ml_df = pd.read_csv(Path(ml_path), index_col="model")
    return pd.concat([baseline_df, ml_df], axis=0)


def save_all_models_metric(df: pd.DataFrame, metric: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write all_models_{metric}.csv with model index; return path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"all_models_{metric}.csv"
    df.to_csv(path, index=True, index_label="model")
    return path


def save_all_models_rmsle(df: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> Path:
    """Backwards-compatible wrapper around save_all_models_metric for RMSLE."""
    return save_all_models_metric(df, "rmsle", results_dir)


def plot_all_models_metric(df: pd.DataFrame, metric: str, figures_dir: Path = FIGURES_DIR) -> Path:
    """Plot metric vs horizon for all models; save all_models_{metric}.pdf."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    horizons = list(range(1, HORIZON + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in df.index:
        vals = df.loc[model_name].values.astype(float)
        ax.plot(horizons, vals, marker="o", label=model_name)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(horizons)
    ax.legend(fontsize=8)
    ax.set_title(f"All Models: {metric.upper()} by Horizon")
    plt.tight_layout()
    path = figures_dir / f"all_models_{metric}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_all_models_comparison(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Backwards-compatible wrapper around plot_all_models_metric for RMSLE."""
    return plot_all_models_metric(df, "rmsle", figures_dir)


def save_ml_metrics(
    models: list,
    fm: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Evaluate models; write one CSV per metric; return dict of {metric: path}."""
    metrics = evaluate_all(models, fm)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for metric_name, metric_df in metrics.items():
        path = results_dir / f"{metric_name}_by_model_and_horizon.csv"
        metric_df.to_csv(path, index=True, index_label="model")
        paths[metric_name] = path
    return paths


def save_all_predictions(
    model: BaseMLForecaster,
    fm: pd.DataFrame,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> list[Path]:
    """Call save_predictions for h=1..HORIZON; return list of written paths."""
    preds = model.predict_all(fm)
    paths = []
    for h in range(1, HORIZON + 1):
        p = save_predictions(model, preds, h, fm, predictions_dir)
        paths.append(p)
    return paths


if __name__ == "__main__":
    from features import load_feature_matrix

    print("Loading feature matrix …")
    fm = load_feature_matrix()
    print(f"  Feature matrix shape: {fm.shape}")

    models: list[BaseMLForecaster] = [
        XGBoostForecaster(),
        RandomForestForecaster(),
        LightGBMForecaster(),
    ]

    for m in models:
        print(f"\nFitting {m.name} …")
        m.fit_all(fm)
        m.save()
        paths = save_all_predictions(m, fm)
        print(f"  {len(paths)} prediction CSVs written")

    print("\nEvaluating all models …")
    metric_paths = save_ml_metrics(models, fm)
    for name, p in metric_paths.items():
        print(f"  {name}_by_model_and_horizon.csv written: {p}")
    


    for metric, ml_path in metric_paths.items():
        baseline_path = RESULTS_DIR / f"baseline_{metric}.csv"
        if baseline_path.exists():
            merged = merge_metric_matrices(baseline_path, ml_path)
            save_all_models_metric(merged, metric)
            plot_all_models_metric(merged, metric)
            print(f"  all_models_{metric}.csv and plot written ({len(merged)} rows)")
        else:
            print(f"  Skipping merge for {metric} — {baseline_path} not found")
    

    print("\nPlotting feature importance …")
    xgb_m = next(m for m in models if m.name == "xgboost")
    lgb_m = next(m for m in models if m.name == "lightgbm")
    fi_paths = plot_feature_importance(xgb_m, lgb_m)
    print(f"  XGBoost importance PDF: {fi_paths['xgb']}")
    print(f"  LightGBM importance PDF: {fi_paths['lgb']}")
    print("\nCycle 4 complete.")
