"""Evaluation metrics and visualisation for demand forecasting models."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES_DIR: Path = Path("results/tables")
FIGURES_DIR: Path = Path("results/figures")
PREDICTIONS_DIR: Path = Path("results/predictions")
SEQ_DIR: Path = Path("data/processed/sequences")


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-horizon RMSLE; shape (10,). Numpy scratch per Constitution V."""
    n_h = y_true.shape[1]
    result = np.empty(n_h)
    for h in range(n_h):
        log_diff = (
            np.log1p(np.clip(y_pred[:, h], 0, None)) - np.log1p(y_true[:, h])
        )
        result[h] = np.sqrt(np.mean(log_diff ** 2))
    return result


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-horizon MAE; shape (10,)."""
    return np.mean(np.abs(y_true - y_pred), axis=0)


def compute_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-horizon MAPE excluding y_true < 1; returns (mape (10,), n_excluded (10,))."""
    n_h = y_true.shape[1]
    mape = np.empty(n_h)
    n_excluded = np.empty(n_h)
    for h in range(n_h):
        mask = y_true[:, h] >= 1.0
        n_excluded[h] = float((~mask).sum())
        if mask.sum() > 0:
            mape[h] = float(
                np.mean(
                    np.abs(y_true[mask, h] - y_pred[mask, h]) / y_true[mask, h]
                ) * 100.0
            )
        else:
            mape[h] = np.nan
    return mape, n_excluded


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-horizon SMAPE; shape (10,). Formula: mean(|y-ŷ| / ((|y|+|ŷ|)/2)) × 100."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.where(denominator == 0, 0.0, numerator / np.where(denominator == 0, 1.0, denominator))
    return np.mean(ratio, axis=0) * 100.0


# ── CSV writers ────────────────────────────────────────────────────────────────

def save_metrics_csv(results: dict, path: Path) -> None:
    """Write 5-row × 11-column metrics_lstm.csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = ["rmsle", "mae", "mape", "smape", "mape_n_excluded"]
    header = ["metric"] + [f"h{h + 1}" for h in range(10)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key in metric_keys:
            writer.writerow([key] + [float(v) for v in results[key]])


def append_experiment_log(
    config: "LSTMArchConfig",
    config_name: str,
    val_rmsle: float,
    log_path: Path,
) -> None:
    """Append one row to lstm_arch_experiments.csv; create with header if absent."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["config_name", "config_json", "val_rmsle"])
        writer.writerow([config_name, config.to_json(), float(val_rmsle)])


# ── evaluate_lstm ─────────────────────────────────────────────────────────────

def evaluate_lstm(
    config: "LSTMArchConfig",
    checkpoint_path: Path,
    seq_dir: Path = SEQ_DIR,
    tables_dir: Path = TABLES_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> dict:
    """Load checkpoint; predict eval split; compute 4 metrics; write CSVs."""
    import torch
    from torch.utils.data import DataLoader
    from lstm_model import DemandDataset, build_model

    checkpoint_path = Path(checkpoint_path)
    tables_dir = Path(tables_dir)
    predictions_dir = Path(predictions_dir)
    seq_dir = Path(seq_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    eval_ds = DemandDataset("eval", seq_dir=seq_dir)
    loader = DataLoader(eval_ds, batch_size=256, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for xt, xs, yb in loader:
            preds.append(model(xt.to(device), xs.to(device)).cpu().numpy())
            targets.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)

    mape_vals, n_excluded = compute_mape(y_true, y_pred)
    results = {
        "rmsle": compute_rmsle(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": mape_vals,
        "smape": compute_smape(y_true, y_pred),
        "mape_n_excluded": n_excluded,
    }
    save_metrics_csv(results, tables_dir / "metrics_lstm.csv")

    # Prediction CSVs: use eval_meta.npy if available, else zero placeholders
    meta_path = seq_dir / "eval_meta.npy"
    if meta_path.exists():
        meta = np.load(meta_path)  # (N_eval, 3): [center_id, meal_id, anchor_week]
    else:
        print("[evaluate_lstm] Warning: eval_meta.npy absent; using placeholder ids")
        meta = np.zeros((len(y_pred), 3), dtype=np.float32)

    for h in range(10):
        rows = [
            {
                "center_id": int(meta[i, 0]),
                "meal_id": int(meta[i, 1]),
                "week": int(meta[i, 2]),
                "y_pred": float(y_pred[i, h]),
                "split": "eval",
            }
            for i in range(len(y_pred))
        ]
        pd.DataFrame(rows).to_csv(
            predictions_dir / f"pred_lstm_h{h + 1:02d}.csv", index=False
        )

    return results


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: list[float],
    val_rmsles: list[float],
    early_stop_epoch: int | None,
    out_path: Path,
) -> None:
    """Two-panel PDF: train loss (top) + val RMSLE (bottom); dashed vline at early-stop."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(epochs, train_losses)
    axes[0].set_ylabel("Train Loss (RMSLE)")
    axes[0].set_title("Training Curves — LSTM")
    axes[1].plot(epochs, val_rmsles)
    axes[1].set_ylabel("Val RMSLE")
    axes[1].set_xlabel("Epoch")
    if early_stop_epoch is not None:
        for ax in axes:
            ax.axvline(
                x=early_stop_epoch + 1,
                color="red",
                linestyle="--",
                label=f"Early stop (ep {early_stop_epoch + 1})",
            )
        axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_lstm_vs_ml(
    lstm_results: dict,
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Four metric comparison PDFs: LSTM vs best-ML (auto-selected by h=1 RMSLE)."""
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rmsle_path = tables_dir.parent / "all_models_rmsle.csv"
    if not rmsle_path.exists():
        print(f"[plot_lstm_vs_ml] {rmsle_path} not found; skipping comparison figures")
        return

    df_rmsle = pd.read_csv(rmsle_path, index_col="model")
    ml_candidates = [m for m in ("xgboost", "lightgbm") if m in df_rmsle.index]
    best_ml = df_rmsle.loc[ml_candidates, "h01"].idxmin()

    col_names = [f"h{h:02d}" for h in range(1, 11)]
    metric_csv_map = {
        "rmsle": rmsle_path,
        "mae": tables_dir.parent / "all_models_mae.csv",
        "mape": tables_dir.parent / "all_models_mape.csv",
        "smape": tables_dir.parent / "all_models_smape.csv",
    }
    horizons = list(range(1, 11))

    for metric_name in ("rmsle", "mae", "mape", "smape"):
        lstm_vals = lstm_results[metric_name]
        ml_csv = metric_csv_map[metric_name]
        if ml_csv.exists():
            df_ml = pd.read_csv(ml_csv, index_col="model")
            ml_vals = df_ml.loc[best_ml, col_names].to_numpy(dtype=float)
        else:
            ml_vals = np.full(10, np.nan)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(horizons, lstm_vals, marker="o", label="LSTM")
        ax.plot(horizons, ml_vals, marker="s", linestyle="--", label=best_ml)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()}: LSTM vs {best_ml}")
        ax.set_xticks(horizons)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"metric_{metric_name}_lstm_vs_ml.pdf", format="pdf"
        )
        plt.close(fig)


# ── merge_lstm_metrics ────────────────────────────────────────────────────────

def merge_lstm_metrics(
    lstm_results: dict,
    results_dir: Path = Path("results"),
) -> None:
    """Append LSTM row to all_models_{metric}.csv; replace existing lstm row if present."""
    results_dir = Path(results_dir)
    col_names = [f"h{h:02d}" for h in range(1, 11)]
    metric_map = {
        "rmsle": lstm_results["rmsle"],
        "mae":   lstm_results["mae"],
        "mape":  lstm_results["mape"],
        "smape": lstm_results["smape"],
    }
    for metric_name, vals in metric_map.items():
        csv_path = results_dir / f"all_models_{metric_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["model"] + col_names)
        df = df[df["model"] != "lstm"].copy()
        new_row = {"model": "lstm", **{col: float(v) for col, v in zip(col_names, vals)}}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"[merge_lstm_metrics] Updated {csv_path}")


# ── plot_all_models_comparison ────────────────────────────────────────────────

def plot_all_models_comparison(
    results_dir: Path = Path("results"),
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """One PDF per metric comparing every model across 10 horizons."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    col_names = [f"h{h:02d}" for h in range(1, 11)]
    horizons = list(range(1, 11))
    ml_models = {"xgboost", "random_forest", "lightgbm"}

    for metric_name in ("rmsle", "mae", "mape", "smape"):
        csv_path = results_dir / f"all_models_{metric_name}.csv"
        if not csv_path.exists():
            print(f"[plot_all_models_comparison] {csv_path} not found; skipping")
            continue
        df = pd.read_csv(csv_path, index_col="model")

        fig, ax = plt.subplots(figsize=(10, 5))
        for model in df.index:
            vals = df.loc[model, col_names].to_numpy(dtype=float)
            if model == "lstm":
                ax.plot(horizons, vals, marker="o", linewidth=2.5, label="lstm", zorder=5)
            elif model in ml_models:
                ax.plot(horizons, vals, marker="s", linestyle="--", linewidth=1.5, label=model)
            else:
                ax.plot(horizons, vals, linestyle=":", linewidth=1.0, alpha=0.6, label=model)

        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} — All Models")
        ax.set_xticks(horizons)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = figures_dir / f"all_models_{metric_name}.pdf"
        fig.savefig(out, format="pdf")
        plt.close(fig)
        print(f"[plot_all_models_comparison] Saved {out}")


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from lstm_model import CONFIG_A, LSTMArchConfig

    log_path = TABLES_DIR / "lstm_arch_experiments.csv"
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        best_row = df_log.loc[df_log["val_rmsle"].idxmin()]
        best_config = LSTMArchConfig.from_json(best_row["config_json"])
        best_name = str(best_row["config_name"])
    else:
        print("Warning: no experiment log found; defaulting to CONFIG_A")
        best_config = CONFIG_A
        best_name = "config_a"

    ckpt_path = Path("results/models/lstm/lstm_final.pt")
    results = evaluate_lstm(best_config, ckpt_path)
    print(f"Eval RMSLE  h=1..10: {[f'{v:.4f}' for v in results['rmsle']]}")
    print(f"Eval MAE    h=1..10: {[f'{v:.4f}' for v in results['mae']]}")

    merge_lstm_metrics(results)
    plot_all_models_comparison()
    plot_lstm_vs_ml(results)
    print("Figures saved to results/figures/")
