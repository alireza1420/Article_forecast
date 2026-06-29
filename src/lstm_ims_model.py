"""IMS (Iterative Multi-Step) LSTM: trains 1-step-ahead, rolls 10 steps at inference.

Unlike the DMS model in lstm_model.py (which outputs all 10 horizons in one pass),
this model is trained to predict only h=1.  At evaluation time we iterate: each
predicted demand is fed back into the num_orders slot of the rolling input window,
then the window slides one step forward for the next prediction.
"""
from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lstm_model import (
    DemandRNN,
    LSTMArchConfig,
    set_seed,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    PATIENCE,
    SEQ_DIR,
    SEED,
    N_TEMPORAL,
    N_STATIC,
    PAPER_LOOKBACK,
)
from features import SEQUENCE_TEMPORAL_COLS

# ── Constants ─────────────────────────────────────────────────────────────────

IMS_MODELS_DIR: Path = Path("results/models/lstm_ims")

# num_orders is always index 0 in SEQUENCE_TEMPORAL_COLS
_NUM_ORDERS_IDX: int = SEQUENCE_TEMPORAL_COLS.index("num_orders")


# ── IMS-compatible config (relaxes out_features == 10 constraint) ─────────────

@dataclass
class LSTMArchConfigIMS(LSTMArchConfig):
    """LSTMArchConfig variant for IMS where the head outputs 1 value (not 10)."""

    def validate(self) -> None:
        if not self.lstm_layers:
            raise ValueError("lstm_layers must be non-empty")
        for i, layer in enumerate(self.lstm_layers):
            d = layer.get("dropout", 0.0)
            if not (0.0 <= float(d) < 1.0):
                raise ValueError(
                    f"lstm_layers[{i}]['dropout'] must be in [0, 1): got {d}"
                )
        if not self.head_layers:
            raise ValueError("head_layers must be non-empty")
        final = self.head_layers[-1]
        if final.get("type") != "linear":
            raise ValueError("head_layers[-1] must be a linear layer")


# Same 3-layer tapering as CONFIG_C in lstm_model.py, but head outputs 1 value.
CONFIG_IMS: LSTMArchConfigIMS = LSTMArchConfigIMS(
    lstm_layers=[
        {"hidden_size": 64,  "dropout": 0.2, "activation": "relu"},
        {"hidden_size": 32,  "dropout": 0.2, "activation": "relu"},
        {"hidden_size": 16,  "dropout": 0.0},
    ],
    head_layers=[
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "out_features": 1},
    ],
    bidirectional=False,
)


# ── Dataset ────────────────────────────────────────────────────────────────────

class DemandDatasetIMS(Dataset):
    """Temporal + static features; stores h=1 target for training and full y for eval."""

    def __init__(self, split: str, seq_dir: Path = SEQ_DIR) -> None:
        seq_dir = Path(seq_dir)
        X_full = np.load(seq_dir / f"{split}_X_temporal.npy")   # (N, 26, 32)
        X_stat = np.load(seq_dir / f"{split}_X_static.npy")     # (N, 21)
        y_full = np.load(seq_dir / f"{split}_y.npy")             # (N, 10)  log1p

        T_seq = X_full.shape[1]
        positions = np.arange(T_seq - PAPER_LOOKBACK, T_seq)
        X_new = X_full[:, positions, :].astype(np.float32)       # (N, 10, 32)

        nan_mask = (
            np.isnan(X_new).any(axis=(1, 2)) | np.isnan(X_stat).any(axis=1)
        )
        keep = ~nan_mask
        n_dropped = int(nan_mask.sum())
        if n_dropped:
            print(f"[DemandDatasetIMS:{split}] dropped {n_dropped} NaN rows")

        self.X_temp  = torch.tensor(X_new[keep],               dtype=torch.float32)
        self.X_stat  = torch.tensor(X_stat[keep].astype(np.float32), dtype=torch.float32)
        self.y_h1    = torch.tensor(y_full[keep, 0],           dtype=torch.float32)  # training
        self.y_full  = torch.tensor(y_full[keep].astype(np.float32), dtype=torch.float32)  # eval

    def __len__(self) -> int:
        return len(self.y_h1)

    def __getitem__(self, idx: int):
        return self.X_temp[idx], self.X_stat[idx], self.y_h1[idx]


# ── Model builder ─────────────────────────────────────────────────────────────

def build_ims_model(config: LSTMArchConfigIMS = CONFIG_IMS) -> DemandRNN:
    return DemandRNN(N_TEMPORAL, N_STATIC, config)


# ── IMS rollout ───────────────────────────────────────────────────────────────

def ims_rollout(
    model: DemandRNN,
    X_temporal: torch.Tensor,   # (N, 10, 32)
    X_static: torch.Tensor,     # (N, 21)
    n_steps: int = 10,
    scaler_mean: float = 0.0,
    scaler_std: float = 1.0,
) -> torch.Tensor:
    """Roll n_steps forward, feeding each predicted log1p(orders) back as num_orders.

    The predicted value is standardised before insertion so it matches the
    z-score scale of the training features.

    Returns (N, n_steps) in log1p space.
    """
    model.eval()
    device = next(model.parameters()).device
    window = X_temporal.to(device)   # (N, 10, 32)
    xs     = X_static.to(device)
    preds  = []

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(window, xs)          # (N, 1)
            preds.append(pred)

            # Convert log1p prediction to standardised scale for the input slot
            pred_std = (pred - scaler_mean) / scaler_std   # (N, 1)

            # Build the next timestep: copy last step, replace num_orders
            next_step = window[:, -1:, :].clone()           # (N, 1, 32)
            next_step[:, :, _NUM_ORDERS_IDX] = pred_std

            # Slide window forward by one
            window = torch.cat([window[:, 1:, :], next_step], dim=1)

    return torch.cat(preds, dim=1)   # (N, n_steps)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_ims(
    config: LSTMArchConfigIMS = CONFIG_IMS,
    seq_dir: Path = SEQ_DIR,
    models_dir: Path = IMS_MODELS_DIR,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict:
    """Train 1-step-ahead IMS LSTM; save best checkpoint; return training state."""
    if not torch.cuda.is_available():
        raise RuntimeError("[train_ims] CUDA required but not available.")
    device = torch.device("cuda")

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "best_ims.pt"

    set_seed(SEED)
    train_ds = DemandDatasetIMS("train", seq_dir=seq_dir)
    val_ds   = DemandDatasetIMS("val",   seq_dir=seq_dir)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0
    )

    model     = build_ims_model(config).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    best_val    = float("inf")
    best_epoch  = 0
    patience_ctr = 0
    train_losses: list[float] = []
    val_rmsles:  list[float]  = []

    LOG_EVERY = 10
    print(
        f"[train_ims] device={device} | epochs={max_epochs} | patience={patience} | "
        f"train={len(train_ds):,} | val={len(val_ds):,}"
    )

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xt, xs, yb in train_loader:
            xt, xs, yb = xt.to(device), xs.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xt, xs).squeeze(-1)   # (N,)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(yb)
        train_losses.append(epoch_loss / len(train_ds))

        # Validation: 1-step RMSLE (MSE on log1p targets == RMSLE on raw orders)
        model.eval()
        p_list, t_list = [], []
        with torch.no_grad():
            for xt, xs, yb in val_loader:
                p_list.append(model(xt.to(device), xs.to(device)).squeeze(-1).cpu().numpy())
                t_list.append(yb.numpy())
        y_pred = np.concatenate(p_list)
        y_true = np.concatenate(t_list)
        val_rmsle = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        val_rmsles.append(val_rmsle)
        scheduler.step(val_rmsle)

        is_best = val_rmsle < best_val
        if is_best:
            best_val     = val_rmsle
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_rmsle":  val_rmsle,
                    "config":     dataclasses.asdict(config),
                },
                ckpt_path,
            )
        else:
            patience_ctr += 1

        lr_now = optimiser.param_groups[0]["lr"]
        if (epoch + 1) % LOG_EVERY == 0 or is_best or patience_ctr >= patience:
            tag = " *best*" if is_best else f" patience={patience_ctr}/{patience}"
            print(
                f"  epoch {epoch+1:>4}/{max_epochs} | "
                f"train_loss={train_losses[-1]:.4f} | val_rmsle={val_rmsle:.4f} | "
                f"lr={lr_now:.2e}{tag}"
            )
        if patience_ctr >= patience:
            print(f"[train_ims] early stop at epoch {epoch + 1}")
            break

    print(f"[train_ims] best val_rmsle={best_val:.4f} at epoch {best_epoch + 1}")
    return {
        "best_val_rmsle":  best_val,
        "best_epoch":      best_epoch,
        "train_losses":    train_losses,
        "val_rmsles":      val_rmsles,
        "checkpoint_path": str(ckpt_path),
    }


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_ims(
    ckpt_path: Path | None = None,
    seq_dir: Path = SEQ_DIR,
    config: LSTMArchConfigIMS = CONFIG_IMS,
    n_steps: int = 10,
) -> dict:
    """Load IMS checkpoint, run 10-step rollout on eval split, return per-horizon metrics."""
    from evaluate import compute_rmsle, compute_mae, compute_mape, compute_smape

    if ckpt_path is None:
        ckpt_path = IMS_MODELS_DIR / "best_ims.pt"
    ckpt_path = Path(ckpt_path)

    # Load scaler params for num_orders (z-score of log1p(raw_orders))
    sp_path = Path("data/processed/scaler_params.json")
    with open(sp_path) as f:
        sp = json.load(f)
    scaler_mean = float(sp["num_orders"][0])
    scaler_std  = float(sp["num_orders"][1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_ims_model(config).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[evaluate_ims] Loaded checkpoint {ckpt_path} (val_rmsle={ckpt['val_rmsle']:.4f})")

    eval_ds = DemandDatasetIMS("eval", seq_dir=seq_dir)
    X_temp  = eval_ds.X_temp    # (N, 10, 32)
    X_stat  = eval_ds.X_stat    # (N, 21)
    y_true_log = eval_ds.y_full.numpy()   # (N, 10)  log1p

    # Run IMS rollout in batches
    batch_size = 512
    all_preds = []
    for start in range(0, len(X_temp), batch_size):
        xt = X_temp[start:start + batch_size]
        xs = X_stat[start:start + batch_size]
        pred = ims_rollout(model, xt, xs, n_steps, scaler_mean, scaler_std)
        all_preds.append(pred.cpu().numpy())
    y_pred_log = np.concatenate(all_preds, axis=0)   # (N, 10)  log1p

    # De-transform to raw order counts
    y_pred_raw = np.expm1(np.clip(y_pred_log, 0.0, None))
    y_true_raw = np.expm1(y_true_log)

    mape_vals, n_excl = compute_mape(y_true_raw, y_pred_raw)
    results = {
        "rmsle":           compute_rmsle(y_true_raw, y_pred_raw),
        "mae":             compute_mae(y_true_raw, y_pred_raw),
        "mape":            mape_vals,
        "smape":           compute_smape(y_true_raw, y_pred_raw),
        "mape_n_excluded": n_excl,
    }

    mean_pred = y_pred_raw.mean()
    mean_true = y_true_raw.mean()
    print(f"[evaluate_ims] mean_pred={mean_pred:.1f}  mean_true={mean_true:.1f}  "
          f"bias_ratio={mean_pred / mean_true:.3f}")
    print(f"[evaluate_ims] RMSLE  h=1..10: {[f'{v:.4f}' for v in results['rmsle']]}")
    print(f"[evaluate_ims] MAE    h=1..10: {[f'{v:.1f}'   for v in results['mae']]}")
    print(f"[evaluate_ims] SMAPE  h=1..10: {[f'{v:.2f}'   for v in results['smape']]}")

    return results


# ── Save IMS metrics into shared comparison CSVs ──────────────────────────────

def save_ims_metrics(results: dict, results_dir: Path = Path("results")) -> None:
    """Upsert an 'lstm_ims' row into each all_models_*.csv for cross-model comparison."""
    import pandas as pd

    results_dir = Path(results_dir)
    col_names   = [f"h{h:02d}" for h in range(1, 11)]

    metric_map = {
        "rmsle": results["rmsle"],
        "mae":   results["mae"],
        "mape":  results["mape"],
        "smape": results["smape"],
    }
    for metric_name, vals in metric_map.items():
        csv_path = results_dir / f"all_models_{metric_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["model"] + col_names)
        df = df[df["model"] != "lstm_ims"].copy()
        new_row = {"model": "lstm_ims", **{col: float(v) for col, v in zip(col_names, vals)}}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"[save_ims_metrics] Updated {csv_path}")


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(Path(__file__).resolve().parents[1])   # repo root

    print("=" * 60)
    print("  IMS LSTM — Training")
    print("=" * 60)
    state = train_ims()

    print()
    print("=" * 60)
    print("  IMS LSTM — Evaluation (10-step rollout)")
    print("=" * 60)
    results = evaluate_ims()

    save_ims_metrics(results)

    from evaluate import plot_all_models_comparison
    plot_all_models_comparison()
    print("\nDone. Figures in results/figures/all_models_*.pdf")
