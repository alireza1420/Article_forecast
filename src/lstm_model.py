"""Configurable PyTorch LSTM for multi-step demand forecasting."""
from __future__ import annotations

import dataclasses
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

SEED: int = 42
LR: float = 5e-4
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 256
MAX_EPOCHS: int = 300
MAX_EPOCHS_CPU: int = 50
PATIENCE: int = 20
# 32 features: all SEQUENCE_TEMPORAL_COLS (num_orders, prices, promotions, EWMs, rolling stats, calendar, ranks)
N_TEMPORAL: int = 32
N_STATIC: int = 21
PAPER_LOOKBACK: int = 10  # 10-timestep input window

# Architecture knobs (selected empirically; see DemandRNN docstring)
SKIP_K: int = 3           # last K weeks' full feature snapshots fed directly to the head
STATIC_FUSE_DIM: int = 64  # width of the static-feature projection concatenated at the head
HEAD_DROPOUT: float = 0.3

SEQ_DIR: Path = Path("data/processed/sequences")
MODELS_DIR: Path = Path("results/models/lstm")


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """Set Python, numpy, and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── LSTMArchConfig ────────────────────────────────────────────────────────────

@dataclass
class LSTMArchConfig:
    """Architecture descriptor for DemandRNN, serialisable to/from JSON."""

    lstm_layers: list[dict]
    head_layers: list[dict]
    bidirectional: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if any invariant is violated."""
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
        if final.get("type") != "linear" or final.get("out_features") != 10:
            raise ValueError(
                "head_layers[-1] must be {'type': 'linear', 'out_features': 10}"
            )

    def to_json(self) -> str:
        """Serialise to compact JSON string via dataclasses.asdict()."""
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "LSTMArchConfig":
        """Deserialise from JSON string; re-validates invariants."""
        return cls(**json.loads(s))


# # Paper-matching architectures (Section IV.B: 32→16 hidden units)
# CONFIG_A: LSTMArchConfig = LSTMArchConfig(
#     lstm_layers=[
#         {"hidden_size": 32, "dropout": 0.25},
#         {"hidden_size": 16, "dropout": 0.0},
#     ],
#     head_layers=[
#         {"type": "relu"},
#         {"type": "dropout", "p": 0.1},
#         {"type": "linear", "out_features": 10},
#     ],
#     bidirectional=False,
# )

# CONFIG_A_BI: LSTMArchConfig = LSTMArchConfig(
#     lstm_layers=[
#         {"hidden_size": 32, "dropout": 0.25},
#         {"hidden_size": 16, "dropout": 0.0},
#     ],
#     head_layers=[
#         {"type": "relu"},
#         {"type": "dropout", "p": 0.1},
#         {"type": "linear", "out_features": 10},
#     ],
#     bidirectional=True,
# )

# # Larger ablation variants
# CONFIG_B: LSTMArchConfig = LSTMArchConfig(
#     lstm_layers=[
#         {"hidden_size": 64, "dropout": 0.2},
#         {"hidden_size": 32, "dropout": 0.1},
#     ],
#     head_layers=[
#         {"type": "relu"},
#         {"type": "dropout", "p": 0.1},
#         {"type": "linear", "out_features": 10},
#     ],
#     bidirectional=False,
# )

# CONFIG_B_BI: LSTMArchConfig = LSTMArchConfig(
#     lstm_layers=[
#         {"hidden_size": 64, "dropout": 0.2},
#         {"hidden_size": 32, "dropout": 0.1},
#     ],
#     head_layers=[
#         {"type": "relu"},
#         {"type": "dropout", "p": 0.1},
#         {"type": "linear", "out_features": 10},
#     ],
#     bidirectional=True,
# )

CONFIG_C: LSTMArchConfig = LSTMArchConfig(
    # Uniform stacked LSTM: hidden=128, num_layers=3, recurrent dropout=0.3.
    # DemandRNN reads hidden/num_layers/dropout from these entries and builds its own
    # fusion + skip head (head_layers below only declares the output width=10 so the
    # ablation/validation machinery keeps working).
    lstm_layers=[
        {"hidden_size": 128, "dropout": 0.3},
        {"hidden_size": 128, "dropout": 0.3},
        {"hidden_size": 128, "dropout": 0.3},
    ],
    head_layers=[
        {"type": "linear", "out_features": 10},
    ],
    bidirectional=False,
)


# ── DemandRNN ─────────────────────────────────────────────────────────────────

class DemandRNN(nn.Module):
    """Stacked LSTM + static-level fusion + recent-week skip connection.

    Architecture selected empirically to close the gap with gradient-boosted trees:
      - a uniform stacked ``nn.LSTM`` encodes the (correctly windowed) demand sequence;
      - the static vector (per-series level: center/meal mean orders, price & count
        aggregates) is projected and concatenated to the final hidden state, so the
        head reads series level directly instead of from a washed-out h_0 seed;
      - the last ``SKIP_K`` weeks' full feature snapshots are concatenated to the head
        as a skip connection, giving short-horizon predictions direct access to the
        most recent observed demand / promotions / prices — the signal trees exploit
        via explicit lags. This is what makes h=1 a true 1-step forecast usable.
    Output is ``softplus`` so log1p predictions stay strictly non-negative.

    Hyper-parameters are read from ``config`` for compatibility with the ablation
    machinery: ``hidden`` and ``num_layers`` come from ``lstm_layers`` (first layer's
    ``hidden_size`` and the layer count), the recurrent ``dropout`` from the first
    layer, and the output width from ``head_layers[-1]['out_features']``.
    """

    def __init__(
        self,
        n_temporal: int,
        n_static: int,
        config: LSTMArchConfig,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.lstm_layers[0]["hidden_size"])
        num_layers = len(config.lstm_layers)
        dropout = float(config.lstm_layers[0].get("dropout", 0.0))
        out_features = int(config.head_layers[-1]["out_features"])

        self.lstm = nn.LSTM(
            n_temporal, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_fuse = (
            nn.Sequential(
                nn.Linear(n_static, STATIC_FUSE_DIM), nn.ReLU(), nn.Dropout(HEAD_DROPOUT)
            )
            if n_static > 0 else None
        )
        fuse_dim = STATIC_FUSE_DIM if n_static > 0 else 0
        head_in = hidden + fuse_dim + n_temporal * SKIP_K
        self.head = nn.Sequential(
            nn.Linear(head_in, 128), nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
            nn.Linear(64, out_features),
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """(B, T, N_TEMPORAL), (B, N_STATIC) → (B, out) non-negative log1p predictions."""
        out, _ = self.lstm(x_temporal)
        feats = [out[:, -1, :]]
        if self.static_fuse is not None:
            feats.append(self.static_fuse(x_static))
        feats.append(x_temporal[:, -SKIP_K:, :].reshape(x_temporal.size(0), -1))
        return torch.nn.functional.softplus(self.head(torch.cat(feats, dim=-1)))


# ── RMSLELoss ─────────────────────────────────────────────────────────────────

class RMSLELoss(nn.Module):
    """RMSLE loss: sqrt(mean((log1p(clamp(pred,0)) − log1p(target))²))."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute scalar RMSLE; clips negative predictions to 0."""
        log_diff = torch.log1p(torch.clamp(pred, min=0.0)) - torch.log1p(target)
        return torch.sqrt(torch.mean(log_diff ** 2))


# ── DemandDataset ─────────────────────────────────────────────────────────────

class DemandDataset(Dataset):
    """Uses all 32 SEQUENCE_TEMPORAL_COLS over the last PAPER_LOOKBACK timesteps."""

    def __init__(self, split: str, seq_dir: Path = SEQ_DIR) -> None:
        seq_dir = Path(seq_dir)
        X_full = np.load(seq_dir / f"{split}_X_temporal.npy")  # (N, 26, 32)
        X_stat = np.load(seq_dir / f"{split}_X_static.npy")    # (N, 21)
        y = np.load(seq_dir / f"{split}_y.npy")

        T_seq = X_full.shape[1]
        positions = np.arange(T_seq - PAPER_LOOKBACK, T_seq)

        # (N, 10, 32) — all temporal features over the lookback window
        X_new = X_full[:, positions, :].astype(np.float32)

        nan_mask = (
            np.isnan(X_new).any(axis=(1, 2)) | np.isnan(X_stat).any(axis=1)
        )
        n_dropped = int(nan_mask.sum())
        if n_dropped:
            print(f"[DemandDataset] {split}: dropped {n_dropped} NaN rows")

        keep = ~nan_mask
        self.X_temp = torch.tensor(X_new[keep], dtype=torch.float32)
        self.X_stat = torch.tensor(X_stat[keep].astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y[keep], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_temp[idx], self.X_stat[idx], self.y[idx]


# ── build_model ───────────────────────────────────────────────────────────────

def build_model(config: LSTMArchConfig) -> DemandRNN:
    """Instantiate DemandRNN with N_TEMPORAL=32, N_STATIC=21."""
    return DemandRNN(N_TEMPORAL, N_STATIC, config)


# ── train ──────────────────────────────────────────────────────────────────────

def train(
    config: LSTMArchConfig,
    config_name: str,
    seq_dir: Path = SEQ_DIR,
    models_dir: Path = MODELS_DIR,
    max_epochs: int | None = None,
    patience: int = PATIENCE,
) -> dict:
    """Full training loop: data → model → Adam → early stop → checkpoint."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[train] CUDA is required but not available. "
            "Install a CUDA-enabled PyTorch build: https://pytorch.org/get-started/locally/"
        )
    device = torch.device("cuda")
    if max_epochs is None:
        max_epochs = MAX_EPOCHS

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"best_{config_name}.pt"

    set_seed(SEED)
    train_ds = DemandDataset("train", seq_dir=seq_dir)
    val_ds = DemandDataset("val", seq_dir=seq_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = build_model(config).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_rmsle = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses: list[float] = []
    val_rmsles: list[float] = []
    last_epoch = 0

    LOG_EVERY = 10  # print a summary line every N epochs

    print(
        f"[train] {config_name} | device={device} | "
        f"epochs={max_epochs} | patience={patience} | "
        f"train_samples={len(train_ds)} | val_samples={len(val_ds)}"
    )

    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        epoch_loss = 0.0
        for xt, xs, yb in train_loader:
            xt, xs, yb = xt.to(device), xs.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xt, xs), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += float(loss.item()) * len(yb)
        train_losses.append(epoch_loss / max(len(train_ds), 1))

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xt, xs, yb in val_loader:
                preds.append(model(xt.to(device), xs.to(device)).cpu().numpy())
                targets.append(yb.numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(targets)
        val_rmsle = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        val_rmsles.append(val_rmsle)
        scheduler.step(val_rmsle)

        current_lr = optimiser.param_groups[0]["lr"]
        is_best = val_rmsle < best_val_rmsle

        if is_best:
            best_val_rmsle = val_rmsle
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config_json": config.to_json(),
                    "val_rmsle": val_rmsle,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % LOG_EVERY == 0 or is_best or patience_counter >= patience:
            tag = " *best*" if is_best else f" patience={patience_counter}/{patience}"
            print(
                f"  epoch {epoch + 1:>4}/{max_epochs} | "
                f"train_loss={train_losses[-1]:.4f} | "
                f"val_rmsle={val_rmsle:.4f} | "
                f"lr={current_lr:.2e}"
                f"{tag}"
            )

        if patience_counter >= patience:
            print(f"[train] {config_name} | early stop at epoch {epoch + 1}")
            break

    return {
        "best_val_rmsle": best_val_rmsle,
        "best_epoch": best_epoch,
        "early_stop_epoch": last_epoch if patience_counter >= patience else None,
        "train_losses": train_losses,
        "val_rmsles": val_rmsles,
        "checkpoint_path": str(ckpt_path),
    }


# ── run_ablation ───────────────────────────────────────────────────────────────

def run_ablation(
    seq_dir: Path = SEQ_DIR,
    models_dir: Path = MODELS_DIR,
) -> str:
    """Train Config A and Config B; return name of the best config."""
    from evaluate import append_experiment_log, plot_training_curves, plot_lstm_config_predictions  # avoid import cycle at module load

    models_dir = Path(models_dir)
    log_path = Path("results/tables/lstm_arch_experiments.csv")

    configs = [
        # ("config_a",    CONFIG_A),
        # ("config_b",    CONFIG_B),
        ("config_c",    CONFIG_C),
        # ("config_a_bi", CONFIG_A_BI),
        # ("config_b_bi", CONFIG_B_BI),
    ]
    states: dict[str, dict] = {}

    for name, cfg in configs:
        print(f"\n[run_ablation] Training {name} ...")
        state = train(cfg, name, seq_dir=seq_dir, models_dir=models_dir)
        states[name] = state
        append_experiment_log(cfg, name, state["best_val_rmsle"], log_path)
        print(
            f"[run_ablation] {name}: best_val_rmsle="
            f"{state['best_val_rmsle']:.4f} at epoch {state['best_epoch']}"
        )

    best_name = min(states, key=lambda n: states[n]["best_val_rmsle"])
    best_rmsle = states[best_name]["best_val_rmsle"]
    print(f"\n[run_ablation] Best config: {best_name} (val RMSLE={best_rmsle:.4f})")

    src_ckpt = models_dir / f"best_{best_name}.pt"
    dst_ckpt = models_dir / "lstm_final.pt"
    shutil.copy2(src_ckpt, dst_ckpt)
    print(f"[run_ablation] Final checkpoint saved: {dst_ckpt}")

    best_state = states[best_name]
    plot_training_curves(
        best_state["train_losses"],
        best_state["val_rmsles"],
        best_state["early_stop_epoch"],
        Path("results/figures/lstm_training_curves.pdf"),
    )
    print("[run_ablation] Training curves saved: results/figures/lstm_training_curves.pdf")

    plot_lstm_config_predictions(configs, models_dir=models_dir)

    return best_name


# ── __main__ ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best = run_ablation()
    print(f"\nAblation complete. Best config: {best}")
    print("Checkpoint: results/models/lstm/lstm_final.pt")
