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
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 256
MAX_EPOCHS: int = 150
MAX_EPOCHS_CPU: int = 50
PATIENCE: int = 15
N_TEMPORAL: int = 12
N_STATIC: int = 6

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


CONFIG_A: LSTMArchConfig = LSTMArchConfig(
    lstm_layers=[
        {"hidden_size": 128, "dropout": 0.2},
        {"hidden_size": 64, "dropout": 0.1},
    ],
    head_layers=[
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "out_features": 10},
    ],
    bidirectional=False,
)

CONFIG_B: LSTMArchConfig = LSTMArchConfig(
    lstm_layers=[
        {"hidden_size": 128, "dropout": 0.2},
        {"hidden_size": 64, "dropout": 0.15},
        {"hidden_size": 32, "dropout": 0.0},
    ],
    head_layers=[
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "out_features": 10},
    ],
    bidirectional=False,
)


# ── DemandRNN ─────────────────────────────────────────────────────────────────

class DemandRNN(nn.Module):
    """Config-driven LSTM with static-feature h_0 initialisation."""

    def __init__(
        self,
        n_temporal: int,
        n_static: int,
        config: LSTMArchConfig,
    ) -> None:
        super().__init__()
        self.config = config
        dir_mult = 2 if config.bidirectional else 1
        H0 = config.lstm_layers[0]["hidden_size"]

        self.static_enc = nn.Sequential(
            nn.Linear(n_static, H0),
            nn.Tanh(),
        )

        self.lstm_cells: nn.ModuleList = nn.ModuleList()
        self.lstm_drops: nn.ModuleList = nn.ModuleList()
        input_size = n_temporal
        for layer_cfg in config.lstm_layers:
            H = layer_cfg["hidden_size"]
            self.lstm_cells.append(
                nn.LSTM(
                    input_size, H,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=config.bidirectional,
                )
            )
            self.lstm_drops.append(nn.Dropout(p=float(layer_cfg["dropout"])))
            input_size = H * dir_mult

        H_last = config.lstm_layers[-1]["hidden_size"] * dir_mult
        current_size = H_last
        head_modules: list[nn.Module] = []
        for spec in config.head_layers:
            t = spec["type"]
            if t == "relu":
                head_modules.append(nn.ReLU())
            elif t == "dropout":
                head_modules.append(nn.Dropout(p=float(spec["p"])))
            elif t == "batchnorm":
                head_modules.append(nn.BatchNorm1d(current_size))
            elif t == "linear":
                out_features = spec["out_features"]
                head_modules.append(nn.Linear(current_size, out_features))
                current_size = out_features
            else:
                raise ValueError(f"Unknown head layer type: {t!r}")
        self.head = nn.Sequential(*head_modules)

    def encode_static(self, x_static: torch.Tensor) -> torch.Tensor:
        """Encode static features to h_0 of shape (1 or 2, B, H0)."""
        h = self.static_enc(x_static).unsqueeze(0)  # (1, B, H0)
        if self.config.bidirectional:
            h = h.expand(2, -1, -1).contiguous()    # (2, B, H0), both equal
        return h

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: (B, 26, 12) × (B, 6) → (B, 10)."""
        B = x_temporal.size(0)
        device = x_temporal.device
        dir_mult = 2 if self.config.bidirectional else 1

        out = x_temporal
        for i, (lstm, drop) in enumerate(zip(self.lstm_cells, self.lstm_drops)):
            if i == 0:
                h0 = self.encode_static(x_static)
            else:
                H = self.config.lstm_layers[i]["hidden_size"]
                h0 = torch.zeros(dir_mult, B, H, device=device)
            c0 = torch.zeros_like(h0)
            out, _ = lstm(out, (h0, c0))
            out = drop(out)

        return self.head(out[:, -1, :])


# ── RMSLELoss ─────────────────────────────────────────────────────────────────

class RMSLELoss(nn.Module):
    """RMSLE loss: sqrt(mean((log1p(clamp(pred,0)) − log1p(target))²))."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute scalar RMSLE; clips negative predictions to 0."""
        log_diff = torch.log1p(torch.clamp(pred, min=0.0)) - torch.log1p(target)
        return torch.sqrt(torch.mean(log_diff ** 2))


# ── DemandDataset ─────────────────────────────────────────────────────────────

class DemandDataset(Dataset):
    """Loads {split}_X_temporal.npy, {split}_X_static.npy, {split}_y.npy only."""

    def __init__(self, split: str, seq_dir: Path = SEQ_DIR) -> None:
        """Load arrays; drop NaN rows at init time; log drop count."""
        seq_dir = Path(seq_dir)
        X_temp = np.load(seq_dir / f"{split}_X_temporal.npy")
        X_stat = np.load(seq_dir / f"{split}_X_static.npy")
        y = np.load(seq_dir / f"{split}_y.npy")

        nan_mask = (
            np.isnan(X_temp).any(axis=(1, 2)) | np.isnan(X_stat).any(axis=1)
        )
        n_dropped = int(nan_mask.sum())
        if n_dropped:
            print(f"[DemandDataset] {split}: dropped {n_dropped} NaN rows")

        keep = ~nan_mask
        self.X_temp = torch.tensor(X_temp[keep], dtype=torch.float32)
        self.X_stat = torch.tensor(X_stat[keep], dtype=torch.float32)
        self.y = torch.tensor(y[keep], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_temp[idx], self.X_stat[idx], self.y[idx]


# ── build_model ───────────────────────────────────────────────────────────────

def build_model(config: LSTMArchConfig) -> DemandRNN:
    """Instantiate DemandRNN with N_TEMPORAL=12, N_STATIC=6."""
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(config).to(device)
    criterion = RMSLELoss().to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_rmsle = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses: list[float] = []
    val_rmsles: list[float] = []
    last_epoch = 0

    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        epoch_loss = 0.0
        for xt, xs, yb in train_loader:
            xt, xs, yb = xt.to(device), xs.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xt, xs), yb)
            loss.backward()
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
        val_rmsle = float(np.sqrt(np.mean(
            (np.log1p(np.clip(y_pred, 0, None)) - np.log1p(y_true)) ** 2
        )))
        val_rmsles.append(val_rmsle)

        if val_rmsle < best_val_rmsle:
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
            if patience_counter >= patience:
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
    from evaluate import append_experiment_log  # avoid import cycle at module load

    models_dir = Path(models_dir)
    log_path = Path("results/tables/lstm_arch_experiments.csv")

    configs = [("config_a", CONFIG_A), ("config_b", CONFIG_B)]
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

    return best_name


# ── __main__ ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best = run_ablation()
    print(f"\nAblation complete. Best config: {best}")
    print("Checkpoint: results/models/lstm/lstm_final.pt")
