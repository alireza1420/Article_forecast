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
from torch.utils.data import DataLoader, Dataset, Subset

# canonical stored feature order; dual import keeps both `lstm_model` (scripts, cwd=src
# on sys.path) and `src.lstm_model` (pytest from repo root) working
try:
    from features import SEQUENCE_TEMPORAL_COLS, SEQUENCE_STATIC_COLS
except ModuleNotFoundError:
    from src.features import SEQUENCE_TEMPORAL_COLS, SEQUENCE_STATIC_COLS

# ── Constants ─────────────────────────────────────────────────────────────────

SEED: int = 42
LR: float = 1e-4
# Raised 1e-4 → 1e-3: val RMSLE bottomed at epoch ~7 while train loss kept falling
# (classic overfit divergence), and LR halving didn't help — so penalise capacity, not step size.
WEIGHT_DECAY: float = 1e-3
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 300
MAX_EPOCHS_CPU: int = 50
PATIENCE: int = 20            # early-stopping patience (epochs without val improvement)
# LR scheduler: drop LR well before early-stopping fires so later epochs stay productive.
# Evidence from runs: val RMSLE improved right after the first LR halving, so we want
# several halvings inside the early-stop window rather than just one.
LR_SCHEDULER_PATIENCE: int = 5
LR_SCHEDULER_FACTOR: float = 0.5
MIN_LR: float = 1e-6
# Full widths of the stored .npy arrays (all SEQUENCE_TEMPORAL_COLS / SEQUENCE_STATIC_COLS)
N_TEMPORAL: int = 32
N_STATIC: int = 21
PAPER_LOOKBACK: int = 10  # 10-timestep input window

# Optional feature selection: lists of names from SEQUENCE_TEMPORAL_COLS /
# SEQUENCE_STATIC_COLS to slice out of the stored 32/21-feature arrays (no
# build_dl_sequences re-run needed); None = all features, [] = none (static only).
# These module defaults can be overridden per call — DemandDataset, build_model, train,
# and run_ablation all accept temporal_cols/static_cols. train() records the selection
# in its checkpoint so evaluation rebuilds the model and dataset with matching widths.
# All 32 temporal features, spelled out (canonical order) so lines can be pruned freely;
# equivalent to None.
LSTM_TEMPORAL_COLS: list[str] | None = [
    "num_orders",
    "checkout_price",
    "base_price",
    "discount_rate",
    "log_checkout_price",
    "emailer_for_promotion",
    "homepage_featured",
    "email_x_discount",
    "homepage_x_discount",
    "week_number",
    "week_of_year_sin",
    "week_of_year_cos",
    "center_week_count",
    "center_cat_week_count",
    "city_meal_week_count",
    "meal_week_count",
    "region_meal_week_count",
    "type_meal_week_count",
    "center_week_price_rank",
    "meal_week_price_rank",
    "ewm_alpha05",
    "ewm_span10",
    "rolling_mean_4w", "rolling_std_4w", "rolling_min_4w", "rolling_max_4w",
    "rolling_mean_8w", "rolling_std_8w", "rolling_min_8w", "rolling_max_8w",
]
LSTM_STATIC_COLS: list[str] | None = []   # no static features fused at the head


def _feature_indices(
    cols: list[str] | None, canonical: list[str], kind: str, allow_empty: bool = False
) -> list[int]:
    """Map feature names to their positions in the stored arrays; None → all columns.

    ``allow_empty`` permits ``[]`` (no features) — valid for static (DemandRNN skips the
    static-fusion head when n_static == 0) but not for temporal (the LSTM needs input).
    """
    if cols is None:
        return list(range(len(canonical)))
    if not cols and not allow_empty:
        raise ValueError(f"{kind}_cols must be None (all features) or non-empty")
    unknown = [c for c in cols if c not in canonical]
    if unknown:
        raise ValueError(
            f"unknown {kind} feature(s) {unknown}; valid names: {canonical}"
        )
    return [canonical.index(c) for c in cols]

# Architecture knobs (selected empirically; see DemandRNN docstring)
SKIP_K: int = 3           # last K weeks' full feature snapshots fed directly to the head
STATIC_FUSE_DIM: int = 64  # width of the static-feature projection concatenated at the head
# Head regularisation, tightened against overfitting (best val at epoch ~7, then rising):
# dropout 0.3 → 0.4 and head widths 128/64 → 64/32. The head is where most capacity sits
# once the feature set is small (e.g. 3 temporal cols ⇒ LSTM input is tiny but the old
# 128-wide head could still memorise), so shrinking it targets the actual problem.
# NOTE: changing HEAD_HIDDEN breaks state_dict compatibility with old checkpoints —
# retrain rather than loading previous best_*.pt files.
HEAD_DROPOUT: float = 0.4
HEAD_HIDDEN: tuple[int, int] = (64, 32)

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


# Paper-matching architectures (Section IV.B: 32→16 hidden units)
CONFIG_A: LSTMArchConfig = LSTMArchConfig(
    lstm_layers=[
        {"hidden_size": 32, "dropout": 0.5},
        {"hidden_size": 16, "dropout": 0.3},
    ],
    head_layers=[
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "out_features": 10},
    ],
    bidirectional=False,
)

CONFIG_A_BI: LSTMArchConfig = LSTMArchConfig(
    lstm_layers=[
        {"hidden_size": 32, "dropout": 0.5},
        {"hidden_size": 16, "dropout": 0.3},
    ],
    head_layers=[
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "out_features": 10},
    ],
    bidirectional=True,
)

# Larger ablation variants
CONFIG_B: LSTMArchConfig = LSTMArchConfig(
    lstm_layers=[
        {"hidden_size": 64, "dropout": 0.5},
        {"hidden_size": 32, "dropout": 0.3},
    ],
    head_layers=[
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "out_features": 10},
    ],
    bidirectional=False,
)

CONFIG_B_BI: LSTMArchConfig = LSTMArchConfig(
    lstm_layers=[
        {"hidden_size": 64, "dropout": 0.5},
        {"hidden_size": 32, "dropout": 0.3},
    ],
    head_layers=[
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "out_features": 10},
    ],
    bidirectional=True,
)

CONFIG_C: LSTMArchConfig = LSTMArchConfig(
    # Uniform stacked LSTM: hidden=128, num_layers=2, recurrent dropout=0.3.
    # DemandRNN reads hidden/num_layers/dropout from these entries and builds its own
    # fusion + skip head (head_layers below only declares the output width=10 so the
    # ablation/validation machinery keeps working).
    lstm_layers=[
        {"hidden_size": 128, "dropout": 0.5},
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
        h1, h2 = HEAD_HIDDEN
        self.head = nn.Sequential(
            nn.Linear(head_in, h1), nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
            nn.Linear(h2, out_features),
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
    """Serves the selected temporal/static features over the last PAPER_LOOKBACK timesteps.

    ``temporal_cols`` / ``static_cols`` are lists of names from SEQUENCE_TEMPORAL_COLS /
    SEQUENCE_STATIC_COLS; None falls back to the module defaults (all features).
    """

    def __init__(
        self,
        split: str,
        seq_dir: Path = SEQ_DIR,
        temporal_cols: list[str] | None = None,
        static_cols: list[str] | None = None,
    ) -> None:
        seq_dir = Path(seq_dir)
        t_idx = _feature_indices(
            temporal_cols if temporal_cols is not None else LSTM_TEMPORAL_COLS,
            SEQUENCE_TEMPORAL_COLS, "temporal",
        )
        s_idx = _feature_indices(
            static_cols if static_cols is not None else LSTM_STATIC_COLS,
            SEQUENCE_STATIC_COLS, "static", allow_empty=True,
        )
        X_full = np.load(seq_dir / f"{split}_X_temporal.npy")   # (N, 26, 32)
        X_stat_full = np.load(seq_dir / f"{split}_X_static.npy")       # (N, 21)
        y = np.load(seq_dir / f"{split}_y.npy")

        # name-based selection only works on arrays stored in canonical column order
        if max(t_idx, default=-1) >= X_full.shape[2] or max(s_idx, default=-1) >= X_stat_full.shape[1]:
            raise ValueError(
                f"stored arrays ({X_full.shape[2]} temporal / {X_stat_full.shape[1]} static cols) "
                f"are narrower than the requested feature indices — rebuild sequences with "
                f"build_dl_sequences (expects {len(SEQUENCE_TEMPORAL_COLS)}/{len(SEQUENCE_STATIC_COLS)})"
            )
        X_stat = X_stat_full[:, s_idx]                                  # (N, n_static)

        T_seq = X_full.shape[1]
        positions = np.arange(T_seq - PAPER_LOOKBACK, T_seq)

        # (N, PAPER_LOOKBACK, n_temporal) — selected temporal features over the window
        X_new = X_full[:, positions, :][:, :, t_idx].astype(np.float32)

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

def build_model(
    config: LSTMArchConfig,
    temporal_cols: list[str] | None = None,
    static_cols: list[str] | None = None,
) -> DemandRNN:
    """Instantiate DemandRNN sized to the selected features (all 32/21 when None)."""
    n_temporal = len(_feature_indices(
        temporal_cols if temporal_cols is not None else LSTM_TEMPORAL_COLS,
        SEQUENCE_TEMPORAL_COLS, "temporal",
    ))
    n_static = len(_feature_indices(
        static_cols if static_cols is not None else LSTM_STATIC_COLS,
        SEQUENCE_STATIC_COLS, "static", allow_empty=True,
    ))
    return DemandRNN(n_temporal, n_static, config)


# ── train ──────────────────────────────────────────────────────────────────────

def train(
    config: LSTMArchConfig,
    config_name: str,
    seq_dir: Path = SEQ_DIR,
    models_dir: Path = MODELS_DIR,
    max_epochs: int | None = None,
    patience: int = PATIENCE,
    temporal_cols: list[str] | None = None,
    static_cols: list[str] | None = None,
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
    # resolve module defaults up front so the checkpoint records the effective selection
    if temporal_cols is None:
        temporal_cols = LSTM_TEMPORAL_COLS
    if static_cols is None:
        static_cols = LSTM_STATIC_COLS

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"best_{config_name}.pt"

    set_seed(SEED)
    train_ds = DemandDataset("train", seq_dir=seq_dir, temporal_cols=temporal_cols, static_cols=static_cols)
    val_ds = DemandDataset("val", seq_dir=seq_dir, temporal_cols=temporal_cols, static_cols=static_cols)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Diagnostic loader: fixed random subsample of train, evaluated in eval mode each
    # epoch so train_rmsle is directly comparable to val_rmsle (same metric, no dropout
    # noise, ~same cost as the val pass). train_loss (MSE, dropout on) is NOT comparable
    # to val_rmsle; this is. A large train↔val RMSLE gap ⇒ memorisation; a small gap
    # with val still rising ⇒ train/val regime shift (walk-forward split), which no
    # amount of regularisation will fix.
    diag_n = min(len(train_ds), len(val_ds))
    diag_idx = torch.randperm(
        len(train_ds), generator=torch.Generator().manual_seed(SEED)
    )[:diag_n].tolist()
    diag_loader = DataLoader(
        Subset(train_ds, diag_idx), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    model = build_model(config, temporal_cols=temporal_cols, static_cols=static_cols).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE, min_lr=MIN_LR,
    )

    best_val_rmsle = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses: list[float] = []
    train_rmsles: list[float] = []   # eval-mode RMSLE on the fixed train subsample
    val_rmsles: list[float] = []
    last_epoch = 0

    LOG_EVERY = 10  # print a summary line every N epochs

    n_t = len(temporal_cols) if temporal_cols is not None else N_TEMPORAL
    n_s = len(static_cols) if static_cols is not None else N_STATIC
    print(
        f"[train] {config_name} | device={device} | "
        f"epochs={max_epochs} | patience={patience} | "
        f"train_samples={len(train_ds)} | val_samples={len(val_ds)} | "
        f"features: temporal={n_t}/{N_TEMPORAL} static={n_s}/{N_STATIC}"
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

        def _eval_rmsle(loader: DataLoader) -> float:
            preds, targets = [], []
            with torch.no_grad():
                for xt, xs, yb in loader:
                    preds.append(model(xt.to(device), xs.to(device)).cpu().numpy())
                    targets.append(yb.numpy())
            y_pred = np.concatenate(preds)
            y_true = np.concatenate(targets)
            return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        val_rmsle = _eval_rmsle(val_loader)
        train_rmsle = _eval_rmsle(diag_loader)
        val_rmsles.append(val_rmsle)
        train_rmsles.append(train_rmsle)
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
                    # feature selection (None = all); evaluation rebuilds matching widths
                    "temporal_cols": temporal_cols,
                    "static_cols": static_cols,
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
                f"train_rmsle={train_rmsle:.4f} | "
                f"val_rmsle={val_rmsle:.4f} | "
                f"gap={val_rmsle - train_rmsle:+.4f} | "
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
        "train_rmsles": train_rmsles,   # eval-mode, same scale as val_rmsles
        "val_rmsles": val_rmsles,
        "checkpoint_path": str(ckpt_path),
    }


# ── run_ablation ───────────────────────────────────────────────────────────────

def run_ablation(
    seq_dir: Path = SEQ_DIR,
    models_dir: Path = MODELS_DIR,
    temporal_cols: list[str] | None = None,
    static_cols: list[str] | None = None,
) -> str:
    """Train Config A and Config B; return name of the best config."""
    from evaluate import append_experiment_log, plot_training_curves, plot_lstm_config_predictions  # avoid import cycle at module load

    models_dir = Path(models_dir)
    log_path = Path("results/tables/lstm_arch_experiments.csv")

    configs = [
        ("config_a",    CONFIG_A),
        ("config_b",    CONFIG_B),
        ("config_c",    CONFIG_C),
        ("config_a_bi", CONFIG_A_BI),
        ("config_b_bi", CONFIG_B_BI),
    ]
    states: dict[str, dict] = {}

    for name, cfg in configs:
        print(f"\n[run_ablation] Training {name} ...")
        state = train(cfg, name, seq_dir=seq_dir, models_dir=models_dir,
                      temporal_cols=temporal_cols, static_cols=static_cols)
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