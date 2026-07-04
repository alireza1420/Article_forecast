"""PG-S2S canonical-protocol evaluation (feature 007): ε=0 eval-split decode of
the 7 FR-8 systems over ≥3 seeds, rolling-origin tree regeneration (DEC-3),
contract prediction CSVs, comparison tables and figures (FR-8/12/13).

REQUIRES CUDA for model inference (D-015). Metrics are imported from
src/evaluate.py (Constitution V) — never reimplemented.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from pgs2s_data import PGS2S_DIR, SEQ_DIR, from_feedback_scale
    from pgs2s_model import Seq2SeqLSTM, encoder_input, seed_from_window
    from pgs2s_agent import PolicyNet
    from pgs2s_train import RUNS_DIR, _load_latest_checkpoint, require_cuda
    from evaluate import compute_mae, compute_mape, compute_rmsle, compute_smape
except ModuleNotFoundError:
    from src.pgs2s_data import PGS2S_DIR, SEQ_DIR, from_feedback_scale
    from src.pgs2s_model import Seq2SeqLSTM, encoder_input, seed_from_window
    from src.pgs2s_agent import PolicyNet
    from src.pgs2s_train import RUNS_DIR, _load_latest_checkpoint, require_cuda
    from src.evaluate import compute_mae, compute_mape, compute_rmsle, compute_smape

_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = _ROOT / "results"

SEED: int = 42
HORIZON: int = 10
EVAL_CHUNK: int = 4096
METRICS: tuple[str, ...] = ("rmsle", "mae", "mape", "smape")
# FR-8 comparison systems (registry cycle-8 names + existing baselines)
SYSTEMS: tuple[str, ...] = (
    "pgs2s", "s2s_free", "s2s_tf", "s2s_teach_xgb", "s2s_teach_lgbm",
    "demandrnn", "lstm_ims",
)
H_COLS: list[str] = [f"h{h:02d}" for h in range(1, HORIZON + 1)]
STD_COLS: list[str] = [f"h{h:02d}_std" for h in range(1, HORIZON + 1)]

FR13_FOOTNOTE: str = (
    "# FR-13 deviation (DEC-2): the PG-S2S policy was trained on val-split anchors "
    "116-125 and its REINFORCE reward read ground truth in weeks 117-135, overlapping "
    "the evaluation window's target weeks; sanctioned as a scoped exception by "
    "constitution v1.2.0. Eval-split samples never contributed to any parameter update."
)


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV with the mandatory FR-13 deviation footnote as a leading # comment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(FR13_FOOTNOTE + "\n")
        df.to_csv(fh, index=False)
    return path


# ── Pure table/CSV builders (T030-tested) ─────────────────────────────────────

def comparison_table(per_system: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """FR-8 schema [model, h01..h10, h01_std..h10_std]: mean±std across seeds;
    single-seed (deterministic) systems carry std 0."""
    rows = []
    ordered = [s for s in SYSTEMS if s in per_system]
    ordered += [s for s in per_system if s not in SYSTEMS]
    for name in ordered:
        stacked = np.stack(per_system[name])          # (n_seeds, 10)
        row: dict = {"model": name}
        row.update({c: float(v) for c, v in zip(H_COLS, stacked.mean(axis=0))})
        row.update({c: float(v) for c, v in zip(STD_COLS, stacked.std(axis=0))})
        rows.append(row)
    return pd.DataFrame(rows, columns=["model"] + H_COLS + STD_COLS)


def write_prediction_csvs(model_name: str, meta: np.ndarray, preds_raw: np.ndarray,
                          out_dir: Path | None = None) -> list[Path]:
    """Contract files pred_{model}_h{hh}.csv: [center_id, meal_id, week, y_pred, split];
    week = target week (anchor + h, D-013); y_pred raw orders ≥ 0."""
    if (preds_raw < 0).any():
        raise ValueError("[pgs2s] negative raw-order predictions — expm1+clip upstream")
    out_dir = RESULTS_DIR / "predictions" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for h in range(1, HORIZON + 1):
        df = pd.DataFrame({
            "center_id": meta[:, 0].astype(np.int64),
            "meal_id": meta[:, 1].astype(np.int64),
            "week": meta[:, 2].astype(np.int64) + h,
            "y_pred": preds_raw[:, h - 1],
            "split": "eval",
        })
        path = out_dir / f"pred_{model_name}_h{h:02d}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def _metric_arrays(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, np.ndarray]:
    """All four per-horizon metric vectors from src/evaluate.py (Constitution V)."""
    mape, _ = compute_mape(y_true_raw, y_pred_raw)
    return {"rmsle": compute_rmsle(y_true_raw, y_pred_raw),
            "mae": compute_mae(y_true_raw, y_pred_raw),
            "mape": mape,
            "smape": compute_smape(y_true_raw, y_pred_raw)}


# ── DEC-3 rolling-origin tree regeneration (T032; CPU-safe) ───────────────────

def regenerate_rolling_origin_trees(out_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """DEC-3 supplementary table: xgb/lgbm rolling-origin eval metrics from the eval aux caches."""
    y_true_raw = np.expm1(np.load(SEQ_DIR / "eval_y.npy").astype(np.float64))
    per_metric: dict[str, list[dict]] = {m: [] for m in METRICS}
    result_rows = []
    for model_name, cache_name in (("xgboost", "eval_aux_xgb.npy"),
                                   ("lightgbm", "eval_aux_lgbm.npy")):
        cache = np.load(PGS2S_DIR / cache_name).astype(np.float64)
        pred_raw = np.expm1(np.clip(from_feedback_scale(cache), 0.0, None))
        metrics = _metric_arrays(y_true_raw, pred_raw)
        for metric in METRICS:
            row = {"model": model_name}
            row.update({c: float(v) for c, v in zip(H_COLS, metrics[metric])})
            per_metric[metric].append(row)
        result_rows.append({"model": model_name,
                            **{c: float(v) for c, v in zip(H_COLS, metrics["rmsle"])}})

    tables_dir = Path(out_dir) / "tables"
    for metric in METRICS:
        _write_table(pd.DataFrame(per_metric[metric], columns=["model"] + H_COLS),
                     tables_dir / f"pgs2s_trees_rolling_origin_{metric}.csv")
    return pd.DataFrame(result_rows, columns=["model"] + H_COLS)


# ── Seq2seq system decode (T031; CUDA) ────────────────────────────────────────

def _decode_system(model: Seq2SeqLSTM, policy: PolicyNet | None, system: str,
                   tensors: dict, device: torch.device) -> np.ndarray:
    """ε=0 eval-split decode of one seq2seq system; returns (N, 10) log1p preds."""
    model.eval()
    n = tensors["x"].shape[0]
    parts = []
    with torch.no_grad():
        for lo in range(0, n, EVAL_CHUNK):
            sl = slice(lo, lo + EVAL_CHUNK)
            x, seed = tensors["x"][sl], tensors["y_seed"][sl]
            if system == "pgs2s":
                out = model(x, seed, mode="policy", aux=tensors["aux"][sl],
                            policy=policy, epsilon=0.0)
            elif system == "s2s_free":
                out = model(x, seed, mode="free_running")
            elif system == "s2s_tf":
                out = model(x, seed, mode="teacher_forcing", teacher=tensors["y"][sl])
            elif system in ("s2s_teach_xgb", "s2s_teach_lgbm"):
                const = 1 if system == "s2s_teach_xgb" else 2
                actions = torch.full((x.shape[0], HORIZON), const,
                                     dtype=torch.int64, device=device)
                out = model(x, seed, mode="policy", aux=tensors["aux"][sl],
                            actions=actions)
            else:
                raise ValueError(f"unknown seq2seq system {system!r}")
            parts.append(out.preds.cpu())
    return torch.cat(parts).numpy().astype(np.float64)


def _evaluate_demandrnn(device: torch.device) -> np.ndarray:
    """DMS baseline: best_config_c.pt forward on the eval split; (N, 10) log1p."""
    try:
        from lstm_model import DemandDataset, LSTMArchConfig, build_model
    except ModuleNotFoundError:
        from src.lstm_model import DemandDataset, LSTMArchConfig, build_model

    ckpt = torch.load(_ROOT / "results" / "models" / "lstm" / "best_config_c.pt",
                      map_location=device, weights_only=False)
    config = LSTMArchConfig.from_json(ckpt["config_json"])
    model = build_model(config, temporal_cols=ckpt["temporal_cols"],
                        static_cols=ckpt["static_cols"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = DemandDataset("eval", temporal_cols=ckpt["temporal_cols"],
                       static_cols=ckpt["static_cols"])
    parts = []
    with torch.no_grad():
        for lo in range(0, len(ds), EVAL_CHUNK):
            xt = ds.X_temp[lo:lo + EVAL_CHUNK].to(device)
            xs = ds.X_stat[lo:lo + EVAL_CHUNK].to(device)
            parts.append(model(xt, xs).cpu())
    return torch.cat(parts).numpy().astype(np.float64)


def _evaluate_ims() -> dict[str, np.ndarray]:
    """IMS baseline metrics via the reviewed lstm_ims_model implementation."""
    try:
        from lstm_ims_model import evaluate_ims
    except ModuleNotFoundError:
        from src.lstm_ims_model import evaluate_ims

    results = evaluate_ims()
    return {m: np.asarray(results[m], dtype=np.float64) for m in METRICS}


# ── Diagnostics artifacts (T033) ──────────────────────────────────────────────

def export_diagnostics(run_id: str, out_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Flatten diagnostics.jsonl to CSV and render the selection/RMSE figure."""
    diag_path = Path(RUNS_DIR) / run_id / "diagnostics.jsonl"
    records = [json.loads(line) for line in diag_path.read_text().splitlines()]
    rows = []
    for rec in records:
        row = {"round": rec["round"], "view": rec["view"], "epsilon": rec["epsilon"],
               "collapse_warning": rec["collapse_warning"]}
        row.update({f"sel_{k}": v for k, v in rec["selection_pct"].items()})
        row.update({f"rmse_{k}": v for k, v in rec["candidate_rmse_log1p"].items()})
        row.update({f"rmsle_h{h:02d}": v
                    for h, v in enumerate(rec["pgs2s_rmsle_per_h"], start=1)})
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = Path(out_dir) / "tables" / f"pgs2s_diagnostics_{run_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"policy_train": "-", "policy_holdout": "--"}
    colors = {"decoder": "tab:blue", "xgb": "tab:orange", "lgbm": "tab:green"}
    for view, style in styles.items():
        sub = df[df["view"] == view].sort_values("round")
        if sub.empty:
            continue
        for action, color in colors.items():
            axes[0].plot(sub["round"], sub[f"sel_{action}"], style, color=color,
                         label=f"{action} ({view.split('_')[1]})")
            axes[1].plot(sub["round"], sub[f"rmse_{action}"], style, color=color)
    axes[0].set_xlabel("round"), axes[0].set_ylabel("selection %")
    axes[0].set_title("Action selection per round"), axes[0].legend(fontsize=7)
    axes[1].set_xlabel("round"), axes[1].set_ylabel("candidate RMSE (log1p)")
    axes[1].set_title("Candidate accuracy per round")
    fig.suptitle(f"PG-S2S diagnostics — {run_id}")
    fig.tight_layout()
    fig_path = Path(out_dir) / "figures" / f"pgs2s_diagnostics_{run_id}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    return csv_path, fig_path


def _comparison_figure(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Per-horizon RMSLE comparison figure across the 7 systems."""
    df = tables["rmsle"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, HORIZON + 1)
    for _, row in df.iterrows():
        means = row[H_COLS].values.astype(float)
        stds = row[STD_COLS].values.astype(float)
        ax.errorbar(xs, means, yerr=stds, marker="o", markersize=3,
                    capsize=2, label=row["model"])
    ax.set_xlabel("horizon h"), ax.set_ylabel("RMSLE")
    ax.set_title("PG-S2S 7-system comparison — eval split (mean ± std over seeds)")
    ax.set_xticks(xs), ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(out_dir) / "figures" / "pgs2s_comparison_rmsle.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Orchestrator (T031/T033; CUDA) ────────────────────────────────────────────

def evaluate_all_systems(run_ids: list[str], out_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """ε=0 eval-split evaluation of the 7 FR-8 systems over ≥3 seeds; writes comparison tables, figures, prediction CSVs."""
    device = require_cuda()
    out_dir = Path(out_dir)

    x = encoder_input(np.load(SEQ_DIR / "eval_X_temporal.npy", mmap_mode="r"))
    tensors = {
        "x": torch.from_numpy(x).to(device),
        "y_seed": torch.from_numpy(seed_from_window(x)).to(device),
        "y": torch.from_numpy(np.load(SEQ_DIR / "eval_y.npy").astype(np.float32)).to(device),
        "aux": torch.from_numpy(np.stack(
            [np.load(PGS2S_DIR / "eval_aux_xgb.npy"),
             np.load(PGS2S_DIR / "eval_aux_lgbm.npy")], axis=1).astype(np.float32)).to(device),
    }
    meta = np.load(SEQ_DIR / "eval_meta.npy")
    y_true_raw = np.expm1(np.load(SEQ_DIR / "eval_y.npy").astype(np.float64))

    s2s_systems = ("pgs2s", "s2s_free", "s2s_tf", "s2s_teach_xgb", "s2s_teach_lgbm")
    per_metric: dict[str, dict[str, list[np.ndarray]]] = \
        {m: {s: [] for s in SYSTEMS} for m in METRICS}
    csv_seed_run: str | None = None
    csv_preds: dict[str, np.ndarray] = {}

    for run_id in run_ids:
        run_dir = Path(RUNS_DIR) / run_id
        ckpt = _load_latest_checkpoint(run_dir)
        if ckpt is None:
            raise RuntimeError(f"[pgs2s] no complete round checkpoint in {run_dir}")
        run_rec = json.load(open(run_dir / "run.json"))
        model = Seq2SeqLSTM().to(device)
        policy = PolicyNet().to(device)
        model.load_state_dict(ckpt["seq2seq_state"])
        policy.load_state_dict(ckpt["policy_state"])

        is_csv_run = csv_seed_run is None or run_rec.get("seed") == SEED
        for system in s2s_systems:
            preds_log1p = _decode_system(model, policy, system, tensors, device)
            preds_raw = np.expm1(preds_log1p)
            metrics = _metric_arrays(y_true_raw, preds_raw)
            for metric in METRICS:
                per_metric[metric][system].append(metrics[metric])
            if is_csv_run:
                csv_preds[system] = preds_raw
        if is_csv_run:
            csv_seed_run = run_id
        print(f"[pgs2s_eval] {run_id} (seed {run_rec.get('seed')}, "
              f"round {ckpt['round']}): 5 seq2seq systems evaluated")

    dm_preds_raw = np.expm1(_evaluate_demandrnn(device))
    dm_metrics = _metric_arrays(y_true_raw, dm_preds_raw)
    ims_metrics = _evaluate_ims()
    for metric in METRICS:
        per_metric[metric]["demandrnn"].append(dm_metrics[metric])
        per_metric[metric]["lstm_ims"].append(ims_metrics[metric])

    tables: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        tables[metric] = comparison_table(per_metric[metric])
        _write_table(tables[metric],
                     out_dir / "tables" / f"pgs2s_comparison_{metric}.csv")

    for system, preds_raw in csv_preds.items():
        write_prediction_csvs(system, meta, preds_raw, out_dir / "predictions")

    _comparison_figure(tables, out_dir)
    for run_id in run_ids:
        export_diagnostics(run_id, out_dir)

    print(f"[pgs2s_eval] tables/figures/predictions written under {out_dir} "
          f"(prediction CSVs from run {csv_seed_run})")
    return tables["rmsle"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PG-S2S 7-system evaluation (CUDA)")
    parser.add_argument("--runs", nargs="+", default=None,
                        metavar="RUN_ID", help="trained run ids (>= 3 seeds)")
    args = parser.parse_args()
    if args.runs:
        regenerate_rolling_origin_trees()
        df = evaluate_all_systems(args.runs)
        print(df.to_string(index=False))
    else:
        print(f"[pgs2s_eval] SYSTEMS={SYSTEMS}")
        print("[pgs2s_eval] smoke OK (usage: --runs <id42> <id43> <id44>)")
