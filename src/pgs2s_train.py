"""PG-S2S asynchronous alternation trainer (feature 007): paper Alg. 1 with the
DEC-2 split restriction, teacher-forcing warm-up, per-round checkpoints, dual
val/cal diagnostics, resume, and the DEC-6 pilot gate (FR-7/12).

REQUIRES CUDA — raises RuntimeError at entry when no GPU is available (D-015).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

# D-008: the cuBLAS workspace config must exist before the first CUDA call —
# set it at import time so no entry-point ordering can defeat it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

try:
    from pgs2s_agent import (ALPHA, BETA, GAMMA, EPS_START, EPS_DECAY, EPS_MIN, N_P,
                             PolicyNet, compute_rewards, discounted_returns,
                             reinforce_update)
    from pgs2s_data import (PGS2S_DIR, SEQ_DIR, _sha256_of, from_feedback_scale,
                            verify_manifest)
    from pgs2s_model import Seq2SeqLSTM, encoder_input, seed_from_window
except ModuleNotFoundError:
    from src.pgs2s_agent import (ALPHA, BETA, GAMMA, EPS_START, EPS_DECAY, EPS_MIN, N_P,
                                 PolicyNet, compute_rewards, discounted_returns,
                                 reinforce_update)
    from src.pgs2s_data import (PGS2S_DIR, SEQ_DIR, _sha256_of, from_feedback_scale,
                                verify_manifest)
    from src.pgs2s_model import Seq2SeqLSTM, encoder_input, seed_from_window

# ── Training constants (research D-008) ───────────────────────────────────────

LR_POLICY: float = 1e-3   # l₁ — policy Adam LR
LR_RNN: float = 1e-4      # l₂ — seq2seq Adam LR (repo convention)
BATCH_SIZE: int = 64
ROUNDS: int = 10          # full-scale alternation rounds
AGENT_EPOCHS: int = 5     # policy passes per round (full scale)
RNN_EPOCHS: int = 10      # seq2seq epochs per round (full scale)
# Pilot overrides (spec §Pilot Configuration)
PILOT_ROUNDS: int = 3
PILOT_AGENT_EPOCHS: int = 3
PILOT_RNN_EPOCHS: int = 5

SEED: int = 42
RUN_SEEDS: tuple[int, ...] = (42, 43, 44)

POLICY_WEEKS: tuple[int, int] = (116, 125)   # DEC-2 policy-phase anchor window
TRAIN_WEEKS: tuple[int, int] = (26, 105)     # RNN-phase anchor window
COLLAPSE_PCT: float = 95.0                   # single-action collapse threshold
EVAL_CHUNK: int = 4096                       # diagnostics decode chunk size

_ROOT: Path = Path(__file__).resolve().parent.parent
RUNS_DIR: Path = _ROOT / "results" / "models" / "pgs2s"


def require_cuda() -> torch.device:
    """Return cuda device or raise RuntimeError('[pgs2s] CUDA is required…')."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[pgs2s] CUDA is required — run on a GPU machine/Colab "
            "(data/cache building via pgs2s_data.py is CPU-safe)."
        )
    return torch.device("cuda")


def set_determinism(seed: int) -> None:
    """Seed python/numpy/torch and enable full CUDA determinism (research D-008, SC-006)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # also pinned at module import
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def code_hash() -> str:
    """sha256 over sorted contents of src/pgs2s_{data,model,agent,train}.py — the DEC-6 gate key (D-012)."""
    src_dir = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in sorted(["pgs2s_data.py", "pgs2s_model.py", "pgs2s_agent.py", "pgs2s_train.py"]):
        h.update((src_dir / name).read_bytes())
    return h.hexdigest()


def _git_commit() -> str:
    """Current git HEAD sha for traceability (never part of the DEC-6 gate key)."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# ── Alternation phases (T026) ─────────────────────────────────────────────────

def policy_phase(model: Seq2SeqLSTM, policy: PolicyNet,
                 opt_policy: torch.optim.Optimizer, view: dict,
                 epsilon: float, agent_epochs: int, batch_size: int) -> float:
    """Seq2seq frozen (no_grad): collect ε-greedy trajectories on the policy-train
    view and REINFORCE-update the policy (paper Alg. 1 policy step; D-011)."""
    model.eval()
    n = view["x"].shape[0]
    losses: list[float] = []
    for _ in range(agent_epochs):
        perm = torch.randperm(n)
        for lo in range(0, n, batch_size):
            idx = perm[lo:lo + batch_size]
            with torch.no_grad():
                out = model(view["x"][idx], view["y_seed"][idx], mode="policy",
                            aux=view["aux"][idx], policy=policy, epsilon=epsilon)
            rewards = compute_rewards(out.preds, from_feedback_scale(view["aux"][idx]),
                                      view["y"][idx], out.actions)
            returns = discounted_returns(rewards)
            losses.append(reinforce_update(policy, opt_policy, out.states,
                                           out.actions, returns))
    return float(np.mean(losses))


def rnn_phase(model: Seq2SeqLSTM, policy: PolicyNet,
              opt_rnn: torch.optim.Optimizer, train: dict,
              rnn_epochs: int, batch_size: int) -> float:
    """Policy frozen: greedy (ε=0) policy-mode decode on the train split,
    MSE on log1p targets (DEC-5), update the seq2seq (paper Alg. 1 RNN step)."""
    crit = nn.MSELoss()
    n = train["x"].shape[0]
    losses: list[float] = []
    for _ in range(rnn_epochs):
        model.train()
        perm = torch.randperm(n)
        for lo in range(0, n, batch_size):
            idx = perm[lo:lo + batch_size]
            out = model(train["x"][idx], train["y_seed"][idx], mode="policy",
                        aux=train["aux"][idx], policy=policy, epsilon=0.0)
            loss = crit(out.preds, train["y"][idx])
            opt_rnn.zero_grad()
            loss.backward()
            opt_rnn.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def warmup_phase(model: Seq2SeqLSTM, opt_rnn: torch.optim.Optimizer, train: dict,
                 rnn_epochs: int, batch_size: int) -> float:
    """Round 0: teacher-forcing warm-up so the first policy phase ranks sane
    decoder candidates (D-011)."""
    crit = nn.MSELoss()
    n = train["x"].shape[0]
    losses: list[float] = []
    for _ in range(rnn_epochs):
        model.train()
        perm = torch.randperm(n)
        for lo in range(0, n, batch_size):
            idx = perm[lo:lo + batch_size]
            out = model(train["x"][idx], train["y_seed"][idx],
                        mode="teacher_forcing", teacher=train["y"][idx])
            loss = crit(out.preds, train["y"][idx])
            opt_rnn.zero_grad()
            loss.backward()
            opt_rnn.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


# ── Diagnostics (T027) ────────────────────────────────────────────────────────

ACTION_NAMES: tuple[str, ...] = ("decoder", "xgb", "lgbm")


def _diagnose_view(model: Seq2SeqLSTM, policy: PolicyNet, view: dict) -> dict:
    """ε=0 greedy decode of a policy view: selection %, candidate RMSE, per-h RMSLE."""
    model.eval()
    n = view["x"].shape[0]
    preds_parts, actions_parts = [], []
    with torch.no_grad():
        for lo in range(0, n, EVAL_CHUNK):
            sl = slice(lo, lo + EVAL_CHUNK)
            out = model(view["x"][sl], view["y_seed"][sl], mode="policy",
                        aux=view["aux"][sl], policy=policy, epsilon=0.0)
            preds_parts.append(out.preds)
            actions_parts.append(out.actions)
    preds = torch.cat(preds_parts)
    actions = torch.cat(actions_parts)
    y = view["y"]
    aux_log1p = from_feedback_scale(view["aux"])

    counts = torch.bincount(actions.reshape(-1), minlength=3).float()
    selection_pct = {name: float(100.0 * counts[i] / counts.sum())
                     for i, name in enumerate(ACTION_NAMES)}
    candidate_rmse = {
        "decoder": float(torch.sqrt(torch.mean((preds - y) ** 2))),
        "xgb": float(torch.sqrt(torch.mean((aux_log1p[:, 0, :] - y) ** 2))),
        "lgbm": float(torch.sqrt(torch.mean((aux_log1p[:, 1, :] - y) ** 2))),
    }
    per_h = torch.sqrt(torch.mean((preds - y) ** 2, dim=0)).tolist()
    return {"selection_pct": selection_pct, "candidate_rmse_log1p": candidate_rmse,
            "pgs2s_rmsle_per_h": per_h}


# ── Startup assertions & gates (T026/T028) ────────────────────────────────────

def _assert_policy_views(data: dict) -> None:
    """Pair-disjointness + full anchor-week coverage of both policy views (US4-AS4)."""
    pt = {(int(c), int(m)) for c, m in data["policy_train"]["meta"][:, :2]}
    ph = {(int(c), int(m)) for c, m in data["policy_holdout"]["meta"][:, :2]}
    overlap = pt & ph
    if overlap:
        raise RuntimeError(
            f"[pgs2s] policy views are NOT pair-disjoint — {len(overlap)} shared "
            f"pair(s), e.g. {sorted(overlap)[:5]} (US4-AS4 abort)")
    lo, hi = POLICY_WEEKS
    for name in ("policy_train", "policy_holdout"):
        weeks = set(np.unique(data[name]["meta"][:, 2]).astype(int))
        missing = set(range(lo, hi + 1)) - weeks
        if missing:
            raise RuntimeError(
                f"[pgs2s] {name} view missing anchor week(s) {sorted(missing)} "
                f"— every week {lo}-{hi} must appear in both views")


def _assert_anchor_bounds(data: dict) -> None:
    """Constitution II: RNN-phase anchors ≤ 105, policy anchors ∈ [116, 125]."""
    checks = [("train", TRAIN_WEEKS), ("policy_train", POLICY_WEEKS),
              ("policy_holdout", POLICY_WEEKS)]
    for name, (lo, hi) in checks:
        weeks = data[name]["meta"][:, 2]
        if len(weeks) and (weeks.min() < lo or weeks.max() > hi):
            raise RuntimeError(
                f"[pgs2s] {name} anchor weeks outside [{lo}, {hi}]: "
                f"observed [{int(weeks.min())}, {int(weeks.max())}] (Constitution II)")


def _check_full_gate(runs_dir: Path, manifest_sha: str) -> None:
    """DEC-6: a full run needs a successful pilot whose code_hash + manifest_sha256 match (D-012)."""
    current = code_hash()
    for run_json in sorted(Path(runs_dir).glob("*/run.json")):
        try:
            rec = json.load(open(run_json))
        except (OSError, json.JSONDecodeError):
            continue
        if (rec.get("pilot") is True and rec.get("status") == "success"
                and rec.get("code_hash") == current
                and rec.get("manifest_sha256") == manifest_sha):
            return
    raise RuntimeError(
        "[pgs2s] --full refused: no successful pilot record with matching "
        "code_hash + manifest_sha256 found (DEC-6, D-012). Run "
        "`python src/pgs2s_train.py --pilot` first; training-code or data "
        "changes invalidate earlier pilots.")


# ── Data loading (T028) ───────────────────────────────────────────────────────

def load_real_data(pilot: bool) -> dict:
    """Load train + masked policy views from the manifest-verified artifacts;
    pilot=True restricts every view to pilot_pairs.npy members (DEC-6)."""

    def load_split(split: str) -> dict:
        x = encoder_input(np.load(SEQ_DIR / f"{split}_X_temporal.npy", mmap_mode="r"))
        return {
            "x": x,
            "y_seed": seed_from_window(x),
            "y": np.load(SEQ_DIR / f"{split}_y.npy").astype(np.float32),
            "aux": np.stack([np.load(PGS2S_DIR / f"{split}_aux_xgb.npy"),
                             np.load(PGS2S_DIR / f"{split}_aux_lgbm.npy")],
                            axis=1).astype(np.float32),
            "meta": np.load(SEQ_DIR / f"{split}_meta.npy"),
        }

    train = load_split("train")
    val = load_split("val")
    cal = load_split("cal")
    train_mask = np.load(PGS2S_DIR / "policy_train_mask.npy")
    holdout_mask = np.load(PGS2S_DIR / "policy_holdout_mask.npy")
    data = {
        "train": train,
        "policy_train": {k: v[train_mask] for k, v in val.items()},
        "policy_holdout": {k: v[holdout_mask] for k, v in cal.items()},
    }

    if pilot:
        pilot_pairs = {(int(c), int(m))
                       for c, m in np.load(PGS2S_DIR / "pilot_pairs.npy")}
        for name, block in data.items():
            keep = np.fromiter(((int(c), int(m)) in pilot_pairs
                                for c, m in block["meta"][:, :2]),
                               dtype=bool, count=len(block["meta"]))
            data[name] = {k: v[keep] for k, v in block.items()}
            print(f"[pgs2s] pilot {name}: {int(keep.sum())} rows "
                  f"({len({(int(c), int(m)) for c, m in data[name]['meta'][:, :2]})} pairs)")
        # spec edge case "sparse pilot pairs": log pairs with incomplete anchor coverage
        for name, block in data.items():
            meta = block["meta"]
            if not len(meta):
                continue
            n_weeks = len(np.unique(meta[:, 2]))
            sparse = []
            for pair in {(int(c), int(m)) for c, m in meta[:, :2]}:
                rows = (meta[:, 0] == pair[0]) & (meta[:, 1] == pair[1])
                if len(np.unique(meta[rows][:, 2])) < n_weeks:
                    sparse.append(pair)
            if sparse:
                print(f"[pgs2s] pilot {name}: {len(sparse)} pair(s) with incomplete "
                      f"anchor coverage (informational): {sorted(sparse)[:10]}")
    return data


# ── Training loop (T026/T027) ─────────────────────────────────────────────────

def _hyperparameters(rounds: int, agent_epochs: int, rnn_epochs: int,
                     batch_size: int) -> dict:
    return {"ALPHA": ALPHA, "BETA": BETA, "GAMMA": GAMMA, "EPS_START": EPS_START,
            "EPS_DECAY": EPS_DECAY, "EPS_MIN": EPS_MIN, "N_P": N_P,
            "LR_POLICY": LR_POLICY, "LR_RNN": LR_RNN, "BATCH_SIZE": batch_size,
            "ROUNDS": rounds, "AGENT_EPOCHS": agent_epochs, "RNN_EPOCHS": rnn_epochs}


def _save_checkpoint(run_dir: Path, rnd: int, model: Seq2SeqLSTM, policy: PolicyNet,
                     opt_rnn: torch.optim.Optimizer, opt_policy: torch.optim.Optimizer,
                     epsilon: float, device: torch.device) -> None:
    ckpt = {
        "round": rnd,
        "seq2seq_state": model.state_dict(),
        "policy_state": policy.state_dict(),
        "opt_rnn_state": opt_rnn.state_dict(),
        "opt_policy_state": opt_policy.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "epsilon": epsilon,
    }
    if device.type == "cuda":
        ckpt["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, run_dir / f"round_{rnd:02d}.pt")


def _load_latest_checkpoint(run_dir: Path) -> dict | None:
    """Highest complete round checkpoint; incomplete/corrupt entries are discarded (D-012)."""
    latest = None
    for path in sorted(run_dir.glob("round_*.pt")):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if all(k in ckpt for k in ("round", "seq2seq_state", "policy_state",
                                       "opt_rnn_state", "opt_policy_state",
                                       "torch_rng", "numpy_rng", "epsilon")):
                latest = ckpt
            else:
                print(f"[pgs2s] discarding incomplete checkpoint {path.name}")
        except Exception:
            print(f"[pgs2s] discarding unreadable checkpoint {path.name}")
    return latest


def run_training(seed: int = SEED, pilot: bool = True,
                 run_dir: Path | None = None, resume: bool = False,
                 data: dict | None = None, device: str | None = None,
                 rounds: int | None = None, agent_epochs: int | None = None,
                 rnn_epochs: int | None = None, batch_size: int = BATCH_SIZE,
                 runs_dir: Path = RUNS_DIR, manifest_sha: str | None = None,
                 run_id: str | None = None) -> dict:
    """Full Alg.-1 alternation with DEC-2 split restriction; per-round checkpoints + dual diagnostics; returns run summary."""
    # codex: The device override is useful for synthetic unit tests, but the
    # spec requires real training entry points to hard-require CUDA. Guard
    # data=None + device="cpu" so direct run_training() cannot bypass D-015.
    dev = torch.device(device) if device is not None else require_cuda()
    set_determinism(seed)

    rounds = rounds if rounds is not None else (PILOT_ROUNDS if pilot else ROUNDS)
    agent_epochs = agent_epochs if agent_epochs is not None else \
        (PILOT_AGENT_EPOCHS if pilot else AGENT_EPOCHS)
    rnn_epochs = rnn_epochs if rnn_epochs is not None else \
        (PILOT_RNN_EPOCHS if pilot else RNN_EPOCHS)

    if data is None:
        data = load_real_data(pilot)
    if manifest_sha is None:
        manifest_sha = _sha256_of(PGS2S_DIR / "manifest.json")

    _assert_policy_views(data)
    _assert_anchor_bounds(data)
    if not pilot:
        _check_full_gate(runs_dir, manifest_sha)

    if run_id is None:
        run_id = f"{'pilot' if pilot else 'full'}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(runs_dir) / run_id if run_dir is None else Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = run_dir / "run.json"
    diag_path = run_dir / "diagnostics.jsonl"

    tensors = {name: {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev)
                      for k, v in block.items() if k != "meta"}
               for name, block in data.items()}

    model = Seq2SeqLSTM().to(dev)
    policy = PolicyNet().to(dev)
    opt_rnn = torch.optim.Adam(model.parameters(), lr=LR_RNN)
    opt_policy = torch.optim.Adam(policy.parameters(), lr=LR_POLICY)

    epsilon = EPS_START
    start_round = 1
    prev_holdout_high = False

    if resume:
        ckpt = _load_latest_checkpoint(run_dir)
        if ckpt is None:
            raise RuntimeError(f"[pgs2s] resume requested but no complete round "
                               f"checkpoint found in {run_dir}")
        model.load_state_dict(ckpt["seq2seq_state"])
        policy.load_state_dict(ckpt["policy_state"])
        opt_rnn.load_state_dict(ckpt["opt_rnn_state"])
        opt_policy.load_state_dict(ckpt["opt_policy_state"])
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        if dev.type == "cuda" and "cuda_rng" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        start_round = int(ckpt["round"]) + 1
        epsilon = max(EPS_MIN, float(ckpt["epsilon"]) * EPS_DECAY)
        if diag_path.exists():                      # restore collapse tracking
            for line in diag_path.read_text().splitlines():
                rec = json.loads(line)
                if rec["view"] == "policy_holdout":
                    prev_holdout_high = max(rec["selection_pct"].values()) >= COLLAPSE_PCT
        # codex: D-012 says an interrupted round is re-run and never partially
        # trusted. If diagnostics.jsonl already contains stale lines for
        # start_round or later, this appends duplicates instead of truncating to
        # the highest checkpointed round before resuming.
        record = json.load(open(run_json_path)) if run_json_path.exists() else {}
        record.update({"status": "running"})
        print(f"[pgs2s] resuming {run_id} at round {start_round} (epsilon={epsilon:.3f})")
    else:
        record = {
            "run_id": run_id, "seed": seed, "git_commit": _git_commit(),
            "code_hash": code_hash(), "pilot": pilot,
            "hyperparameters": _hyperparameters(rounds, agent_epochs, rnn_epochs,
                                                batch_size),
            "manifest_sha256": manifest_sha, "status": "running",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished": None,
        }
    json.dump(record, open(run_json_path, "w"), indent=2)

    try:
        if start_round == 1:                       # fresh run: TF warm-up (round 0)
            t0 = time.time()
            warm_loss = warmup_phase(model, opt_rnn, tensors["train"],
                                     rnn_epochs, batch_size)
            print(f"[pgs2s] warm-up (TF x{rnn_epochs}) loss={warm_loss:.4f} "
                  f"[{time.time() - t0:.0f}s]")

        for rnd in range(start_round, rounds + 1):
            t0 = time.time()
            p_loss = policy_phase(model, policy, opt_policy, tensors["policy_train"],
                                  epsilon, agent_epochs, batch_size)
            r_loss = rnn_phase(model, policy, opt_rnn, tensors["train"],
                               rnn_epochs, batch_size)

            holdout_high = False
            for view_name in ("policy_train", "policy_holdout"):
                diag = _diagnose_view(model, policy, tensors[view_name])
                max_sel = max(diag["selection_pct"].values())
                if view_name == "policy_holdout":
                    holdout_high = max_sel >= COLLAPSE_PCT
                    collapse = holdout_high and prev_holdout_high
                else:
                    collapse = False
                rec = {"round": rnd, "view": view_name, **diag,
                       "epsilon": epsilon, "collapse_warning": collapse}
                with open(diag_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if collapse:
                    print(f"[pgs2s] WARNING round {rnd}: action collapse on holdout "
                          f"(>= {COLLAPSE_PCT}% single action, 2 consecutive rounds)")
            prev_holdout_high = holdout_high

            _save_checkpoint(run_dir, rnd, model, policy, opt_rnn, opt_policy,
                             epsilon, dev)
            print(f"[pgs2s] round {rnd}/{rounds} policy_loss={p_loss:.4f} "
                  f"rnn_loss={r_loss:.4f} epsilon={epsilon:.3f} "
                  f"[{time.time() - t0:.0f}s]")
            epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
    except Exception:
        record.update({"status": "failed",
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        json.dump(record, open(run_json_path, "w"), indent=2)
        raise

    record.update({"status": "success",
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    json.dump(record, open(run_json_path, "w"), indent=2)
    return {"run_id": run_id, "run_dir": str(run_dir), "status": "success",
            "rounds": rounds, "seed": seed, "pilot": pilot}


# ── CLI + startup gates (T028) ────────────────────────────────────────────────

def _check_toy_gate() -> None:
    """FR-14: refuse demand-data training while any agent toy test is red."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(_ROOT / "tests" / "test_pgs2s_agent.py"),
         "-q", "--no-header"], capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            "[pgs2s] FR-14 toy-test gate RED — fix tests/test_pgs2s_agent.py before "
            "any demand-data training:\n" + result.stdout[-2000:])


def main() -> None:
    """CLI: python src/pgs2s_train.py --pilot|--full [--seed 42] [--resume RUN_ID]."""
    import argparse

    parser = argparse.ArgumentParser(description="PG-S2S alternation trainer (CUDA)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="100-pair pilot run (DEC-6)")
    group.add_argument("--full", action="store_true", help="full-scale run (pilot-gated)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_ID")
    args = parser.parse_args()

    require_cuda()                                 # startup sequence per contract
    _check_toy_gate()
    verify_manifest()

    resume = args.resume is not None
    summary = run_training(seed=args.seed, pilot=args.pilot, resume=resume,
                           run_id=args.resume if resume else None)
    print(f"[pgs2s_train] {summary['run_id']}: {summary['status']} "
          f"({summary['rounds']} rounds)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        print(f"[pgs2s_train] code_hash={code_hash()[:12]}")
        print("[pgs2s_train] smoke OK (usage: --pilot | --full [--seed N] [--resume RUN_ID])")
