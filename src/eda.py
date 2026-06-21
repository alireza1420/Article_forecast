"""EDA analysis and visualisation for the CDI project."""
import matplotlib
matplotlib.use("Agg")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")

SEED = 42
ACF_MAX_LAGS = 52
GRID_N_PAIRS = 9
DPI = 300


# ---------------------------------------------------------------------------
# US2 — profile_dataset
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> dict:
    """Return shape, dtypes, missing-value counts, unique counts, and num_orders stats."""
    if "num_orders" not in df.columns:
        raise KeyError("'num_orders' column is required")

    history = df.groupby(["center_id", "meal_id"])["week"].nunique()

    missing = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        missing[col] = {"count": n, "pct": round(n / len(df) * 100, 4)}

    unique_keys = ["center_id", "meal_id", "week", "center_type", "cuisine", "category"]
    unique_counts = {k: int(df[k].nunique()) for k in unique_keys if k in df.columns}

    orders = df["num_orders"]
    return {
        "shape": df.shape,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": missing,
        "unique_counts": unique_counts,
        "num_orders_stats": {
            "mean": float(orders.mean()),
            "std": float(orders.std()),
            "median": float(orders.median()),
            "p95": float(orders.quantile(0.95)),
            "min": float(orders.min()),
            "max": float(orders.max()),
            "zero_order_rate": float((orders == 0).sum() / len(orders)),
        },
        "history_lengths": {
            "min": int(history.min()),
            "median": float(history.median()),
            "max": int(history.max()),
            "distribution": history.value_counts().sort_index(),
        },
    }


# ---------------------------------------------------------------------------
# US3 — plot_order_distribution
# ---------------------------------------------------------------------------

def plot_order_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save histogram + boxplot of num_orders; return path to saved PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    orders = df["num_orders"]

    sns.histplot(orders, ax=ax1, bins=50, kde=True)
    ax1.set_title("Distribution of num_orders")
    ax1.set_xlabel("num_orders")
    ax1.set_ylabel("Count")

    sns.boxplot(x=orders, ax=ax2, orient="h")
    ax2.set_title("Boxplot of num_orders")
    ax2.set_xlabel("num_orders")

    fig.tight_layout()
    out = out_dir / "fig1_order_distribution.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# US4 — plot_timeseries_grid
# ---------------------------------------------------------------------------

def plot_timeseries_grid(
    df: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
    n_pairs: int = 9,
) -> Path:
    """Sample n_pairs (center_id, meal_id) pairs and plot time-series grid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = df[["center_id", "meal_id"]].drop_duplicates().values.tolist()
    rng = np.random.default_rng(seed)
    n_sample = min(n_pairs, len(pairs))
    indices = rng.choice(len(pairs), size=n_sample, replace=False)
    sampled = [pairs[i] for i in indices]

    ncols = int(np.ceil(np.sqrt(n_sample)))
    nrows = int(np.ceil(n_sample / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, (center_id, meal_id) in enumerate(sampled):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        pair_data = df[(df["center_id"] == center_id) & (df["meal_id"] == meal_id)].sort_values("week")
        ax.plot(pair_data["week"], pair_data["num_orders"])
        ax.set_title(f"c{center_id}·m{meal_id}")
        ax.set_xlabel("week")
        ax.set_ylabel("num_orders")

    for idx in range(n_sample, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out = out_dir / "fig2_timeseries_grid.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# US5 — analyse_group_demand + save_group_demand_tex
# ---------------------------------------------------------------------------

def analyse_group_demand(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group num_orders by center_type, cuisine, category; return dict of DataFrames."""
    result = {}
    for col in ("center_type", "cuisine", "category"):
        grouped = (
            df.groupby(col)["num_orders"]
            .agg(mean_orders="mean", std_orders="std", count="count")
            .reset_index()
            .rename(columns={col: "group_value"})
        )
        result[col] = grouped[["group_value", "mean_orders", "std_orders", "count"]]
    return result


def save_group_demand_tex(group_stats: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Write group_demand.tex booktabs tabular from group_stats dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "group_demand.tex"

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Grouping & Value & Mean Orders & Std Orders & Count \\",
        r"\midrule",
    ]

    for section_key in ("center_type", "cuisine"):
        if section_key not in group_stats:
            continue
        df_sec = group_stats[section_key]
        label = section_key.replace("_", " ").title()
        for _, row in df_sec.iterrows():
            lines.append(
                f"{label} & {row['group_value']} & {row['mean_orders']:.2f} "
                f"& {row['std_orders']:.2f} & {int(row['count'])} \\\\"
            )
            label = ""  # only print section label on first row

    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# US6 — analyse_promotion_effect
# ---------------------------------------------------------------------------

def analyse_promotion_effect(df: pd.DataFrame) -> dict[str, dict]:
    """Compute mean num_orders and lift for emailer and homepage promotion flags."""
    flag_map = {
        "emailer": "emailer_for_promotion",
        "homepage": "homepage_featured",
    }
    result = {}
    for key, col in flag_map.items():
        unique_vals = df[col].unique()
        if len(unique_vals) < 2:
            raise ValueError(
                f"Column '{col}' has only one unique value ({unique_vals}). "
                "Cannot compute promotion effect."
            )
        mean_0 = float(df[df[col] == 0]["num_orders"].mean())
        mean_1 = float(df[df[col] == 1]["num_orders"].mean())
        result[key] = {
            "mean_0": mean_0,
            "mean_1": mean_1,
            "lift_pct": (mean_1 - mean_0) / mean_0 * 100,
            "diff": mean_1 - mean_0,
        }
    return result


# ---------------------------------------------------------------------------
# US7 — analyse_price_elasticity + plot_promotion_elasticity
# ---------------------------------------------------------------------------

def analyse_price_elasticity(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Compute group-level Pearson r between discount_rate and num_orders."""
    n_zero = (df["base_price"] == 0).sum()
    if n_zero > 0:
        warnings.warn(
            f"{n_zero} rows with base_price == 0 excluded from elasticity analysis.",
            UserWarning,
            stacklevel=2,
        )

    valid = df[df["base_price"] > 0].copy()
    valid["discount_rate"] = (valid["base_price"] - valid["checkout_price"]) / valid["base_price"]

    scatter_df = (
        valid.groupby(["center_id", "meal_id"])
        .agg(mean_discount_rate=("discount_rate", "mean"), mean_orders=("num_orders", "mean"))
        .reset_index()[["mean_discount_rate", "mean_orders"]]
    )

    if len(scatter_df) < 2:
        return 0.0, scatter_df

    r_matrix = np.corrcoef(scatter_df["mean_discount_rate"], scatter_df["mean_orders"])
    pearson_r = float(r_matrix[0, 1])
    return pearson_r, scatter_df


def plot_promotion_elasticity(
    promo_stats: dict[str, dict],
    scatter_df: pd.DataFrame,
    pearson_r: float,
    out_dir: Path,
) -> Path:
    """Save fig3_promotion_elasticity.pdf: promotion bars + elasticity scatter."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, label) in zip(axes[:2], [("emailer", "Emailer"), ("homepage", "Homepage")]):
        stats_d = promo_stats[key]
        bars = ax.bar(["No Promo", "Promo"], [stats_d["mean_0"], stats_d["mean_1"]])
        ax.set_title(f"{label} Effect (lift: {stats_d['lift_pct']:.1f}%)")
        ax.set_ylabel("Mean num_orders")

    axes[2].scatter(scatter_df["mean_discount_rate"], scatter_df["mean_orders"], alpha=0.6)
    axes[2].set_xlabel("Mean Discount Rate")
    axes[2].set_ylabel("Mean num_orders")
    axes[2].set_title(f"Price Elasticity (r = {pearson_r:.4f})")
    axes[2].annotate(f"r = {pearson_r:.4f}", xy=(0.05, 0.9), xycoords="axes fraction")

    fig.tight_layout()
    out = out_dir / "fig3_promotion_elasticity.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# US8 — plot_acf_analysis
# ---------------------------------------------------------------------------

def _compute_acf(series: np.ndarray, max_lags: int) -> tuple[np.ndarray, int]:
    """Compute normalised ACF for series up to max_lags."""
    n = len(series)
    lags = min(max_lags, n - 1)
    mean = series.mean()
    var = ((series - mean) ** 2).mean()
    acf_vals = np.ones(lags + 1)
    if var == 0:
        return acf_vals, lags
    for lag in range(1, lags + 1):
        acf_vals[lag] = ((series[lag:] - mean) * (series[:-lag] - mean)).mean() / var
    return acf_vals, lags


def plot_acf_analysis(df: pd.DataFrame, out_dir: Path, max_lags: int = 52) -> Path:
    """Select 3 representative pairs by percentile; plot and save ACF figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = df.groupby(["center_id", "meal_id"])["num_orders"].sum()
    pairs = totals.index.tolist()

    n = len(totals)
    if n < 3:
        selected_idx = list(range(n))
    else:
        p25_val, p50_val, p75_val = np.percentile(totals.values, [25, 50, 75])
        def nearest(target):
            return int(np.argmin(np.abs(totals.values - target)))
        selected_idx = list(dict.fromkeys([nearest(p25_val), nearest(p50_val), nearest(p75_val)]))

    n_plots = len(selected_idx)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4), squeeze=False)

    for plot_i, pair_i in enumerate(selected_idx):
        center_id, meal_id = pairs[pair_i]
        series = (
            df[(df["center_id"] == center_id) & (df["meal_id"] == meal_id)]
            .sort_values("week")["num_orders"]
            .values
            .astype(float)
        )
        acf_vals, lags = _compute_acf(series, max_lags)
        ax = axes[0][plot_i]
        lag_x = np.arange(lags + 1)
        ax.bar(lag_x, acf_vals, width=0.5)
        ci = 1.96 / np.sqrt(len(series))
        ax.axhline(ci, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"c{center_id}·m{meal_id}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")

    fig.tight_layout()
    out = out_dir / "fig_acf_pairs.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# US9 — save_dataset_stats_tex
# ---------------------------------------------------------------------------

def save_dataset_stats_tex(profile: dict, out_dir: Path) -> Path:
    """Write dataset_stats.tex booktabs tabular from profile dict."""
    required = ("shape", "unique_counts", "num_orders_stats", "history_lengths")
    for key in required:
        if key not in profile:
            raise KeyError(f"Required profile key missing: '{key}'")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = profile["shape"]
    uc = profile["unique_counts"]
    nos = profile["num_orders_stats"]
    hl = profile["history_lengths"]

    rows = [
        ("n\\_weeks", str(uc.get("week", "—"))),
        ("n\\_centers", str(uc.get("center_id", "—"))),
        ("n\\_meals", str(uc.get("meal_id", "—"))),
        ("n\\_pairs", str(s[0])),
        ("mean num\\_orders", f"{nos['mean']:.2f}"),
        ("std num\\_orders", f"{nos['std']:.2f}"),
        ("median num\\_orders", f"{nos['median']:.2f}"),
        ("p95 num\\_orders", f"{nos['p95']:.2f}"),
        ("min num\\_orders", f"{nos['min']:.0f}"),
        ("max num\\_orders", f"{nos['max']:.0f}"),
        ("zero\\_order\\_rate", f"{nos['zero_order_rate']:.4f}"),
        ("history\\_length min", str(hl["min"])),
        ("history\\_length median", f"{hl['median']:.1f}"),
        ("history\\_length max", str(hl["max"])),
    ]

    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for metric, val in rows:
        lines.append(f"{metric} & {val} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    out = out_dir / "dataset_stats.tex"
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# __main__ — full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_merged, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR

    print("Loading merged dataset …")
    df = load_merged(PROCESSED_DIR)
    print(f"  {len(df):,} rows loaded.")

    print("Profiling dataset …")
    profile = profile_dataset(df)
    print(f"  Shape: {profile['shape']}")

    print("Plotting order distribution …")
    p = plot_order_distribution(df, FIGURES_DIR)
    print(f"  Saved {p}")

    print("Plotting time-series grid …")
    p = plot_timeseries_grid(df, FIGURES_DIR)
    print(f"  Saved {p}")

    print("Analysing group demand …")
    group_stats = analyse_group_demand(df)
    p = save_group_demand_tex(group_stats, TABLES_DIR)
    print(f"  Saved {p}")

    print("Analysing promotion effect …")
    promo_stats = analyse_promotion_effect(df)
    print(f"  Emailer lift: {promo_stats['emailer']['lift_pct']:.2f}%")
    print(f"  Homepage lift: {promo_stats['homepage']['lift_pct']:.2f}%")

    print("Analysing price elasticity …")
    pearson_r, scatter_df = analyse_price_elasticity(df)
    print(f"  Pearson r: {pearson_r:.4f}")

    print("Plotting promotion & elasticity …")
    p = plot_promotion_elasticity(promo_stats, scatter_df, pearson_r, FIGURES_DIR)
    print(f"  Saved {p}")

    print("Plotting ACF analysis …")
    p = plot_acf_analysis(df, FIGURES_DIR)
    print(f"  Saved {p}")

    print("Saving dataset stats table …")
    p = save_dataset_stats_tex(profile, TABLES_DIR)
    print(f"  Saved {p}")

    print("EDA pipeline complete.")
