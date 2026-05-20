"""
Known-Groups Validation: Early vs. Late Session Fatigue

Strategy
--------
For each subject-session pair, we have time-windowed EEG rows (from all_windows.csv)
and a compound fatigue index built from PCA on [kss, rt_mean_ms, rt_std_ms,
global_ratio_slow_fast] (from pca_input_features.csv / subject_session_features.csv).

Steps:
  1. Load the compound index (PC1) by re-running PCA with the same feature set used
     in the notebook, so scores are reproducible.
  2. For each subject-session, bin the EEG windows into the first 20 % (early) and
     last 20 % (late) of recording time and compute the mean ratio_slow_fast for
     each bin.
  3. Merge the early/late EEG values with the compound index score.
  4. Across subjects, run a paired Wilcoxon signed-rank test:
       H0 : early fatigue ratio == late fatigue ratio  (within subjects)
  5. Separately, test whether the compound index (PC1) differs across sessions that
     are labelled "early" vs "late" in their session ordering per subject.
  6. Save a summary table and plots.

File layout expected (mirrors the main pipeline output):
    <base_dir>/tables/all_windows.csv
    <base_dir>/tables/pca_input_features.csv          (or subject_session_features.csv)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Features used to build the compound index — must match your notebook exactly
COMPOUND_INDEX_FEATURES = [
    "kss",
    "rt_mean_ms",
    "rt_std_ms",
    "global_ratio_slow_fast",
]

# Fraction of recording time counted as "early" and "late"
EARLY_FRAC = 0.20
LATE_FRAC = 0.20

DPI = 150


# ---------------------------------------------------------------------------
# Step 1 – Build compound fatigue index (PC1)
# ---------------------------------------------------------------------------

def build_compound_index(
    features_csv: Path,
    feature_cols: list[str],
    subject_col: str = "subject_id",
    session_col: str = "session_id",
) -> pd.DataFrame:
    """
    Reproduce the PCA compound index from the notebook.
    Returns a dataframe with subject_id, session_id, and PC1 (fatigue_index).
    The sign of PC1 is flipped when needed so that higher values = more fatigue
    (i.e. PC1 correlates positively with KSS when KSS is in the feature set).
    """
    df = pd.read_csv(features_csv)

    # Keep only rows that have at least one non-null feature
    df_feat = df[feature_cols].copy()
    valid_rows = df_feat.notna().any(axis=1)
    df = df[valid_rows].copy()

    X = df[feature_cols].values
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    pc1 = scores[:, 0]

    # Convention: positive PC1 should align with higher KSS (more tired).
    # If KSS is available, check correlation sign and flip if negative.
    if "kss" in feature_cols and "kss" in df.columns:
        kss_vals = imputer.transform(df[feature_cols].values)[:, feature_cols.index("kss")]
        if np.corrcoef(pc1, kss_vals)[0, 1] < 0:
            pc1 = -pc1

    result = df[[subject_col, session_col]].copy().reset_index(drop=True)
    result["fatigue_index"] = pc1
    result["subject_id"] = result["subject_id"].astype(str)
    result["session_id"] = result["session_id"].astype(str)
    return result


# ---------------------------------------------------------------------------
# Step 2 – Compute early / late EEG fatigue ratio per subject-session
# ---------------------------------------------------------------------------

def early_late_eeg(
    windows_csv: Path,
    early_frac: float = EARLY_FRAC,
    late_frac: float = LATE_FRAC,
    fatigue_col: str = "ratio_slow_fast",
) -> pd.DataFrame:
    """
    For each subject-session, split time windows into the first `early_frac`
    and last `late_frac` of the recording and return mean fatigue ratios.
    """
    df = pd.read_csv(windows_csv)
    df["subject_id"] = df["subject_id"].astype(str)
    df["session_id"] = df["session_id"].astype(str)

    rows = []
    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        grp = grp.sort_values("t_mid")
        t_min, t_max = grp["t_mid"].min(), grp["t_mid"].max()
        duration = t_max - t_min

        if duration <= 0:
            continue

        early_cutoff = t_min + early_frac * duration
        late_cutoff = t_max - late_frac * duration

        early_vals = grp.loc[grp["t_mid"] <= early_cutoff, fatigue_col].dropna()
        late_vals = grp.loc[grp["t_mid"] >= late_cutoff, fatigue_col].dropna()

        if early_vals.empty or late_vals.empty:
            continue

        rows.append(
            {
                "subject_id": subj,
                "session_id": sess,
                "early_fatigue_ratio": early_vals.mean(),
                "late_fatigue_ratio": late_vals.mean(),
                "n_early_windows": len(early_vals),
                "n_late_windows": len(late_vals),
                "early_cutoff_sec": early_cutoff,
                "late_cutoff_sec": late_cutoff,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 – Wilcoxon tests
# ---------------------------------------------------------------------------

def wilcoxon_early_vs_late(df: pd.DataFrame) -> dict:
    """
    Paired Wilcoxon signed-rank test: late_fatigue_ratio > early_fatigue_ratio.
    One pair per subject-session row (within-subject, within-session).
    """
    early = df["early_fatigue_ratio"].values
    late = df["late_fatigue_ratio"].values
    diffs = late - early

    n_pairs = len(diffs)
    n_increase = int((diffs > 0).sum())
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    if n_pairs < 2:
        return {
            "test": "Wilcoxon signed-rank (early vs late EEG ratio)",
            "n_pairs": n_pairs,
            "n_increased": n_increase,
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "statistic": np.nan,
            "p_value": np.nan,
            "note": "Not enough pairs for Wilcoxon test.",
        }

    # alternative='greater': late > early  →  fatigue increases over time
    try:
        stat, p = wilcoxon(late, early, alternative="greater", zero_method="wilcox")
    except ValueError as e:
        stat, p = np.nan, np.nan

    return {
        "test": "Wilcoxon signed-rank (early vs late EEG ratio)",
        "n_pairs": n_pairs,
        "n_increased": n_increase,
        "pct_increased": round(100 * n_increase / n_pairs, 1),
        "mean_diff": round(mean_diff, 4),
        "median_diff": round(median_diff, 4),
        "statistic": stat,
        "p_value": round(p, 4) if not np.isnan(p) else np.nan,
        "significant_alpha_0.05": bool(p < 0.05) if not np.isnan(p) else False,
    }


def wilcoxon_session_order(df_merged: pd.DataFrame) -> dict:
    """
    Across subjects: do later sessions (higher session number) have higher
    compound fatigue index than earlier sessions?

    Strategy: for each subject that has >= 2 sessions, rank sessions by
    session_id (assumed to be 1, 2, 3 …) and test whether fatigue_index
    increases from the first-ranked to the last-ranked session.
    """
    pairs_early, pairs_late = [], []

    for subj, grp in df_merged.groupby("subject_id"):
        grp = grp.copy()
        try:
            grp["sess_int"] = grp["session_id"].astype(int)
        except ValueError:
            grp["sess_int"] = grp["session_id"].rank(method="dense").astype(int)

        grp = grp.sort_values("sess_int")

        if len(grp) < 2:
            continue

        pairs_early.append(grp.iloc[0]["fatigue_index"])
        pairs_late.append(grp.iloc[-1]["fatigue_index"])

    n_pairs = len(pairs_early)
    if n_pairs < 2:
        return {
            "test": "Wilcoxon signed-rank (first vs last session compound index)",
            "n_subjects": n_pairs,
            "note": "Not enough subjects with >= 2 sessions.",
            "statistic": np.nan,
            "p_value": np.nan,
        }

    early_arr = np.array(pairs_early)
    late_arr = np.array(pairs_late)
    diffs = late_arr - early_arr
    n_increase = int((diffs > 0).sum())

    try:
        stat, p = wilcoxon(late_arr, early_arr, alternative="greater", zero_method="wilcox")
    except ValueError:
        stat, p = np.nan, np.nan

    return {
        "test": "Wilcoxon signed-rank (first vs last session compound index)",
        "n_subjects": n_pairs,
        "n_increased": n_increase,
        "pct_increased": round(100 * n_increase / n_pairs, 1),
        "mean_diff_index": round(float(np.mean(diffs)), 4),
        "median_diff_index": round(float(np.median(diffs)), 4),
        "statistic": stat,
        "p_value": round(p, 4) if not np.isnan(p) else np.nan,
        "significant_alpha_0.05": bool(p < 0.05) if not np.isnan(p) else False,
    }


# ---------------------------------------------------------------------------
# Step 4 – Plots
# ---------------------------------------------------------------------------

def plot_early_late_paired(df: pd.DataFrame, out_path: Path) -> None:
    """Paired line plot: each line = one subject-session."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in df.iterrows():
        color = "steelblue" if row["late_fatigue_ratio"] > row["early_fatigue_ratio"] else "tomato"
        ax.plot([0, 1], [row["early_fatigue_ratio"], row["late_fatigue_ratio"]],
                color=color, alpha=0.5, linewidth=1.2)

    # Group means
    ax.plot([0, 1],
            [df["early_fatigue_ratio"].mean(), df["late_fatigue_ratio"].mean()],
            color="black", linewidth=2.5, linestyle="--", label="Group mean", zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Early (first 20%)", "Late (last 20%)"], fontsize=12)
    ax.set_ylabel("(Theta + Alpha) / Beta", fontsize=11)
    ax.set_title("Within-Session Fatigue: Early vs Late Windows", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotation: % sessions where fatigue increased
    n_inc = (df["late_fatigue_ratio"] > df["early_fatigue_ratio"]).sum()
    pct = 100 * n_inc / len(df)
    ax.text(0.98, 0.02, f"{n_inc}/{len(df)} sessions ↑ ({pct:.0f}%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="steelblue")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def plot_compound_index_by_session(df_merged: pd.DataFrame, out_path: Path) -> None:
    """
    Box / swarm plot of compound fatigue index across sessions.
    Overlays individual subject trajectories.
    """
    try:
        df_merged["sess_int"] = df_merged["session_id"].astype(int)
    except (ValueError, TypeError):
        df_merged = df_merged.copy()
        df_merged["sess_int"] = df_merged["session_id"].rank(method="dense").astype(int)

    sessions = sorted(df_merged["sess_int"].unique())
    data_by_sess = [df_merged.loc[df_merged["sess_int"] == s, "fatigue_index"].values for s in sessions]
    sess_labels = [str(s) for s in sessions]

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(data_by_sess, positions=sessions, widths=0.35,
                    patch_artist=True, medianprops=dict(color="black", linewidth=2))

    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(sessions)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jitter individual points
    rng = np.random.default_rng(42)
    for i, (sess, vals) in enumerate(zip(sessions, data_by_sess)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(sess + jitter, vals, color="black", s=20, alpha=0.6, zorder=3)

    # Subject trajectories
    for subj, grp in df_merged.groupby("subject_id"):
        grp = grp.sort_values("sess_int")
        ax.plot(grp["sess_int"], grp["fatigue_index"],
                color="grey", alpha=0.3, linewidth=1, zorder=2)

    ax.set_xticks(sessions)
    ax.set_xticklabels([f"Session {s}" for s in sessions], fontsize=11)
    ax.set_ylabel("Compound Fatigue Index (PC1)", fontsize=11)
    ax.set_title("Compound Fatigue Index Across Sessions", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def plot_diff_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Histogram of (late - early) fatigue ratio differences."""
    diffs = df["late_fatigue_ratio"] - df["early_fatigue_ratio"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(diffs, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="No change")
    ax.axvline(diffs.mean(), color="tomato", linewidth=1.5, linestyle="-",
               label=f"Mean diff = {diffs.mean():.3f}")
    ax.set_xlabel("Late − Early Fatigue Ratio", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Within-Session Fatigue Change", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_validation(
    base_dir: Path,
    out_dir: Path,
    feature_cols: list[str] = COMPOUND_INDEX_FEATURES,
    early_frac: float = EARLY_FRAC,
    late_frac: float = LATE_FRAC,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    table_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # Locate input files
    windows_csv = base_dir / "tables" / "all_windows.csv"
    features_csv = base_dir / "tables" / "pca_input_features.csv"

    # Fallback: some runs save it as subject_session_features.csv
    if not features_csv.exists():
        features_csv = base_dir / "tables" / "subject_session_features.csv"

    if not windows_csv.exists():
        raise FileNotFoundError(f"Cannot find all_windows.csv in {base_dir / 'tables'}")
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Cannot find pca_input_features.csv or subject_session_features.csv in {base_dir / 'tables'}"
        )

    print("=" * 60)
    print("Known-Groups Validation")
    print("=" * 60)

    # --- Compound index ---
    print("\n[1/4] Building compound fatigue index (PCA on selected features)...")
    df_index = build_compound_index(features_csv, feature_cols)
    print(f"      {len(df_index)} subject-session pairs with a fatigue index.")
    df_index.to_csv(table_dir / "compound_fatigue_index.csv", index=False)

    # --- Early / late EEG windows ---
    print(f"\n[2/4] Computing early ({early_frac*100:.0f}%) / late ({late_frac*100:.0f}%) "
          f"EEG fatigue ratios from windowed data...")
    df_el = early_late_eeg(windows_csv, early_frac, late_frac)
    print(f"      {len(df_el)} subject-session pairs with sufficient window coverage.")
    df_el.to_csv(table_dir / "early_late_eeg.csv", index=False)

    # --- Merge ---
    df_merged = df_el.merge(df_index, on=["subject_id", "session_id"], how="inner")
    print(f"      {len(df_merged)} pairs after merging with compound index.")
    df_merged.to_csv(table_dir / "merged_validation.csv", index=False)

    if df_merged.empty:
        print("\nNo overlapping pairs found. Check subject_id / session_id formatting.")
        return

    # --- Wilcoxon test 1: within-session early vs late EEG ratio ---
    print("\n[3/4] Running Wilcoxon test: early vs late within-session EEG ratio...")
    result_within = wilcoxon_early_vs_late(df_merged)
    for k, v in result_within.items():
        print(f"      {k}: {v}")

    # --- Wilcoxon test 2: first vs last session compound index ---
    print("\n[4/4] Running Wilcoxon test: first vs last session compound fatigue index...")
    result_session = wilcoxon_session_order(df_merged)
    for k, v in result_session.items():
        print(f"      {k}: {v}")

    # --- Save stats table ---
    stats_df = pd.DataFrame([result_within, result_session])
    stats_df.to_csv(table_dir / "wilcoxon_results.csv", index=False)

    # --- Plots ---
    print("\nGenerating plots...")
    plot_early_late_paired(df_merged, plot_dir / "early_late_paired.png")
    plot_compound_index_by_session(df_merged, plot_dir / "compound_index_by_session.png")
    plot_diff_distribution(df_merged, plot_dir / "diff_distribution.png")

    print(f"\nDone. Results saved to: {out_dir}")
    print("  tables/compound_fatigue_index.csv    — PC1 per subject-session")
    print("  tables/early_late_eeg.csv            — early/late window means")
    print("  tables/merged_validation.csv         — joined table")
    print("  tables/wilcoxon_results.csv          — Wilcoxon test statistics")
    print("  plots/early_late_paired.png          — paired line plot")
    print("  plots/compound_index_by_session.png  — box plot across sessions")
    print("  plots/diff_distribution.png          — histogram of changes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Known-groups validation: early vs late session fatigue (Wilcoxon)"
    )
    ap.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base output directory from the main pipeline (contains 'tables/' subfolder)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to save validation outputs",
    )
    ap.add_argument(
        "--early_frac",
        type=float,
        default=EARLY_FRAC,
        help=f"Fraction of recording counted as 'early' (default: {EARLY_FRAC})",
    )
    ap.add_argument(
        "--late_frac",
        type=float,
        default=LATE_FRAC,
        help=f"Fraction of recording counted as 'late' (default: {LATE_FRAC})",
    )
    ap.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=COMPOUND_INDEX_FEATURES,
        help="Feature columns used to build the compound index (must exist in pca_input_features.csv)",
    )
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()

    run_validation(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        feature_cols=args.features,
        early_frac=args.early_frac,
        late_frac=args.late_frac,
    )