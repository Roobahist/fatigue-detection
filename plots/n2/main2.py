"""
Within-Session Fatigue Trajectory Analysis
===========================================

Goal
----
Test whether a compound fatigue index *increases during* a session for each subject.

Design
------
1. Time-align EEG windows and RT trials onto a shared time axis.
   - EEG windows already have a `t_mid` column (seconds from recording start).
   - RT trials are parsed from the raw CSV files (stimulus + response timestamps).
     Each trial's elapsed time is computed relative to the session start timestamp
     (first line of the RT CSV), then assigned to the nearest EEG time bin.

2. For every EEG time bin that also has RT data nearby, compute a feature vector:
       [ratio_slow_fast, rel_theta, rt_value]
   where rt_value is the RT (ms) of the trial(s) falling in that bin (mean if > 1).

3. Run PCA across all bins × all subject-sessions to get a single within-session
   fatigue score per time bin (PC1, sign-corrected so higher = more fatigue).

4. KSS is constant within a session but differs between sessions.
   It is added back as a session-level offset AFTER the within-session PCA:
       composite_index(t) = PC1_within(t) + w_kss * z_kss
   where w_kss is PC1's loading on KSS from a *session-level* PCA (same weight
   used in the original compound index), and z_kss is the z-scored KSS value.
   This lets KSS shift the absolute level between sessions without compressing
   within-session variance.

5. Trajectory tests (per subject-session, then pooled):
   a. Bin test  — split bins into early (first 20 %) and late (last 20 %),
                  compare means with a paired Wilcoxon signed-rank test across
                  all subject-sessions.
   b. Slope test — fit a linear trend to the composite index over time for each
                   subject-session; test whether the distribution of slopes is
                   significantly > 0 with a one-sample Wilcoxon signed-rank test.

Outputs
-------
tables/
    time_bins.csv                   — per-bin features (EEG + RT, aligned)
    within_session_pca_scores.csv   — PC1 per bin
    composite_index_timeseries.csv  — PC1 + KSS offset, per bin
    per_session_slopes.csv          — slope + early/late means per subject-session
    wilcoxon_bin_test.csv           — early vs late Wilcoxon result
    wilcoxon_slope_test.csv         — slope > 0 Wilcoxon result

plots/
    <subj>_<sess>_trajectory.png    — individual session trajectory
    group_early_late.png            — paired plot across all sessions
    slope_distribution.png          — histogram of per-session slopes
    group_mean_trajectory.png       — grand mean ± SD trajectory
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, wilcoxon
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants — edit to match your feature choices
# ---------------------------------------------------------------------------

# Features used in the original session-level compound index (to extract KSS weight)
SESSION_LEVEL_FEATURES = ["kss", "rt_mean_ms", "rt_std_ms", "global_ratio_slow_fast"]

# Features used for the *within-session* (time-varying) PCA
WITHIN_SESSION_EEG_FEATURES = ["ratio_slow_fast", "rel_theta", "rel_alpha"]

# Maximum gap (seconds) allowed between an RT trial time and an EEG window centre
# for the trial to be assigned to that window
RT_MATCH_TOLERANCE_SEC = 30.0

# Early / late fractions
EARLY_FRAC = 0.20
LATE_FRAC  = 0.20

TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S.%f"
DPI = 150


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FMT)


def split_line(line: str) -> List[str]:
    for delim in (";", ",", "\t"):
        parts = [p.strip() for p in line.split(delim)]
        if len(parts) >= 2:
            return parts
    return [line.strip()]


def read_text_safe(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {path}")


# ---------------------------------------------------------------------------
# RT parsing — returns (session_start_dt, list of (elapsed_sec, rt_ms))
# ---------------------------------------------------------------------------

def parse_rt_file(csv_path: Path) -> Tuple[Optional[datetime], List[Tuple[float, float]]]:
    """
    Line 0  : session start timestamp (ignored for RT, used as time origin)
    Lines 1+: stimulus_timestamp ; response_timestamp

    Returns
    -------
    session_start : datetime or None
    trials        : list of (elapsed_seconds_of_stimulus, rt_ms)
                    elapsed is relative to session_start
    """
    text = read_text_safe(csv_path)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if not lines:
        return None, []

    # First line = session start
    try:
        session_start = parse_timestamp(lines[0].split(";")[0].split(",")[0].strip())
    except ValueError:
        session_start = None

    trials: List[Tuple[float, float]] = []
    for line in lines[1:]:
        parts = split_line(line)
        if len(parts) < 2:
            continue
        try:
            stim_dt = parse_timestamp(parts[0])
            resp_dt = parse_timestamp(parts[1])
        except ValueError:
            continue

        rt_ms = (resp_dt - stim_dt).total_seconds() * 1000.0
        if rt_ms < 0:
            continue

        elapsed_sec = (
            (stim_dt - session_start).total_seconds()
            if session_start is not None
            else 0.0
        )
        trials.append((elapsed_sec, rt_ms))

    return session_start, trials


# ---------------------------------------------------------------------------
# Align RT trials to EEG time bins
# ---------------------------------------------------------------------------

def assign_rt_to_eeg_bins(
    t_mids: np.ndarray,
    trials: List[Tuple[float, float]],
    tolerance_sec: float = RT_MATCH_TOLERANCE_SEC,
) -> np.ndarray:
    """
    For each EEG bin centre (t_mids), find RT trials within `tolerance_sec`
    and return the mean RT (ms).  NaN if no trial falls in the window.
    """
    rt_per_bin = np.full(len(t_mids), np.nan)

    if not trials:
        return rt_per_bin

    trial_times = np.array([t[0] for t in trials])
    trial_rts   = np.array([t[1] for t in trials])

    for i, t in enumerate(t_mids):
        mask = np.abs(trial_times - t) <= tolerance_sec
        if mask.any():
            rt_per_bin[i] = trial_rts[mask].mean()

    return rt_per_bin


# ---------------------------------------------------------------------------
# Build aligned time-bin table for all subject-sessions
# ---------------------------------------------------------------------------

def build_time_bin_table(
    windows_csv: Path,
    rt_dir: Path,
    eeg_features: List[str],
) -> pd.DataFrame:
    """
    Returns a dataframe with one row per (subject, session, time_bin).
    Columns: subject_id, session_id, t_mid, <eeg_features>, rt_ms
    """
    df_win = pd.read_csv(windows_csv)
    df_win["subject_id"] = df_win["subject_id"].astype(str)
    df_win["session_id"] = df_win["session_id"].astype(str)

    # Average EEG features across channels per time bin
    agg_dict = {f: "mean" for f in eeg_features if f in df_win.columns}
    df_binned = (
        df_win.groupby(["subject_id", "session_id", "t_mid"])
        .agg(agg_dict)
        .reset_index()
    )

    # Parse RT files — build lookup: (subject_id, session_id) -> trials
    rt_lookup: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for rt_file in sorted(rt_dir.glob("*.csv")):
        stem = rt_file.stem
        parts = stem.split("-")
        if len(parts) >= 2:
            subj = "-".join(parts[:-1])
            sess = parts[-1]
        else:
            subj, sess = stem, "1"
        _, trials = parse_rt_file(rt_file)
        rt_lookup[(str(subj), str(sess))] = trials

    # Assign RT values to EEG bins
    rt_col = np.full(len(df_binned), np.nan)

    for (subj, sess), grp in df_binned.groupby(["subject_id", "session_id"]):
        idx = grp.index
        t_mids = grp["t_mid"].values
        trials = rt_lookup.get((subj, sess), [])
        rt_vals = assign_rt_to_eeg_bins(t_mids, trials, RT_MATCH_TOLERANCE_SEC)
        rt_col[idx] = rt_vals

    df_binned["rt_ms"] = rt_col
    return df_binned


# ---------------------------------------------------------------------------
# Within-session PCA
# ---------------------------------------------------------------------------

def fit_within_session_pca(
    df_bins: pd.DataFrame,
    eeg_features: List[str],
) -> Tuple[pd.DataFrame, PCA, SimpleImputer, StandardScaler, List[str]]:
    """
    Fits PCA on [eeg_features + rt_ms] across all bins.
    Returns df_bins with added 'pc1_raw' column, plus fitted objects.
    Sign convention: PC1 correlates positively with ratio_slow_fast
    (higher = more fatigued).
    """
    feature_cols = [f for f in eeg_features if f in df_bins.columns] + ["rt_ms"]

    X = df_bins[feature_cols].values
    imputer = SimpleImputer(strategy="mean")
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    pc1 = scores[:, 0]

    # Sign correction: PC1 should increase with ratio_slow_fast
    if "ratio_slow_fast" in feature_cols:
        ref = df_bins["ratio_slow_fast"].fillna(df_bins["ratio_slow_fast"].mean()).values
        if np.corrcoef(pc1, ref)[0, 1] < 0:
            pc1 = -pc1

    df_out = df_bins.copy()
    df_out["pc1_raw"] = pc1
    return df_out, pca, imputer, scaler, feature_cols


# ---------------------------------------------------------------------------
# KSS offset
# ---------------------------------------------------------------------------

def compute_kss_weight(
    features_csv: Path,
    session_features: List[str],
) -> float:
    """
    From the session-level PCA (same one used in the original compound index),
    extract PC1's loading on KSS.  This becomes the weight for the KSS offset.
    Returns 0.0 if KSS is not available.
    """
    if "kss" not in session_features:
        return 0.0

    df = pd.read_csv(features_csv)
    cols = [c for c in session_features if c in df.columns]
    if "kss" not in cols:
        return 0.0

    X = df[cols].values
    imputer = SimpleImputer(strategy="mean")
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    # Sign-correct PC1 to align with KSS
    kss_idx = cols.index("kss")
    kss_z = X_scaled[:, kss_idx]
    pc1 = scores[:, 0]
    if np.corrcoef(pc1, kss_z)[0, 1] < 0:
        flip = -1.0
    else:
        flip = 1.0

    # Loading = component weight for KSS on PC1
    loading = float(flip * pca.components_[0, kss_idx])
    return loading


def add_kss_offset(
    df_bins: pd.DataFrame,
    features_csv: Path,
    session_features: List[str],
) -> pd.DataFrame:
    """
    Merges session-level KSS into the bin table, z-scores it across sessions,
    then adds w_kss * z_kss to pc1_raw to produce composite_index.
    """
    df_sess = pd.read_csv(features_csv)[["subject_id", "session_id", "kss"]].copy()
    df_sess["subject_id"] = df_sess["subject_id"].astype(str)
    df_sess["session_id"] = df_sess["session_id"].astype(str)

    # Z-score KSS across all sessions
    kss_mean = df_sess["kss"].mean()
    kss_std  = df_sess["kss"].std(ddof=1)
    df_sess["kss_z"] = (df_sess["kss"] - kss_mean) / (kss_std if kss_std > 0 else 1.0)

    df_out = df_bins.merge(df_sess[["subject_id", "session_id", "kss", "kss_z"]],
                           on=["subject_id", "session_id"], how="left")

    w_kss = compute_kss_weight(features_csv, session_features)
    print(f"      KSS loading on session-level PC1: {w_kss:.4f}")

    df_out["composite_index"] = df_out["pc1_raw"] + w_kss * df_out["kss_z"].fillna(0.0)
    return df_out


# ---------------------------------------------------------------------------
# Trajectory statistics
# ---------------------------------------------------------------------------

def compute_per_session_stats(
    df: pd.DataFrame,
    early_frac: float = EARLY_FRAC,
    late_frac:  float = LATE_FRAC,
    index_col:  str   = "composite_index",
) -> pd.DataFrame:
    """
    For each subject-session:
      - linear slope of index_col vs t_mid
      - mean index in early and late time fraction
      - number of bins
    """
    rows = []
    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        grp = grp.sort_values("t_mid").dropna(subset=[index_col])
        n = len(grp)
        if n < 3:
            continue

        t = grp["t_mid"].values
        y = grp[index_col].values

        # Linear slope
        res = linregress(t, y)
        slope       = res.slope
        slope_pval  = res.pvalue
        r_squared   = res.rvalue ** 2

        # Early / late means
        t_min, t_max = t.min(), t.max()
        duration = t_max - t_min
        early_mask = t <= t_min + early_frac * duration
        late_mask  = t >= t_max - late_frac  * duration

        early_mean = float(y[early_mask].mean()) if early_mask.any() else np.nan
        late_mean  = float(y[late_mask].mean())  if late_mask.any()  else np.nan

        rows.append({
            "subject_id":   subj,
            "session_id":   sess,
            "n_bins":       n,
            "slope":        slope,
            "slope_per_min": slope * 60,   # more interpretable unit
            "slope_pval":   slope_pval,
            "r_squared":    r_squared,
            "early_mean":   early_mean,
            "late_mean":    late_mean,
            "early_late_diff": late_mean - early_mean if not np.isnan(early_mean + late_mean) else np.nan,
            "kss":          grp["kss"].iloc[0] if "kss" in grp.columns else np.nan,
        })

    return pd.DataFrame(rows)


def run_wilcoxon_bin_test(df_stats: pd.DataFrame) -> dict:
    """Paired Wilcoxon: late_mean > early_mean across all subject-sessions."""
    valid = df_stats.dropna(subset=["early_mean", "late_mean"])
    early = valid["early_mean"].values
    late  = valid["late_mean"].values
    diffs = late - early
    n = len(diffs)

    n_increase = int((diffs > 0).sum())
    try:
        stat, p = wilcoxon(late, early, alternative="greater", zero_method="wilcox")
    except ValueError:
        stat, p = np.nan, np.nan

    return {
        "test":              "Wilcoxon bin test (late > early composite index)",
        "n_sessions":        n,
        "n_increased":       n_increase,
        "pct_increased":     round(100 * n_increase / n, 1) if n > 0 else np.nan,
        "mean_diff":         round(float(np.mean(diffs)), 4),
        "median_diff":       round(float(np.median(diffs)), 4),
        "statistic":         stat,
        "p_value":           round(p, 4) if not np.isnan(p) else np.nan,
        "significant_0.05":  bool(p < 0.05) if not np.isnan(p) else False,
    }


def run_wilcoxon_slope_test(df_stats: pd.DataFrame) -> dict:
    """One-sample Wilcoxon: distribution of per-session slopes > 0."""
    slopes = df_stats["slope"].dropna().values
    n = len(slopes)
    n_positive = int((slopes > 0).sum())

    # One-sample Wilcoxon = signed-rank test against 0
    try:
        stat, p = wilcoxon(slopes, alternative="greater", zero_method="wilcox")
    except ValueError:
        stat, p = np.nan, np.nan

    return {
        "test":              "Wilcoxon slope test (slope > 0)",
        "n_sessions":        n,
        "n_positive_slope":  n_positive,
        "pct_positive":      round(100 * n_positive / n, 1) if n > 0 else np.nan,
        "mean_slope_per_min":  round(float(np.mean(df_stats["slope_per_min"].dropna())), 6),
        "median_slope_per_min": round(float(np.median(df_stats["slope_per_min"].dropna())), 6),
        "statistic":         stat,
        "p_value":           round(p, 4) if not np.isnan(p) else np.nan,
        "significant_0.05":  bool(p < 0.05) if not np.isnan(p) else False,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_individual_trajectory(
    df_session: pd.DataFrame,
    subj: str,
    sess: str,
    slope: float,
    out_path: Path,
) -> None:
    grp = df_session.sort_values("t_mid")
    t   = grp["t_mid"].values / 60.0   # convert to minutes for readability
    y   = grp["composite_index"].values

    # Smoothed trend (rolling mean, ~2 min window)
    dt_min = np.diff(t).mean() if len(t) > 1 else 1.0
    win = max(1, int(2.0 / dt_min))
    y_smooth = pd.Series(y).rolling(win, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(t, y, s=12, alpha=0.35, color="steelblue", label="Bin value")
    ax.plot(t, y_smooth, linewidth=2, color="steelblue", label="2-min rolling mean")

    # Linear trend line
    if len(t) > 2 and not np.isnan(slope):
        res = linregress(t, y)
        ax.plot(t, res.intercept + res.slope * t, "k--", linewidth=1.5,
                label=f"Trend  slope={res.slope:.4f}/min  p={res.pvalue:.3f}")

    # Early / late shading
    dur = t[-1] - t[0]
    ax.axvspan(t[0],  t[0] + EARLY_FRAC * dur, alpha=0.08, color="green",  label="Early 20%")
    ax.axvspan(t[-1] - LATE_FRAC * dur, t[-1], alpha=0.08, color="tomato", label="Late 20%")

    kss_val = grp["kss"].iloc[0] if "kss" in grp.columns else "n/a"
    ax.set_title(f"Subject {subj} | Session {sess} | KSS={kss_val}", fontsize=12)
    ax.set_xlabel("Time in session (min)")
    ax.set_ylabel("Composite fatigue index")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def plot_group_early_late(df_stats: pd.DataFrame, out_path: Path) -> None:
    valid = df_stats.dropna(subset=["early_mean", "late_mean"])

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in valid.iterrows():
        color = "steelblue" if row["late_mean"] > row["early_mean"] else "tomato"
        ax.plot([0, 1], [row["early_mean"], row["late_mean"]],
                color=color, alpha=0.45, linewidth=1.2)

    ax.plot([0, 1],
            [valid["early_mean"].mean(), valid["late_mean"].mean()],
            color="black", linewidth=2.5, linestyle="--",
            label=f"Group mean  (Δ={valid['early_late_diff'].mean():.3f})", zorder=5)

    n_inc = (valid["late_mean"] > valid["early_mean"]).sum()
    ax.text(0.98, 0.02,
            f"{n_inc}/{len(valid)} sessions ↑ ({100*n_inc/len(valid):.0f}%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="steelblue")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Early (first {int(EARLY_FRAC*100)}%)",
                        f"Late (last {int(LATE_FRAC*100)}%)"], fontsize=12)
    ax.set_ylabel("Composite fatigue index", fontsize=11)
    ax.set_title("Within-Session Fatigue Change:\nEarly vs Late Bins", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def plot_slope_distribution(df_stats: pd.DataFrame, out_path: Path) -> None:
    slopes = df_stats["slope_per_min"].dropna().values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(slopes, bins=max(8, len(slopes) // 2),
            color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0,             color="black",  linewidth=1.5, linestyle="--", label="No change")
    ax.axvline(slopes.mean(), color="tomato", linewidth=1.5,
               label=f"Mean = {slopes.mean():.4f} / min")

    pct_pos = 100 * (slopes > 0).mean()
    ax.text(0.98, 0.97, f"{pct_pos:.0f}% positive slopes",
            transform=ax.transAxes, ha="right", va="top", fontsize=10)

    ax.set_xlabel("Slope of composite index (units / min)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Within-Session Fatigue Slopes", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def plot_group_mean_trajectory(df: pd.DataFrame, out_path: Path) -> None:
    """
    Normalise each session's time axis to [0, 1], bin into 20 quantile bins,
    then plot grand mean ± SD across sessions.
    """
    n_quantile_bins = 20
    bin_edges = np.linspace(0, 1, n_quantile_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    all_curves: List[np.ndarray] = []

    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        grp = grp.sort_values("t_mid").dropna(subset=["composite_index"])
        if len(grp) < n_quantile_bins:
            continue

        t = grp["t_mid"].values
        y = grp["composite_index"].values
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)

        curve = np.full(n_quantile_bins, np.nan)
        for b_idx in range(n_quantile_bins):
            mask = (t_norm >= bin_edges[b_idx]) & (t_norm < bin_edges[b_idx + 1])
            if mask.any():
                curve[b_idx] = y[mask].mean()

        if not np.isnan(curve).all():
            all_curves.append(curve)

    if not all_curves:
        return

    mat   = np.vstack(all_curves)
    mean  = np.nanmean(mat, axis=0)
    std   = np.nanstd(mat,  axis=0)
    count = np.sum(~np.isnan(mat), axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(bin_centres, mean - std, mean + std,
                    alpha=0.2, color="steelblue", label="±1 SD")
    ax.plot(bin_centres, mean, linewidth=2.5, color="steelblue",
            marker="o", markersize=4, label=f"Group mean  (n={len(all_curves)} sessions)")

    # Trend on the group mean
    valid_mask = ~np.isnan(mean)
    if valid_mask.sum() > 2:
        res = linregress(bin_centres[valid_mask], mean[valid_mask])
        ax.plot(bin_centres, res.intercept + res.slope * bin_centres,
                "k--", linewidth=1.5,
                label=f"Linear trend  slope={res.slope:.3f}/unit  p={res.pvalue:.3f}")

    ax.set_xlabel("Normalised time in session (0 = start, 1 = end)", fontsize=11)
    ax.set_ylabel("Composite fatigue index (mean ± SD)", fontsize=11)
    ax.set_title("Group Mean Within-Session Fatigue Trajectory", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_within_session_validation(
    base_dir:  Path,
    rt_dir:    Path,
    out_dir:   Path,
    eeg_features:     List[str] = WITHIN_SESSION_EEG_FEATURES,
    session_features: List[str] = SESSION_LEVEL_FEATURES,
    early_frac: float = EARLY_FRAC,
    late_frac:  float = LATE_FRAC,
) -> None:
    table_dir = out_dir / "tables"
    plot_dir  = out_dir / "plots"
    indiv_dir = plot_dir / "individual_sessions"
    for d in (table_dir, plot_dir, indiv_dir):
        ensure_dir(d)

    windows_csv  = base_dir / "tables" / "all_windows.csv"
    features_csv = base_dir / "tables" / "pca_input_features.csv"
    if not features_csv.exists():
        features_csv = base_dir / "tables" / "subject_session_features.csv"

    for p in (windows_csv, features_csv):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print("=" * 62)
    print("  Within-Session Fatigue Trajectory Analysis")
    print("=" * 62)

    # 1. Build time-aligned bins
    print("\n[1/5] Aligning EEG windows with RT trials...")
    df_bins = build_time_bin_table(windows_csv, rt_dir, eeg_features)
    missing_rt = df_bins["rt_ms"].isna().mean()
    print(f"      {len(df_bins)} total bins  |  RT coverage: {100*(1-missing_rt):.1f}% of bins")
    df_bins.to_csv(table_dir / "time_bins.csv", index=False)

    # 2. Within-session PCA
    print("\n[2/5] Fitting within-session PCA on EEG + RT features...")
    df_scored, pca_ws, _, _, ws_features = fit_within_session_pca(df_bins, eeg_features)
    ev = pca_ws.explained_variance_ratio_
    print(f"      Features: {ws_features}")
    print(f"      PC1 explains {ev[0]*100:.1f}% of within-session variance")
    df_scored[["subject_id", "session_id", "t_mid", "pc1_raw"]].to_csv(
        table_dir / "within_session_pca_scores.csv", index=False
    )

    # 3. Add KSS offset
    print("\n[3/5] Adding KSS session-level offset...")
    df_full = add_kss_offset(df_scored, features_csv, session_features)
    df_full.to_csv(table_dir / "composite_index_timeseries.csv", index=False)

    # 4. Per-session trajectory statistics
    print("\n[4/5] Computing per-session slopes and early/late means...")
    df_stats = compute_per_session_stats(df_full, early_frac, late_frac)
    df_stats.to_csv(table_dir / "per_session_slopes.csv", index=False)
    print(f"      {len(df_stats)} sessions with sufficient bins (≥3)")
    print(f"      Positive slopes: {(df_stats['slope']>0).sum()}/{len(df_stats)}")
    print(f"      Mean slope: {df_stats['slope_per_min'].mean():.5f} / min")

    # 5. Statistical tests
    print("\n[5/5] Running Wilcoxon tests...")
    result_bin   = run_wilcoxon_bin_test(df_stats)
    result_slope = run_wilcoxon_slope_test(df_stats)

    print("\n  --- Bin test (late > early) ---")
    for k, v in result_bin.items():
        print(f"      {k}: {v}")

    print("\n  --- Slope test (slope > 0) ---")
    for k, v in result_slope.items():
        print(f"      {k}: {v}")

    pd.DataFrame([result_bin]).to_csv(table_dir / "wilcoxon_bin_test.csv", index=False)
    pd.DataFrame([result_slope]).to_csv(table_dir / "wilcoxon_slope_test.csv", index=False)

    # Plots
    print("\nGenerating plots...")

    # Individual session trajectories
    slope_lookup = df_stats.set_index(["subject_id", "session_id"])["slope"].to_dict()
    for (subj, sess), grp in df_full.groupby(["subject_id", "session_id"]):
        slope = slope_lookup.get((subj, sess), np.nan)
        plot_individual_trajectory(
            grp, subj, sess, slope,
            indiv_dir / f"{subj}_{sess}_trajectory.png",
        )

    plot_group_early_late(df_stats,  plot_dir / "group_early_late.png")
    plot_slope_distribution(df_stats, plot_dir / "slope_distribution.png")
    plot_group_mean_trajectory(df_full, plot_dir / "group_mean_trajectory.png")

    print(f"\nDone. All outputs saved to: {out_dir}")
    print("\nKey files:")
    print("  tables/composite_index_timeseries.csv  — full time series per session")
    print("  tables/per_session_slopes.csv          — slope + early/late per session")
    print("  tables/wilcoxon_bin_test.csv            — early vs late result")
    print("  tables/wilcoxon_slope_test.csv          — slope > 0 result")
    print("  plots/group_mean_trajectory.png         — grand mean trajectory")
    print("  plots/individual_sessions/              — one plot per session")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Within-session fatigue trajectory analysis"
    )
    ap.add_argument("--base_dir", required=True,
                    help="Pipeline output dir (contains tables/all_windows.csv etc.)")
    ap.add_argument("--rt_dir",   required=True,
                    help="Directory containing raw RT CSV files")
    ap.add_argument("--out_dir",  required=True,
                    help="Where to write outputs")
    ap.add_argument("--early_frac", type=float, default=EARLY_FRAC,
                    help=f"Fraction of session counted as early (default {EARLY_FRAC})")
    ap.add_argument("--late_frac",  type=float, default=LATE_FRAC,
                    help=f"Fraction of session counted as late (default {LATE_FRAC})")
    ap.add_argument("--eeg_features", nargs="+", default=WITHIN_SESSION_EEG_FEATURES,
                    help="EEG feature columns to include in within-session PCA")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_within_session_validation(
        base_dir         = Path(args.base_dir),
        rt_dir           = Path(args.rt_dir),
        out_dir          = Path(args.out_dir),
        eeg_features     = args.eeg_features,
        early_frac       = args.early_frac,
        late_frac        = args.late_frac,
    )