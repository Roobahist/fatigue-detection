"""
EEG Fatigue + KSS + Reaction Time Aggregation with PCA

This script extends the DROZY EDF fatigue analysis pipeline by:
1. Aggregating EEG-derived metrics to one row per subject-session pair.
2. Loading KSS self-reported tiredness values from a text file.
3. Parsing reaction-time CSV files and computing RT summary statistics.
4. Running PCA on the merged subject-session dataframe.
5. Saving tables and plots for downstream analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy.stats import iqr, linregress
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Config and Definitions
# ----------------------------

BANDS_HZ: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

ROIS: Dict[str, List[str]] = {
    "Frontal": ["Fz", "F3", "F4"],
    "Central": ["Cz", "C3", "C4"],
    "Posterior": ["Pz", "Oz", "P3", "P4", "O1", "O2"],
}

DEFAULT_EEG_PICKS = ["Fz", "Cz", "C3", "C4", "Pz", "Oz"]
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S.%f"


@dataclass(frozen=True)
class PipelineConfig:
    sfreq_expected: float = 512.0

    # Filtering
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freqs: Tuple[float, ...] = (50.0, 60.0)

    # Referencing
    reference: str = "average"

    # ICA
    ica_n_components: Optional[int] = None
    ica_method: str = "fastica"
    ica_random_state: int = 42
    ica_max_iter: int = 500

    # Band power windows
    window_sec: float = 4.0
    overlap: float = 0.5

    # PSD estimation
    psd_fmin: float = 0.5
    psd_fmax: float = 40.0
    psd_n_fft: int = 2048

    # KSS parsing
    kss_subject_start_index: int = 1

    # Text reading fallback encodings
    fallback_encodings: Tuple[str, ...] = (
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
        "utf-16",
    )

    # Output
    dpi: int = 150


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_subject_session(stem: str) -> Tuple[str, str]:
    parts = stem.split("-")
    if len(parts) < 2:
        return stem, "1"
    subject_id = "-".join(parts[:-1])
    session_id = parts[-1]
    return subject_id, session_id


# ----------------------------
# Processing Components
# ----------------------------

RawLoader = Callable[[Path], mne.io.BaseRaw]
Preprocessor = Callable[[mne.io.BaseRaw, PipelineConfig], Tuple[mne.io.BaseRaw, Dict[str, object]]]
BandpowerComputer = Callable[[mne.io.BaseRaw, PipelineConfig, Sequence[str]], pd.DataFrame]


def load_raw_edf(path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return raw


def preprocess_with_plots(raw_in: mne.io.BaseRaw, cfg: PipelineConfig) -> Tuple[mne.io.BaseRaw, Dict[str, object]]:
    meta: Dict[str, object] = {}
    raw = raw_in.copy()

    # 1. Channel Selection & Typing
    target_types = {}
    for ch in raw.ch_names:
        if ch in DEFAULT_EEG_PICKS:
            target_types[ch] = "eeg"
        elif "EOG" in ch.upper():
            target_types[ch] = "eog"
        elif "ECG" in ch.upper():
            target_types[ch] = "ecg"

    if target_types:
        raw.set_channel_types(target_types, on_unit_change="ignore")

    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True)
    raw.pick(picks)

    # 2. Montage
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

    # 3. Filtering
    if cfg.notch_freqs:
        raw.notch_filter(cfg.notch_freqs, verbose="ERROR")
    raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")

    # 4. Re-referencing
    raw.set_eeg_reference(cfg.reference, verbose="ERROR")

    # 5. ICA
    raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")

    ica = ICA(
        n_components=cfg.ica_n_components,
        method=cfg.ica_method,
        random_state=cfg.ica_random_state,
        max_iter=cfg.ica_max_iter,
    )
    ica.fit(raw_ica_fit, verbose="ERROR")

    eog_inds, ecg_inds = [], []

    if "eog" in raw:
        eog_inds, _ = ica.find_bads_eog(raw, verbose="ERROR")
    if "ecg" in raw:
        ecg_inds, _ = ica.find_bads_ecg(raw, verbose="ERROR")

    ica.exclude = list(set(eog_inds + ecg_inds))

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose="ERROR")

    meta["ica"] = ica
    meta["n_components_removed"] = len(ica.exclude)

    return raw_clean, meta


def compute_bandpower_windowed(
    raw: mne.io.BaseRaw,
    cfg: PipelineConfig,
    eeg_channels: Sequence[str],
) -> pd.DataFrame:
    valid_chs = [ch for ch in eeg_channels if ch in raw.ch_names]
    if not valid_chs:
        return pd.DataFrame()

    sfreq = raw.info["sfreq"]
    n_fft = min(cfg.psd_n_fft, int(cfg.window_sec * sfreq))

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=cfg.window_sec,
        overlap=cfg.window_sec * cfg.overlap,
        verbose="ERROR",
    )

    psds, freqs = epochs.compute_psd(
        method="welch",
        fmin=cfg.psd_fmin,
        fmax=cfg.psd_fmax,
        n_fft=n_fft,
    ).get_data(return_freqs=True)

    rows = []

    for epoch_idx in range(psds.shape[0]):
        t_start = epoch_idx * (cfg.window_sec * (1 - cfg.overlap))
        t_mid = t_start + (cfg.window_sec / 2)

        for ch_idx, ch_name in enumerate(valid_chs):
            psd_ch = psds[epoch_idx, ch_idx, :]

            powers = {}
            total_p = 0.0

            for band, (f_low, f_high) in BANDS_HZ.items():
                mask = (freqs >= f_low) & (freqs <= f_high)
                p = np.trapz(psd_ch[mask], freqs[mask])
                powers[band] = p
                total_p += p

            roi_name = "Other"
            for r_name, r_chs in ROIS.items():
                if ch_name in r_chs:
                    roi_name = r_name
                    break

            row = {
                "t_mid": t_mid,
                "channel": ch_name,
                "roi": roi_name,
                "total_power": total_p,
            }

            for band, p in powers.items():
                row[f"abs_{band}"] = p
                row[f"rel_{band}"] = p / total_p if total_p > 1e-20 else 0.0
                row[f"log_{band}"] = np.log10(p + 1e-20)

            row["ratio_theta_beta"] = powers["theta"] / powers["beta"] if powers["beta"] > 0 else np.nan
            row["ratio_slow_fast"] = (
                (powers["theta"] + powers["alpha"]) / powers["beta"]
                if powers["beta"] > 0
                else np.nan
            )

            rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# KSS Parsing
# ----------------------------

def load_kss_table(kss_txt_path: Path, subject_start_index: int = 1) -> pd.DataFrame:
    """
    Expects one row per subject and 3 values per row corresponding to sessions 1, 2, 3.

    Supports whitespace, comma, or semicolon separated values.
    """
    rows = []

    with kss_txt_path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f, start=subject_start_index):
            line = line.strip()
            if not line:
                continue

            values = [x for x in line.replace(",", " ").replace(";", " ").split() if x]
            if len(values) != 3:
                raise ValueError(
                    f"KSS row for subject {row_idx} must contain exactly 3 values, got {len(values)}: {line}"
                )

            for sess_idx, value in enumerate(values, start=1):
                rows.append(
                    {
                        "subject_id": str(row_idx),
                        "session_id": str(sess_idx),
                        "kss": float(value),
                    }
                )

    return pd.DataFrame(rows)


# ----------------------------
# Reaction Time Parsing
# ----------------------------

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FMT)


def read_text_file_with_fallbacks(path: Path, encodings: Sequence[str]) -> str:
    last_error: Optional[Exception] = None

    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Could not decode file: {path}")


def split_line_with_fallback_delimiters(line: str) -> List[str]:
    for delimiter in (";", ",", "\t"):
        parts = [part.strip() for part in line.split(delimiter)]
        if len(parts) >= 2:
            return parts
    return [line.strip()]


def compute_reaction_times_from_file(csv_path: Path, cfg: PipelineConfig) -> List[float]:
    """
    The first line is session start timestamp and is ignored for RT computation.
    Every subsequent line has:
        stimulus_timestamp;response_timestamp
    Returns reaction times in milliseconds.
    """
    reaction_times_ms: List[float] = []

    text = read_text_file_with_fallbacks(csv_path, cfg.fallback_encodings)
    rows: List[List[str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(split_line_with_fallback_delimiters(line))

    if not rows:
        return reaction_times_ms

    # First row is session start timestamp and is ignored
    for row in rows[1:]:
        if len(row) < 2:
            continue

        try:
            stim_ts = parse_timestamp(row[0])
            resp_ts = parse_timestamp(row[1])
        except ValueError:
            continue

        rt_ms = (resp_ts - stim_ts).total_seconds() * 1000.0
        if rt_ms >= 0:
            reaction_times_ms.append(rt_ms)

    return reaction_times_ms


def summarize_reaction_times(reaction_times_ms: Sequence[float]) -> Dict[str, float]:
    if len(reaction_times_ms) == 0:
        return {
            "rt_n": 0,
            "rt_mean_ms": np.nan,
            "rt_std_ms": np.nan,
            "rt_median_ms": np.nan,
            "rt_iqr_ms": np.nan,
        }

    arr = np.asarray(reaction_times_ms, dtype=float)
    return {
        "rt_n": int(arr.size),
        "rt_mean_ms": float(np.mean(arr)),
        "rt_std_ms": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "rt_median_ms": float(np.median(arr)),
        "rt_iqr_ms": float(iqr(arr)),
    }


def load_reaction_time_table(rt_dir: Path, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []

    for csv_path in sorted(rt_dir.glob("*.csv")):
        subj, sess = parse_subject_session(csv_path.stem)

        try:
            rt_values = compute_reaction_times_from_file(csv_path, cfg)
            summary = summarize_reaction_times(rt_values)
        except Exception as e:
            print(f"  Error reading reaction-time file {csv_path.name}: {e}")
            summary = {
                "rt_n": 0,
                "rt_mean_ms": np.nan,
                "rt_std_ms": np.nan,
                "rt_median_ms": np.nan,
                "rt_iqr_ms": np.nan,
            }

        row = {
            "subject_id": str(subj),
            "session_id": str(sess),
        }
        row.update(summary)
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Aggregation
# ----------------------------

def build_session_level_eeg_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per subject-session pair.
    Columns contain ROI-specific aggregated EEG metrics.
    """
    metric_cols = [
        "rel_theta",
        "rel_alpha",
        "rel_beta",
        "ratio_theta_beta",
        "ratio_slow_fast",
        "total_power",
    ]

    agg_dict = {col: "mean" for col in metric_cols}

    grouped = (
        master_df.groupby(["subject_id", "session_id", "roi"])
        .agg(agg_dict)
        .reset_index()
    )

    slope_rows = []
    for (subj, sess, roi), group in master_df.groupby(["subject_id", "session_id", "roi"]):
        if group["t_mid"].nunique() < 2:
            slope = np.nan
        else:
            slope = linregress(group["t_mid"], group["ratio_slow_fast"]).slope

        slope_rows.append(
            {
                "subject_id": subj,
                "session_id": sess,
                "roi": roi,
                "fatigue_slope": slope,
            }
        )

    df_slopes = pd.DataFrame(slope_rows)

    grouped = grouped.merge(
        df_slopes,
        on=["subject_id", "session_id", "roi"],
        how="left",
    )

    wide = grouped.pivot_table(
        index=["subject_id", "session_id"],
        columns="roi",
        values=[
            "rel_theta",
            "rel_alpha",
            "rel_beta",
            "ratio_theta_beta",
            "ratio_slow_fast",
            "total_power",
            "fatigue_slope",
        ],
    )

    wide.columns = [f"{roi}_{metric}" for metric, roi in wide.columns]
    wide = wide.reset_index()

    global_df = (
        master_df.groupby(["subject_id", "session_id"])
        .agg(
            global_rel_theta=("rel_theta", "mean"),
            global_rel_alpha=("rel_alpha", "mean"),
            global_rel_beta=("rel_beta", "mean"),
            global_ratio_theta_beta=("ratio_theta_beta", "mean"),
            global_ratio_slow_fast=("ratio_slow_fast", "mean"),
            global_total_power=("total_power", "mean"),
        )
        .reset_index()
    )

    session_df = wide.merge(global_df, on=["subject_id", "session_id"], how="left")
    return session_df


# ----------------------------
# PCA
# ----------------------------

def prepare_pca_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    feature_df = df.drop(columns=["subject_id", "session_id"], errors="ignore").copy()

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    X = feature_df[numeric_cols].copy()

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    return X, X_scaled, numeric_cols


def run_pca(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA]:
    X, X_scaled, feature_names = prepare_pca_matrix(df)

    n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    df_pca_input = pd.concat(
        [
            df[["subject_id", "session_id"]].reset_index(drop=True),
            X.reset_index(drop=True),
        ],
        axis=1,
    )

    score_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    df_scores = pd.DataFrame(scores, columns=score_cols)
    df_scores.insert(0, "session_id", df["session_id"].values)
    df_scores.insert(0, "subject_id", df["subject_id"].values)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=score_cols,
    ).reset_index().rename(columns={"index": "feature"})

    return df_pca_input, df_scores, loadings, pca


# ----------------------------
# Plotting Functions
# ----------------------------

def plot_session_comparison_relative(
    df_sess: pd.DataFrame,
    subject_id: str,
    out_path: Path,
    cfg: PipelineConfig,
) -> None:
    df_agg = df_sess.groupby(["session_id", "roi"]).mean(numeric_only=True).reset_index()

    try:
        df_agg["sess_int"] = df_agg["session_id"].astype(int)
        df_agg = df_agg.sort_values("sess_int")
    except Exception:
        df_agg = df_agg.sort_values("session_id")

    rois = ["Frontal", "Posterior", "Central"]
    bands_of_interest = ["rel_theta", "rel_alpha", "ratio_slow_fast"]

    fig, axes = plt.subplots(len(bands_of_interest), len(rois), figsize=(12, 10), sharex=True)

    for r_idx, roi in enumerate(rois):
        subset = df_agg[df_agg["roi"] == roi]
        if subset.empty:
            continue

        sessions = subset["session_id"].astype(str).tolist()

        for b_idx, band in enumerate(bands_of_interest):
            ax = axes[b_idx, r_idx]
            vals = subset[band].values

            ax.plot(sessions, vals, marker="o", linestyle="-", linewidth=2)

            if len(vals) > 1:
                x_nums = np.arange(len(vals))
                z = np.polyfit(x_nums, vals, 1)
                p = np.poly1d(z)
                ax.plot(sessions, p(x_nums), "r--", alpha=0.5, label="Trend")

            if b_idx == 0:
                ax.set_title(f"{roi} ROI")
            if r_idx == 0:
                ax.set_ylabel(band)

            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Subject {subject_id}: Relative Power & Fatigue Ratios", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


def plot_time_on_task(
    df_file: pd.DataFrame,
    file_name: str,
    out_path: Path,
    cfg: PipelineConfig,
) -> None:
    df_agg = df_file.groupby("t_mid")[["ratio_slow_fast", "rel_theta"]].mean().reset_index()

    if len(df_agg) < 2:
        return

    dt = df_agg["t_mid"].iloc[1] - df_agg["t_mid"].iloc[0]
    window_pts = max(1, int(60 / dt))
    df_agg["smooth_ratio"] = df_agg["ratio_slow_fast"].rolling(window=window_pts, center=True).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_agg["t_mid"], df_agg["ratio_slow_fast"], alpha=0.1, s=10)
    ax.plot(df_agg["t_mid"], df_agg["smooth_ratio"], linewidth=2, label="(Theta+Alpha)/Beta (1min avg)")

    valid = df_agg.dropna()
    if len(valid) > 2:
        res = linregress(valid["t_mid"], valid["ratio_slow_fast"])
        ax.plot(
            valid["t_mid"],
            res.intercept + res.slope * valid["t_mid"],
            "k--",
            label=f"Slope: {res.slope * 60:.4f} / min",
        )

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Fatigue Ratio")
    ax.set_title(f"{file_name} | Within-Session Fatigue Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


def plot_pca_explained_variance(pca: PCA, out_path: Path, cfg: PipelineConfig) -> None:
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    x = np.arange(1, len(explained) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, explained, marker="o", label="Explained variance ratio")
    ax.plot(x, cumulative, marker="s", label="Cumulative explained variance")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("PCA Explained Variance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


def plot_pca_scores(df_scores: pd.DataFrame, out_path: Path, cfg: PipelineConfig) -> None:
    if "PC1" not in df_scores.columns or "PC2" not in df_scores.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df_scores.iterrows():
        ax.scatter(row["PC1"], row["PC2"], s=40)
        ax.text(
            row["PC1"],
            row["PC2"],
            f"{row['subject_id']}-{row['session_id']}",
            fontsize=8,
            alpha=0.8,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Score Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


def plot_pca_loadings_bars(loadings: pd.DataFrame, out_dir: Path, cfg: PipelineConfig, n_components: int = 2) -> None:
    ensure_dir(out_dir)

    for i in range(1, n_components + 1):
        pc_col = f"PC{i}"
        if pc_col not in loadings.columns:
            continue

        df_plot = loadings[["feature", pc_col]].copy()
        df_plot["abs_loading"] = df_plot[pc_col].abs()
        df_plot = df_plot.sort_values("abs_loading", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df_plot["feature"], df_plot[pc_col])
        ax.set_title(f"{pc_col} Feature Loadings")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Coefficient")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"{pc_col}_loadings.png", dpi=cfg.dpi)
        plt.close()


# ----------------------------
# Orchestrator
# ----------------------------

def analyze_directory(
    edf_dir: Path,
    rt_dir: Path,
    kss_txt_path: Path,
    out_dir: Path,
    cfg: PipelineConfig,
) -> None:
    ensure_dir(out_dir)

    edf_files = sorted(list(edf_dir.glob("*.edf")))
    if not edf_files:
        print(f"No EDF files found in {edf_dir}")
        return

    all_data = []

    for fpath in edf_files:
        subj, sess = parse_subject_session(fpath.stem)
        print(f"Processing: {subj} | Session {sess}")

        try:
            raw = load_raw_edf(fpath)
        except Exception as e:
            print(f"  Error loading {fpath.name}: {e}")
            continue

        try:
            raw_clean, meta = preprocess_with_plots(raw, cfg)
        except Exception as e:
            print(f"  Error preprocessing {fpath.name}: {e}")
            continue

        df_win = compute_bandpower_windowed(raw_clean, cfg, DEFAULT_EEG_PICKS)

        if df_win.empty:
            print(f"  No EEG data extracted for {fpath.name}")
            continue

        df_win["subject_id"] = str(subj)
        df_win["session_id"] = str(sess)
        df_win["file"] = fpath.name

        file_plot_dir = out_dir / "time_on_task"
        ensure_dir(file_plot_dir)
        plot_time_on_task(df_win, fpath.stem, file_plot_dir / f"{fpath.stem}_fatigue.png", cfg)

        all_data.append(df_win)

    if not all_data:
        print("No data processed.")
        return

    master_df = pd.concat(all_data, ignore_index=True)

    table_dir = out_dir / "tables"
    ensure_dir(table_dir)
    master_df.to_csv(table_dir / "all_windows.csv", index=False)

    # Subject-session EEG table
    df_eeg_session = build_session_level_eeg_table(master_df)
    df_eeg_session.to_csv(table_dir / "session_level_eeg_metrics.csv", index=False)

    # KSS
    df_kss = load_kss_table(
        kss_txt_path,
        subject_start_index=cfg.kss_subject_start_index,
    )
    df_kss.to_csv(table_dir / "kss_table.csv", index=False)

    # Reaction Times
    df_rt = load_reaction_time_table(rt_dir, cfg)
    df_rt.to_csv(table_dir / "reaction_time_summary.csv", index=False)

    # Merge everything
    df_final = df_eeg_session.merge(df_kss, on=["subject_id", "session_id"], how="left")
    df_final = df_final.merge(df_rt, on=["subject_id", "session_id"], how="left")

    # Save final merged dataframe
    df_final.to_csv(table_dir / "subject_session_features.csv", index=False)

    # Subject plots
    subj_plot_dir = out_dir / "subject_trends"
    ensure_dir(subj_plot_dir)
    for subj, df_subj in master_df.groupby("subject_id"):
        plot_session_comparison_relative(df_subj, str(subj), subj_plot_dir / f"{subj}_trends.png", cfg)

    # PCA
    pca_dir = out_dir / "pca"
    ensure_dir(pca_dir)

    df_pca_input, df_scores, df_loadings, pca = run_pca(df_final)

    df_pca_input.to_csv(table_dir / "pca_input_features.csv", index=False)
    df_scores.to_csv(table_dir / "pca_scores.csv", index=False)
    df_loadings.to_csv(table_dir / "pca_loadings.csv", index=False)

    pc_cols = [col for col in ["PC1", "PC2"] if col in df_scores.columns]
    df_final_with_pcs = df_final.merge(
        df_scores[["subject_id", "session_id"] + pc_cols],
        on=["subject_id", "session_id"],
        how="left",
    )
    df_final_with_pcs.to_csv(table_dir / "subject_session_features_with_pca.csv", index=False)

    plot_pca_explained_variance(pca, pca_dir / "explained_variance.png", cfg)
    plot_pca_scores(df_scores, pca_dir / "pca_scores_pc1_pc2.png", cfg)
    plot_pca_loadings_bars(df_loadings, pca_dir / "loadings", cfg, n_components=2)

    print(f"Analysis Complete. Results saved to {out_dir}")


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EEG + KSS + Reaction Time PCA Analysis")
    ap.add_argument("--edf_dir", type=str, required=True, help="Path to EDF files")
    ap.add_argument("--rt_dir", type=str, required=True, help="Path to reaction-time CSV files")
    ap.add_argument("--kss_txt", type=str, required=True, help="Path to KSS text file")
    ap.add_argument("--out_dir", type=str, required=True, help="Path to output folder")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()

    cfg = PipelineConfig()

    analyze_directory(
        edf_dir=Path(args.edf_dir),
        rt_dir=Path(args.rt_dir),
        kss_txt_path=Path(args.kss_txt),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )