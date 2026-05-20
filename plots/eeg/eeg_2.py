"""
EEG Fatigue Pattern Check for DROZY EDF files (Optimized for Relative Power & Ratios)

Adjustments:
1. Calculates Relative Power (Band / Total) to normalize global amplitude shifts.
2. Computes specific Fatigue Ratios (Theta/Beta, (Theta+Alpha)/Beta).
3. Groups channels into ROIs (Frontal, Central, Posterior).
4. Computes "Time-on-Task" slopes to detect fatigue accumulation within sessions.

Edits implemented:
- Time-on-task (per subject): always create 3 subplots for Session 1, 2, 3.
  If a session is missing, its subplot stays empty. Example: only Session 3 exists
  then subplot 1 and 2 are empty, subplot 3 shows Session 3.
- Subject trends: enforce consistent y-limits across ALL subjects for:
  * rel_theta: global min/max across all subjects (covers every point)
  * ratio_slow_fast: global min/max across all subjects (covers every point)
  * rel_alpha: global min/max across all subjects (covers every point)
  * ratio_theta_beta: global min/max across all subjects (covers every point)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy import signal
from scipy.stats import linregress


# ----------------------------
# Config and Definitions
# ----------------------------

BANDS_HZ: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Regions of Interest (ROIs)
ROIS: Dict[str, List[str]] = {
    "Frontal": ["Fz", "F3", "F4"],
    "Central": ["Cz", "C3", "C4"],
    "Posterior": ["Pz", "Oz", "P3", "P4", "O1", "O2"],
}

DEFAULT_EEG_PICKS = ["Fz", "Cz", "C3", "C4", "Pz", "Oz"]

# DROZY is commonly treated as 3 sessions. Keep fixed slots for plotting.
CANONICAL_SESSIONS: Tuple[str, str, str] = ("1", "2", "3")


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

    # Smoothing for time-on-task plots (seconds)
    smooth_sec: float = 60.0

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
    return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")


def _has_type(raw: mne.io.BaseRaw, ch_type: str) -> bool:
    try:
        return ch_type in set(raw.get_channel_types())
    except Exception:
        return False


def _find_bads_emg_combined(
    raw_for_ica: mne.io.BaseRaw,
    ica: ICA,
    emg_ch_name: str = "EMG",
    env_band_hz: Tuple[float, float] = (20.0, 40.0),
    corr_thresh: float = 0.30,
    hf_q: float = 0.90,
) -> List[int]:
    """
    EMG component detection using BOTH:
      1) Correlation of ICA sources with EMG envelope
      2) High-frequency dominance in ICA source PSD (20-40 vs 1-20)

    Returns component indices to exclude.
    """
    if emg_ch_name not in raw_for_ica.ch_names:
        return []

    sfreq = float(raw_for_ica.info["sfreq"])

    # EMG envelope
    try:
        emg_raw = raw_for_ica.copy().pick_channels([emg_ch_name])
        emg_raw.filter(l_freq=env_band_hz[0], h_freq=env_band_hz[1], verbose="ERROR")
        emg = emg_raw.get_data()[0]
        emg_env = np.abs(signal.hilbert(v))
    except Exception:
        return []

    # ICA sources
    try:
        src_raw = ica.get_sources(raw_for_ica)
        sources = src_raw.get_data()  # shape: (n_components, n_times)
    except Exception:
        return []

    if sources.ndim != 2 or sources.shape[1] != emg_env.shape[0]:
        return []

    # Method 1: correlation with EMG envelope
    emg_env_z = (emg_env - float(np.mean(emg_env))) / (float(np.std(emg_env)) + 1e-12)
    corrs = np.zeros(sources.shape[0], dtype=float)
    for k in range(sources.shape[0]):
        s = sources[k]
        s_z = (s - float(np.mean(s))) / (float(np.std(s)) + 1e-12)
        corrs[k] = float(np.abs(np.mean(s_z * emg_env_z)))

    # Method 2: high-frequency dominance in component PSD
    hf_ratios = np.zeros(sources.shape[0], dtype=float)
    nperseg = int(min(2048, max(256, sources.shape[1] // 4)))
    for k in range(sources.shape[0]):
        f, pxx = signal.welch(sources[k], fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
        hf_mask = (f >= 20.0) & (f <= 40.0)
        lf_mask = (f >= 1.0) & (f < 20.0)
        hf = float(np.trapz(pxx[hf_mask], f[hf_mask])) if np.any(hf_mask) else 0.0
        lf = float(np.trapz(pxx[lf_mask], f[lf_mask])) if np.any(lf_mask) else 0.0
        hf_ratios[k] = hf / (lf + 1e-20)

    hf_cut = float(np.quantile(hf_ratios[np.isfinite(hf_ratios)], hf_q)) if np.any(np.isfinite(hf_ratios)) else np.inf

    # Combined rule: correlated with EMG envelope AND HF-dominant
    bad = np.where((corrs >= corr_thresh) & (hf_ratios >= hf_cut))[0].tolist()
    return bad


def preprocess_with_plots(raw_in: mne.io.BaseRaw, cfg: PipelineConfig) -> Tuple[mne.io.BaseRaw, Dict[str, object]]:
    meta: Dict[str, object] = {}
    raw = raw_in.copy()

    # 1) Channel typing (best effort)
    target_types: Dict[str, str] = {}
    for ch in raw.ch_names:
        ch_up = ch.upper()
        if ch in DEFAULT_EEG_PICKS:
            target_types[ch] = "eeg"
        elif ch_up in {"EOG-V", "EOG-H"}:
            target_types[ch] = "eog"
        elif ch_up == "ECG":
            target_types[ch] = "ecg"
        elif ch_up == "EMG":
            target_types[ch] = "emg"

    if target_types:
        raw.set_channel_types(target_types, on_unit_change="ignore")

    # 2) Pick only channels we care about (EEG + potential artifact channels)
    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True, exclude=[])
    raw.pick(picks)

    # 3) Montage
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

    # 4) Filtering
    if cfg.notch_freqs:
        raw.notch_filter(cfg.notch_freqs, verbose="ERROR")
    raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")

    # 5) Re-reference (do before ICA)
    raw.set_eeg_reference(cfg.reference, verbose="ERROR")

    # 6) ICA fit on high-passed copy (1 Hz is common)
    raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")

    ica = ICA(
        n_components=cfg.ica_n_components,
        method=cfg.ica_method,
        random_state=cfg.ica_random_state,
        max_iter=cfg.ica_max_iter,
    )
    ica.fit(raw_ica_fit, verbose="ERROR")

    eog_inds: List[int] = []
    ecg_inds: List[int] = []

    if _has_type(raw, "eog"):
        try:
            eog_inds, _ = ica.find_bads_eog(raw, verbose="ERROR")
        except Exception:
            eog_inds = []

    if _has_type(raw, "ecg"):
        try:
            ecg_inds, _ = ica.find_bads_ecg(raw, verbose="ERROR")
        except Exception:
            ecg_inds = []

    emg_inds: List[int] = []
    if _has_type(raw, "emg") and ("EMG" in raw_ica_fit.ch_names):
        try:
            emg_inds = _find_bads_emg_combined(raw_ica_fit, ica, emg_ch_name="EMG")
        except Exception:
            emg_inds = []

    ica.exclude = list(set(eog_inds + ecg_inds + emg_inds))

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose="ERROR")

    meta["ica"] = ica
    meta["n_components_removed"] = len(ica.exclude)
    meta["eog_inds"] = eog_inds
    meta["ecg_inds"] = ecg_inds
    meta["emg_inds"] = emg_inds

    return raw_clean, meta


def compute_bandpower_windowed(
    raw: mne.io.BaseRaw,
    cfg: PipelineConfig,
    eeg_channels: Sequence[str],
) -> pd.DataFrame:
    """
    Computes Absolute Power, Relative Power, and Fatigue Ratios per window.
    """
    valid_chs = [ch for ch in eeg_channels if ch in raw.ch_names]
    if not valid_chs:
        return pd.DataFrame()

    sfreq = float(raw.info["sfreq"])
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

    hop = cfg.window_sec * (1.0 - cfg.overlap)
    rows: List[Dict[str, object]] = []

    for epoch_idx in range(psds.shape[0]):
        t_start = epoch_idx * hop
        t_mid = t_start + (cfg.window_sec / 2.0)

        for ch_idx, ch_name in enumerate(valid_chs):
            psd_ch = psds[epoch_idx, ch_idx, :]

            powers: Dict[str, float] = {}
            total_p = 0.0

            for band, (f_low, f_high) in BANDS_HZ.items():
                mask = (freqs >= f_low) & (freqs <= f_high)
                p = float(np.trapz(psd_ch[mask], freqs[mask]))
                powers[band] = p
                total_p += p

            roi_name = "Other"
            for r_name, r_chs in ROIS.items():
                if ch_name in r_chs:
                    roi_name = r_name
                    break

            row: Dict[str, object] = {
                "t_mid": float(t_mid),
                "channel": ch_name,
                "roi": roi_name,
                "total_power": float(total_p),
            }

            for band, p in powers.items():
                row[f"abs_{band}"] = float(p)
                row[f"rel_{band}"] = float(p / total_p) if total_p > 1e-20 else 0.0
                row[f"log_{band}"] = float(np.log10(p + 1e-20))

            beta = powers.get("beta", 0.0)
            theta = powers.get("theta", 0.0)
            alpha = powers.get("alpha", 0.0)

            row["ratio_theta_beta"] = float(theta / beta) if beta > 0 else np.nan
            row["ratio_slow_fast"] = float((theta + alpha) / beta) if beta > 0 else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Plotting Helpers
# ----------------------------

def _safe_min_max(vals: np.ndarray) -> Tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        pad = 0.1 if vmax == 0 else abs(vmax) * 0.1
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.05
    return vmin - pad, vmax + pad


def _session_sort_key(s: str) -> Tuple[int, str]:
    try:
        return int(s), s
    except Exception:
        return 10**9, s


def plot_session_comparison_relative(
    df_subj: pd.DataFrame,
    subject_id: str,
    out_path: Path,
    cfg: PipelineConfig,
    rel_theta_ylim: Tuple[float, float],
    ratio_slow_fast_ylim: Tuple[float, float],
    rel_alpha_ylim: Tuple[float, float],
    ratio_theta_beta_ylim: Tuple[float, float],
) -> None:
    """
    Subject-level trends across sessions, aggregated by session and ROI.

    Y-limits are consistent across subjects and cover all points:
      - rel_theta: global range
      - ratio_slow_fast: global range
      - rel_alpha: global range
      - ratio_theta_beta: global range
    """
    df_agg = df_subj.groupby(["session_id", "roi"]).mean(numeric_only=True).reset_index()

    sessions_sorted = sorted(df_agg["session_id"].astype(str).unique().tolist(), key=_session_sort_key)
    df_agg["session_id"] = pd.Categorical(df_agg["session_id"].astype(str), categories=sessions_sorted, ordered=True)
    df_agg = df_agg.sort_values(["session_id", "roi"])

    rois = ["Frontal", "Posterior", "Central"]
    metrics = ["rel_theta", "rel_alpha", "ratio_slow_fast", "ratio_theta_beta"]

    fig, axes = plt.subplots(len(metrics), len(rois), figsize=(12, 12), sharex=True)
    if len(metrics) == 1 and len(rois) == 1:
        axes = np.array([[axes]])

    for r_idx, roi in enumerate(rois):
        subset = df_agg[df_agg["roi"] == roi]
        if subset.empty:
            continue

        x_labels = subset["session_id"].astype(str).tolist()

        for m_idx, metric in enumerate(metrics):
            ax = axes[m_idx, r_idx]
            if metric not in subset.columns:
                continue

            vals = subset[metric].to_numpy(dtype=float)
            ax.plot(x_labels, vals, marker="o", linestyle="-", linewidth=2)

            if len(vals) > 1 and np.all(np.isfinite(vals)):
                x_nums = np.arange(len(vals), dtype=float)
                z = np.polyfit(x_nums, vals, 1)
                p = np.poly1d(z)
                ax.plot(x_labels, p(x_nums), linestyle="--", alpha=0.6)

            if m_idx == 0:
                ax.set_title(f"{roi} ROI")
            if r_idx == 0:
                ax.set_ylabel(metric)

            if metric == "rel_theta":
                ax.set_ylim(rel_theta_ylim[0], rel_theta_ylim[1])
            elif metric == "ratio_slow_fast":
                ax.set_ylim(ratio_slow_fast_ylim[0], ratio_slow_fast_ylim[1])
            elif metric == "rel_alpha":
                ax.set_ylim(rel_alpha_ylim[0], rel_alpha_ylim[1])
            elif metric == "ratio_theta_beta":
                ax.set_ylim(ratio_theta_beta_ylim[0], ratio_theta_beta_ylim[1])

            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Subject {subject_id}: Relative Power and Fatigue Ratios", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


def plot_time_on_task_subject(
    df_subj: pd.DataFrame,
    subject_id: str,
    out_path: Path,
    cfg: PipelineConfig,
) -> None:
    """
    For a given subject, plot sessions in fixed slots: Session 1, 2, 3.

    If a session is missing, its subplot is left empty.
    Example: only Session 3 exists -> subplot 1 empty, subplot 2 empty, subplot 3 shows Session 3.

    Same y-limits across the 3 subplots (based on ratio_slow_fast range for that subject).
    """
    df_subj = df_subj.copy()
    df_subj["session_id"] = df_subj["session_id"].astype(str)

    # Precompute y-limits across all available sessions for this subject
    y_all = df_subj["ratio_slow_fast"].to_numpy(dtype=float)
    y_min, y_max = _safe_min_max(y_all)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for idx, sess in enumerate(CANONICAL_SESSIONS):
        ax = axes[idx]
        df_sess = df_subj[df_subj["session_id"] == sess]

        if df_sess.empty:
            # Leave subplot empty, but keep the slot position
            ax.set_axis_off()
            continue

        df_agg = df_sess.groupby("t_mid")[["ratio_slow_fast"]].mean().reset_index().sort_values("t_mid")
        if df_agg.empty:
            ax.set_axis_off()
            continue

        # Determine rolling window points from time step
        t_vals = df_agg["t_mid"].to_numpy(dtype=float)
        if len(t_vals) < 2:
            ax.scatter(df_agg["t_mid"], df_agg["ratio_slow_fast"], alpha=0.2, s=12)
        else:
            dt = float(np.median(np.diff(t_vals)))
            window_pts = max(1, int(round(cfg.smooth_sec / max(dt, 1e-9))))
            df_agg["smooth_ratio"] = (
                df_agg["ratio_slow_fast"]
                .rolling(window=window_pts, center=True, min_periods=max(1, window_pts // 4))
                .mean()
            )

            ax.scatter(df_agg["t_mid"], df_agg["ratio_slow_fast"], alpha=0.12, s=10)
            ax.plot(df_agg["t_mid"], df_agg["smooth_ratio"], linewidth=2, label=f"ratio_slow_fast ({cfg.smooth_sec:.0f}s avg)")

            valid = df_agg.dropna()
            if len(valid) > 2:
                res = linregress(valid["t_mid"].to_numpy(dtype=float), valid["ratio_slow_fast"].to_numpy(dtype=float))
                ax.plot(
                    valid["t_mid"],
                    res.intercept + res.slope * valid["t_mid"],
                    linestyle="--",
                    label=f"slope: {res.slope * 60.0:.4f} per min",
                )

        ax.set_title(f"Subject {subject_id} | Session {sess} | Time-on-task")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Fatigue ratio")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


# ----------------------------
# Orchestrator
# ----------------------------

def analyze_directory(edf_dir: Path, out_dir: Path, cfg: PipelineConfig) -> None:
    ensure_dir(out_dir)

    edf_files = sorted(list(edf_dir.glob("*.edf")))
    if not edf_files:
        print(f"No EDF files found in {edf_dir}")
        return

    all_data: List[pd.DataFrame] = []

    for fpath in edf_files:
        subj, sess = parse_subject_session(fpath.stem)
        print(f"Processing: {subj} | Session {sess}")

        try:
            raw = load_raw_edf(fpath)
        except Exception as e:
            print(f"  Error loading {fpath.name}: {e}")
            continue

        raw_clean, _meta = preprocess_with_plots(raw, cfg)

        df_win = compute_bandpower_windowed(raw_clean, cfg, DEFAULT_EEG_PICKS)
        if df_win.empty:
            print(f"  No EEG data extracted for {fpath.name}")
            continue

        df_win["subject_id"] = subj
        df_win["session_id"] = str(sess)
        df_win["file"] = fpath.name

        all_data.append(df_win)

    if not all_data:
        print("No data processed.")
        return

    master_df = pd.concat(all_data, ignore_index=True)

    # Save master tables
    table_dir = out_dir / "tables"
    ensure_dir(table_dir)
    master_df.to_csv(table_dir / "all_windows.csv", index=False)

    # Session slopes summary
    slope_rows: List[Dict[str, object]] = []
    for (subj, sess, roi), group in master_df.groupby(["subject_id", "session_id", "roi"]):
        g = group.dropna(subset=["t_mid", "ratio_slow_fast"])
        if len(g) < 3:
            continue
        res = linregress(g["t_mid"].to_numpy(dtype=float), g["ratio_slow_fast"].to_numpy(dtype=float))
        slope_rows.append(
            {
                "subject_id": subj,
                "session_id": sess,
                "roi": roi,
                "fatigue_slope": float(res.slope),
                "mean_rel_theta": float(g["rel_theta"].mean()),
                "mean_rel_alpha": float(g["rel_alpha"].mean()),
                "mean_ratio_slow_fast": float(g["ratio_slow_fast"].mean()),
            }
        )

    df_summary = pd.DataFrame(slope_rows)
    df_summary.to_csv(table_dir / "session_summary_slopes.csv", index=False)

    # Compute global y-limits for subject trend plots
    # Use the same aggregation the trend plots use (session_id, roi) means.
    df_for_limits = master_df.groupby(["subject_id", "session_id", "roi"]).mean(numeric_only=True).reset_index()

    rel_alpha_vals = df_for_limits["rel_alpha"].to_numpy(dtype=float) if "rel_alpha" in df_for_limits.columns else np.array([])
    rel_theta_vals = df_for_limits["rel_theta"].to_numpy(dtype=float) if "rel_theta" in df_for_limits.columns else np.array([])
    ratio_slow_fast_vals = (
        df_for_limits["ratio_slow_fast"].to_numpy(dtype=float) if "ratio_slow_fast" in df_for_limits.columns else np.array([])
    )
    ratio_theta_beta_vals = (
        df_for_limits["ratio_theta_beta"].to_numpy(dtype=float) if "ratio_theta_beta" in df_for_limits.columns else np.array([])
    )

    rel_alpha_ylim = _safe_min_max(rel_alpha_vals)
    rel_theta_ylim = _safe_min_max(rel_theta_vals)
    ratio_slow_fast_ylim = _safe_min_max(ratio_slow_fast_vals)
    ratio_theta_beta_ylim = _safe_min_max(ratio_theta_beta_vals)

    print(f"Global rel_alpha y-limits (subject trends): {rel_alpha_ylim[0]:.6f} to {rel_alpha_ylim[1]:.6f}")
    print(f"Global rel_theta y-limits (subject trends): {rel_theta_ylim[0]:.6f} to {rel_theta_ylim[1]:.6f}")
    print(f"Global ratio_slow_fast y-limits (subject trends): {ratio_slow_fast_ylim[0]:.6f} to {ratio_slow_fast_ylim[1]:.6f}")
    print(f"Global ratio_theta_beta y-limits (subject trends): {ratio_theta_beta_ylim[0]:.6f} to {ratio_theta_beta_ylim[1]:.6f}")

    # 1) Per-subject time-on-task plots (fixed slots: Session 1,2,3)
    time_subj_dir = out_dir / "time_on_task_subject"
    ensure_dir(time_subj_dir)
    for subj, df_subj in master_df.groupby("subject_id"):
        plot_time_on_task_subject(df_subj, str(subj), time_subj_dir / f"{subj}_time_on_task.png", cfg)

    # 2) Per-subject session comparison trend plots (consistent y-limits across subjects)
    subj_plot_dir = out_dir / "subject_trends"
    ensure_dir(subj_plot_dir)
    for subj, df_subj in master_df.groupby("subject_id"):
        plot_session_comparison_relative(
            df_subj=df_subj,
            subject_id=str(subj),
            out_path=subj_plot_dir / f"{subj}_trends.png",
            cfg=cfg,
            rel_theta_ylim=rel_theta_ylim,
            ratio_slow_fast_ylim=ratio_slow_fast_ylim,
            rel_alpha_ylim=rel_alpha_ylim,
            ratio_theta_beta_ylim=ratio_theta_beta_ylim,
        )

    # 3) Group boxplot of delta ratio from baseline (Frontal ROI)
    group_plot_dir = out_dir / "group_analysis"
    ensure_dir(group_plot_dir)

    if not df_summary.empty:
        df_summary = df_summary.copy()
        df_summary["sess_int"] = pd.to_numeric(df_summary["session_id"], errors="coerce")
        df_summary = df_summary.sort_values(["subject_id", "roi", "sess_int"])

        df_summary["delta_ratio"] = df_summary.groupby(["subject_id", "roi"])["mean_ratio_slow_fast"].transform(
            lambda x: x - x.iloc[0] if len(x) > 0 else x
        )

        subset_frontal = df_summary[df_summary["roi"] == "Frontal"]
        if not subset_frontal.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            subset_frontal.boxplot(column="delta_ratio", by="session_id", ax=ax, grid=False)
            ax.set_title("Change in fatigue ratio (Theta+Alpha)/Beta from Session 1 (Frontal ROI)")
            ax.set_ylabel("Delta ratio")
            ax.axhline(0, linestyle="--", alpha=0.6)
            plt.suptitle("")
            plt.tight_layout()
            plt.savefig(group_plot_dir / "group_delta_fatigue_frontal.png", dpi=cfg.dpi)
            plt.close()

    print(f"Analysis complete. Results saved to {out_dir}")


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Refined DROZY EEG Fatigue Analysis")
    ap.add_argument("--edf_dir", type=str, required=True, help="Path to EDF files")
    ap.add_argument("--out_dir", type=str, required=True, help="Path to output folder")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    cfg = PipelineConfig()
    analyze_directory(Path(args.edf_dir), Path(args.out_dir), cfg)