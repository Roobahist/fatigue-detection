"""
EEG Fatigue Pattern Check for DROZY EDF files (Optimized for Relative Power & Ratios)

Adjustments:
1. Calculates Relative Power (Band / Total) to normalize global amplitude shifts.
2. Computes specific Fatigue Ratios (Theta/Beta, (Theta+Alpha)/Beta).
3. Groups channels into ROIs (Frontal, Central, Posterior).
4. Computes "Time-on-Task" slopes to detect fatigue accumulation within sessions.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy.stats import linregress

# ----------------------------
# Config and Definitions
# ----------------------------

BANDS_HZ: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0), # Adjusted Alpha upper bound to standard 13Hz
    "beta": (13.0, 30.0),
}

# Regions of Interest (ROIs) for spatial specificity
ROIS: Dict[str, List[str]] = {
    "Frontal": ["Fz", "F3", "F4"],
    "Central": ["Cz", "C3", "C4"],
    "Posterior": ["Pz", "Oz", "P3", "P4", "O1", "O2"],
}

DEFAULT_EEG_PICKS = ["Fz", "Cz", "C3", "C4", "Pz", "Oz"]

@dataclass(frozen=True)
class PipelineConfig:
    sfreq_expected: float = 512.0

    # Filtering
    l_freq: float = 0.5  # Lowered slightly to preserve Delta
    h_freq: float = 40.0
    notch_freqs: Tuple[float, ...] = (50.0, 60.0) # Aggressive notch for both standards

    # Referencing
    reference: str = "average"

    # ICA
    ica_n_components: Optional[int] = None
    ica_method: str = "fastica"
    ica_random_state: int = 42
    ica_max_iter: int = 500

    # Band power windows
    window_sec: float = 4.0 # Shorter windows (4s) to capture micro-sleeps better
    overlap: float = 0.5

    # PSD estimation
    psd_fmin: float = 0.5
    psd_fmax: float = 40.0
    psd_n_fft: int = 2048

    # Output
    dpi: int = 150

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_subject_session(stem: str) -> Tuple[str, str]:
    parts = stem.split("-")
    if len(parts) < 2:
        # Fallback for different naming conventions if necessary
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
    # preload=True is essential for filtering
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return raw

def preprocess_with_plots(raw_in: mne.io.BaseRaw, cfg: PipelineConfig) -> Tuple[mne.io.BaseRaw, Dict[str, object]]:
    meta: Dict[str, object] = {}
    raw = raw_in.copy()

    # 1. Channel Selection & Typing
    # Attempt to keep EOG/ECG if available for ICA
    target_types = {}
    for ch in raw.ch_names:
        if ch in DEFAULT_EEG_PICKS:
            target_types[ch] = 'eeg'
        elif 'EOG' in ch.upper():
            target_types[ch] = 'eog'
        elif 'ECG' in ch.upper():
            target_types[ch] = 'ecg'
    
    # If no channel types defined, try to infer or force EEG
    if not target_types: 
         # Fallback: assume all are EEG if names match standard list
         pass 
    else:
        raw.set_channel_types(target_types, on_unit_change='ignore')

    # Pick only relevant channels + artifacts
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

    # 4. Re-referencing (Average)
    # Important: Do this BEFORE ICA
    raw.set_eeg_reference(cfg.reference, verbose="ERROR")

    # 5. ICA
    # High-pass at 1Hz is recommended for ICA to work best
    raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    
    ica = ICA(n_components=cfg.ica_n_components, method=cfg.ica_method, 
              random_state=cfg.ica_random_state, max_iter=cfg.ica_max_iter)
    
    ica.fit(raw_ica_fit, verbose="ERROR")

    # Detect EOG/ECG
    eog_inds, ecg_inds = [], []
    
    # Use built-in detection if channels exist
    if "eog" in raw:
        eog_inds, _ = ica.find_bads_eog(raw, verbose="ERROR")
    if "ecg" in raw:
        ecg_inds, _ = ica.find_bads_ecg(raw, verbose="ERROR")

    # Conservative Exclusion: Only remove if we are sure. 
    # Fatigue signals (Delta/Theta) can look like eye artifacts to bad algorithms.
    # We trust MNE's correlation methods here.
    ica.exclude = list(set(eog_inds + ecg_inds))
    
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose="ERROR")

    meta['ica'] = ica
    meta['n_components_removed'] = len(ica.exclude)

    return raw_clean, meta

def compute_bandpower_windowed(
    raw: mne.io.BaseRaw, 
    cfg: PipelineConfig, 
    eeg_channels: Sequence[str]
) -> pd.DataFrame:
    """
    Computes Absolute Power, Relative Power, and Fatigue Ratios per window.
    """
    # Ensure we only use channels present in data
    valid_chs = [ch for ch in eeg_channels if ch in raw.ch_names]
    if not valid_chs:
        return pd.DataFrame()

    sfreq = raw.info['sfreq']
    # Welch Parameters
    n_fft = min(cfg.psd_n_fft, int(cfg.window_sec * sfreq))
    
    # Generate sliding windows
    epochs = mne.make_fixed_length_epochs(
        raw, duration=cfg.window_sec, overlap=cfg.window_sec * cfg.overlap, 
        verbose="ERROR"
    )
    
    # Compute PSD for all epochs at once
    # shape: (n_epochs, n_channels, n_freqs)
    psds, freqs = epochs.compute_psd(
        method='welch', 
        fmin=cfg.psd_fmin, 
        fmax=cfg.psd_fmax, 
        n_fft=n_fft
    ).get_data(return_freqs=True)

    rows = []
    
    for epoch_idx in range(psds.shape[0]):
        # Time of the window
        t_start = epoch_idx * (cfg.window_sec * (1 - cfg.overlap))
        t_mid = t_start + (cfg.window_sec / 2)

        for ch_idx, ch_name in enumerate(valid_chs):
            psd_ch = psds[epoch_idx, ch_idx, :]
            
            # Calculate band powers
            powers = {}
            total_p = 0.0
            
            for band, (f_low, f_high) in BANDS_HZ.items():
                mask = (freqs >= f_low) & (freqs <= f_high)
                # Simpson's rule or Trapezoidal for integration
                p = np.trapz(psd_ch[mask], freqs[mask])
                powers[band] = p
                total_p += p
            
            # Identify ROI
            roi_name = "Other"
            for r_name, r_chs in ROIS.items():
                if ch_name in r_chs:
                    roi_name = r_name
                    break

            row = {
                't_mid': t_mid,
                'channel': ch_name,
                'roi': roi_name,
                'total_power': total_p
            }

            # Add Absolute and Relative Powers
            for band, p in powers.items():
                row[f'abs_{band}'] = p
                # Handle effectively zero power
                row[f'rel_{band}'] = p / total_p if total_p > 1e-20 else 0.0
                row[f'log_{band}'] = np.log10(p + 1e-20)

            # --- KEY FATIGUE RATIOS ---
            # Theta / Beta
            row['ratio_theta_beta'] = powers['theta'] / powers['beta'] if powers['beta'] > 0 else np.nan
            # (Theta + Alpha) / Beta  (Often considered the "slowing" index)
            row['ratio_slow_fast'] = (powers['theta'] + powers['alpha']) / powers['beta'] if powers['beta'] > 0 else np.nan
            
            rows.append(row)

    return pd.DataFrame(rows)

# ----------------------------
# Plotting Functions
# ----------------------------

def plot_session_comparison_relative(
    df_sess: pd.DataFrame, 
    subject_id: str, 
    out_path: Path, 
    cfg: PipelineConfig
):
    """
    Plots Relative Power across sessions.
    Focuses on Theta and Alpha increasing relative to Total Power.
    """
    # Aggregate by Session and ROI
    # We want to see: Does Frontal Theta increase? Does Posterior Alpha increase?
    
    df_agg = df_sess.groupby(['session_id', 'roi']).mean(numeric_only=True).reset_index()
    
    # Sort sessions naturally
    try:
        df_agg['sess_int'] = df_agg['session_id'].astype(int)
        df_agg = df_agg.sort_values('sess_int')
    except:
        df_agg = df_agg.sort_values('session_id')

    rois = ['Frontal', 'Posterior', 'Central']
    bands_of_interest = ['rel_theta', 'rel_alpha', 'ratio_slow_fast']
    
    fig, axes = plt.subplots(len(bands_of_interest), len(rois), figsize=(12, 10), sharex=True)
    
    for r_idx, roi in enumerate(rois):
        subset = df_agg[df_agg['roi'] == roi]
        if subset.empty: 
            continue
            
        sessions = subset['session_id'].astype(str).tolist()
        
        for b_idx, band in enumerate(bands_of_interest):
            ax = axes[b_idx, r_idx]
            vals = subset[band].values
            
            # Normalize to first session = 0 for clearer "change" visualization?
            # Let's plot raw relative values first, but mark the trend.
            
            ax.plot(sessions, vals, marker='o', linestyle='-', linewidth=2, color='tab:blue')
            
            # Add simple linear fit line to emphasize trend
            if len(vals) > 1:
                x_nums = np.arange(len(vals))
                z = np.polyfit(x_nums, vals, 1)
                p = np.poly1d(z)
                ax.plot(sessions, p(x_nums), "r--", alpha=0.5, label='Trend')

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
    cfg: PipelineConfig
):
    """
    Plots the 'Time-on-Task' effect: Does Ratio (Theta+Alpha)/Beta increase 
    over the 10 minutes?
    """
    # Smooth data for visualization
    df_agg = df_file.groupby('t_mid')[['ratio_slow_fast', 'rel_theta']].mean().reset_index()
    
    # Calculate rolling mean (e.g., 60s window)
    window_pts = int(60 / (df_agg['t_mid'].iloc[1] - df_agg['t_mid'].iloc[0])) # approx 1 min
    df_agg['smooth_ratio'] = df_agg['ratio_slow_fast'].rolling(window=window_pts, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_agg['t_mid'], df_agg['ratio_slow_fast'], alpha=0.1, color='gray', s=10)
    ax.plot(df_agg['t_mid'], df_agg['smooth_ratio'], color='tab:red', linewidth=2, label='(Theta+Alpha)/Beta (1min avg)')
    
    # Fit line
    valid = df_agg.dropna()
    if len(valid) > 2:
        res = linregress(valid['t_mid'], valid['ratio_slow_fast'])
        ax.plot(valid['t_mid'], res.intercept + res.slope*valid['t_mid'], 'k--', 
                label=f'Slope: {res.slope*60:.4f} / min')

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Fatigue Ratio")
    ax.set_title(f"{file_name} | Within-Session Fatigue Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()

# ----------------------------
# Orchestrator
# ----------------------------

def analyze_directory(edf_dir: Path, out_dir: Path, cfg: PipelineConfig):
    ensure_dir(out_dir)
    
    edf_files = sorted(list(edf_dir.glob("*.edf")))
    if not edf_files:
        print(f"No EDF files found in {edf_dir}")
        return

    all_data = []

    for fpath in edf_files:
        subj, sess = parse_subject_session(fpath.stem)
        print(f"Processing: {subj} | Session {sess}")
        
        # 1. Load
        try:
            raw = load_raw_edf(fpath)
        except Exception as e:
            print(f"  Error loading {fpath.name}: {e}")
            continue

        # 2. Preprocess
        raw_clean, meta = preprocess_with_plots(raw, cfg)
        
        # 3. Compute Band Power (Windowed)
        df_win = compute_bandpower_windowed(raw_clean, cfg, DEFAULT_EEG_PICKS)
        
        if df_win.empty:
            print(f"  No EEG data extracted for {fpath.name}")
            continue

        df_win['subject_id'] = subj
        df_win['session_id'] = sess
        df_win['file'] = fpath.name
        
        # 4. Save per-file plot (Time on Task)
        file_plot_dir = out_dir / "time_on_task"
        ensure_dir(file_plot_dir)
        plot_time_on_task(df_win, fpath.stem, file_plot_dir / f"{fpath.stem}_fatigue.png", cfg)
        
        all_data.append(df_win)

    if not all_data:
        print("No data processed.")
        return

    # 5. Master DataFrame
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Save Tables
    table_dir = out_dir / "tables"
    ensure_dir(table_dir)
    master_df.to_csv(table_dir / "all_windows.csv", index=False)

    # 6. Session Aggregates & Slopes
    # Calculate slope of fatigue ratio per session (Time-on-Task Effect)
    slope_rows = []
    for (subj, sess, roi), group in master_df.groupby(['subject_id', 'session_id', 'roi']):
        res = linregress(group['t_mid'], group['ratio_slow_fast'])
        slope_rows.append({
            'subject_id': subj, 'session_id': sess, 'roi': roi,
            'fatigue_slope': res.slope,
            'mean_rel_theta': group['rel_theta'].mean(),
            'mean_rel_alpha': group['rel_alpha'].mean(),
            'mean_ratio_slow_fast': group['ratio_slow_fast'].mean()
        })
    
    df_summary = pd.DataFrame(slope_rows)
    df_summary.to_csv(table_dir / "session_summary_slopes.csv", index=False)

    # 7. Subject Plots (Session Comparison)
    subj_plot_dir = out_dir / "subject_trends"
    ensure_dir(subj_plot_dir)
    
    for subj, df_subj in master_df.groupby('subject_id'):
        plot_session_comparison_relative(df_subj, subj, subj_plot_dir / f"{subj}_trends.png", cfg)

    # 8. Group Boxplots (Normalized to Session 1)
    # We want to see the distribution of change across the group
    group_plot_dir = out_dir / "group_analysis"
    ensure_dir(group_plot_dir)
    
    # Normalize summary to session 1
    # Sort to ensure Session 1 is first
    df_summary['sess_int'] = pd.to_numeric(df_summary['session_id'], errors='coerce')
    df_summary = df_summary.sort_values(['subject_id', 'roi', 'sess_int'])
    
    df_summary['delta_ratio'] = df_summary.groupby(['subject_id', 'roi'])['mean_ratio_slow_fast'].transform(lambda x: x - x.iloc[0])
    
    # Plot Group Boxplot of DELTA Ratio (Change from Baseline)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for Frontal ROI (Best for mental fatigue)
    subset_frontal = df_summary[df_summary['roi'] == 'Frontal']
    
    subset_frontal.boxplot(column='delta_ratio', by='session_id', ax=ax, grid=False)
    ax.set_title("Change in Fatigue Ratio ((Theta+Alpha)/Beta) from Session 1 (Frontal ROI)")
    ax.set_ylabel("Delta Ratio")
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.suptitle("") # Remove pandas auto-title
    plt.savefig(group_plot_dir / "group_delta_fatigue_frontal.png", dpi=cfg.dpi)
    
    print(f"Analysis Complete. Results saved to {out_dir}")

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
    
    # Standard Config
    cfg = PipelineConfig()
    
    analyze_directory(Path(args.edf_dir), Path(args.out_dir), cfg)