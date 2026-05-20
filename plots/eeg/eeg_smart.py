"""
EEG Fatigue Analysis V2: Smart Baselines & Population Z-Scoring

Features:
1. Standard Preprocessing (ICA, Filtering).
2. "Smart Baseline": Finds subject's lowest 10% fatigue state across ALL sessions.
3. Population Z-Scoring: Normalizes subjects using global variance.
4. Personal Min-Max: Maps fatigue to a 0-100% relative scale.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # Added for better distribution plots
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy.stats import linregress

# ----------------------------
# Config
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

@dataclass(frozen=True)
class PipelineConfig:
    sfreq_expected: float = 512.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freqs: Tuple[float, ...] = (50.0, 60.0)
    reference: str = "average"
    ica_n_components: Optional[int] = None
    ica_method: str = "fastica"
    ica_random_state: int = 42
    window_sec: float = 4.0
    overlap: float = 0.5
    psd_fmin: float = 0.5
    psd_fmax: float = 40.0
    psd_n_fft: int = 2048
    dpi: int = 150

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_subject_session(stem: str) -> Tuple[str, str]:
    parts = stem.split("-")
    if len(parts) < 2: return stem, "1"
    return "-".join(parts[:-1]), parts[-1]

# ----------------------------
# Processing Components
# ----------------------------

def load_raw_edf(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")

def preprocess_with_plots(raw_in: mne.io.BaseRaw, cfg: PipelineConfig) -> Tuple[mne.io.BaseRaw, Dict[str, object]]:
    raw = raw_in.copy()
    
    # Simple channel typing
    target_types = {}
    for ch in raw.ch_names:
        if ch in DEFAULT_EEG_PICKS: target_types[ch] = 'eeg'
        elif 'EOG' in ch.upper(): target_types[ch] = 'eog'
        elif 'ECG' in ch.upper(): target_types[ch] = 'ecg'
    
    if target_types:
        raw.set_channel_types(target_types, on_unit_change='ignore')
    
    raw.pick(mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True))

    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    except: pass

    if cfg.notch_freqs: raw.notch_filter(cfg.notch_freqs, verbose="ERROR")
    raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")
    raw.set_eeg_reference(cfg.reference, verbose="ERROR")

    # ICA
    raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    ica = ICA(n_components=cfg.ica_n_components, method=cfg.ica_method, 
              random_state=cfg.ica_random_state, max_iter=500)
    ica.fit(raw_ica_fit, verbose="ERROR")
    
    eog_inds, ecg_inds = [], []
    if "eog" in raw: eog_inds, _ = ica.find_bads_eog(raw, verbose="ERROR")
    if "ecg" in raw: ecg_inds, _ = ica.find_bads_ecg(raw, verbose="ERROR")
    
    ica.exclude = list(set(eog_inds + ecg_inds))
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose="ERROR")

    return raw_clean, {}

def compute_bandpower_windowed(raw: mne.io.BaseRaw, cfg: PipelineConfig, eeg_channels: Sequence[str]) -> pd.DataFrame:
    valid_chs = [ch for ch in eeg_channels if ch in raw.ch_names]
    if not valid_chs: return pd.DataFrame()

    sfreq = raw.info['sfreq']
    n_fft = min(cfg.psd_n_fft, int(cfg.window_sec * sfreq))
    
    epochs = mne.make_fixed_length_epochs(raw, duration=cfg.window_sec, overlap=cfg.window_sec * cfg.overlap, verbose="ERROR")
    psds, freqs = epochs.compute_psd(method='welch', fmin=cfg.psd_fmin, fmax=cfg.psd_fmax, n_fft=n_fft).get_data(return_freqs=True)

    rows = []
    for epoch_idx in range(psds.shape[0]):
        t_mid = epoch_idx * (cfg.window_sec * (1 - cfg.overlap)) + (cfg.window_sec / 2)

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
                    roi_name = r_name; break

            row = {
                't_mid': t_mid, 'channel': ch_name, 'roi': roi_name, 'total_power': total_p,
                'rel_theta': powers['theta'] / total_p if total_p > 0 else 0,
                'rel_alpha': powers['alpha'] / total_p if total_p > 0 else 0,
                'ratio_slow_fast': (powers['theta'] + powers['alpha']) / powers['beta'] if powers['beta'] > 0 else np.nan
            }
            rows.append(row)

    return pd.DataFrame(rows)

# ----------------------------
# NEW: Smart Normalization Logic
# ----------------------------

def apply_smart_normalization(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the normalization logic requested:
    1. Find Global SD of Fatigue Ratio (Population Scale).
    2. Find Personal 10th Percentile (Personal Baseline).
    3. Compute Z-Score: (Value - Personal_Base) / Global_SD.
    4. Compute Norm 0-1: (Value - Personal_Base) / (Personal_90th - Personal_Base).
    """
    print("Computing Smart Baselines...")
    
    # 1. Global Population Stats (for variance scaling)
    # We use only Frontal ROI for standardizing to avoid noise from other regions
    frontal_data = df_all[df_all['roi'] == 'Frontal']
    global_sd = frontal_data['ratio_slow_fast'].std()
    print(f"  Global Population SD (Frontal Ratio): {global_sd:.4f}")

    # 2. Personal Baselines
    # Calculate 10th and 90th percentiles per subject (across ALL sessions)
    subject_stats = df_all.groupby('subject_id')['ratio_slow_fast'].quantile([0.1, 0.9]).unstack()
    subject_stats.columns = ['p10', 'p90']
    
    # Merge back
    df_norm = df_all.merge(subject_stats, left_on='subject_id', right_index=True)
    
    # 3. Calculation
    # Z-Score Hybrid: Centered on Personal Best, Scaled by Population Variance
    df_norm['z_score_hybrid'] = (df_norm['ratio_slow_fast'] - df_norm['p10']) / global_sd
    
    # Min-Max Personal: 0 = My Best State, 1 = My Worst State (90th percentile)
    # Clip to 0-1.2 range to handle extreme outliers without squashing the plot
    denom = (df_norm['p90'] - df_norm['p10']).replace(0, 1) # Avoid div/0
    df_norm['norm_0_1'] = (df_norm['ratio_slow_fast'] - df_norm['p10']) / denom
    
    return df_norm

# ----------------------------
# Plotting
# ----------------------------

def plot_smart_comparison(df_norm: pd.DataFrame, out_dir: Path, cfg: PipelineConfig):
    """
    Creates plots using the new normalized metrics.
    """
    ensure_dir(out_dir)
    
    # Filter for Frontal ROI (Best for fatigue)
    df_viz = df_norm[df_norm['roi'] == 'Frontal'].copy()
    
    # Plot 1: Population-Wide Boxplot on Z-Scores
    # Does the group move away from their personal baseline?
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_viz, x='session_id', y='z_score_hybrid', showfliers=False, palette="Blues")
    plt.axhline(0, color='r', linestyle='--', label='Personal Alert Baseline (10th %ile)')
    plt.title("Population Fatigue: Distance from Personal Baseline (in Pop. SD units)")
    plt.ylabel("Z-Score (Hybrid)")
    plt.legend()
    plt.savefig(out_dir / "smart_population_zscore.png", dpi=cfg.dpi)
    plt.close()

    # Plot 2: State Distribution Histograms
    # Visualizes how the distribution shifts for the whole group
    plt.figure(figsize=(10, 6))
    sessions = sorted(df_viz['session_id'].unique())
    for sess in sessions:
        subset = df_viz[df_viz['session_id'] == sess]
        sns.kdeplot(subset['norm_0_1'], label=f"Session {sess}", fill=True, alpha=0.3)
    
    plt.xlim(-0.2, 1.5) # Focus on the 0-1 range
    plt.axvline(0, color='k', linestyle='--', label='Alert Baseline')
    plt.axvline(1, color='k', linestyle=':', label='Fatigued Peak (90th)')
    plt.title("Shift in Fatigue Distributions (0 = Personal Best, 1 = Personal Worst)")
    plt.xlabel("Normalized Fatigue (0-1)")
    plt.legend()
    plt.savefig(out_dir / "smart_distribution_shift.png", dpi=cfg.dpi)
    plt.close()

    # Plot 3: Normalized Trajectories (Time-on-Task)
    # Average trajectory across all subjects
    # Bin time into minutes for cleaner lines
    df_viz['minute'] = (df_viz['t_mid'] // 60).astype(int)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_viz, x='minute', y='z_score_hybrid', hue='session_id', palette="tab10", marker="o")
    plt.title("Average Time-on-Task Trajectory (Normalized)")
    plt.ylabel("Fatigue Z-Score (vs Personal Best)")
    plt.xlabel("Minute of Task")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "smart_trajectory_average.png", dpi=cfg.dpi)
    plt.close()

def plot_individual_smart(df_norm: pd.DataFrame, subject_id: str, out_dir: Path, cfg: PipelineConfig):
    """
    Plots individual subject's Z-score trajectory across sessions.
    """
    subset = df_norm[(df_norm['subject_id'] == subject_id) & (df_norm['roi'] == 'Frontal')]
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=subset, x='t_mid', y='z_score_hybrid', hue='session_id', alpha=0.7)
    
    # Draw the personal baseline at 0
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Personal Baseline')
    
    plt.title(f"Subject {subject_id}: Normalized Fatigue Trajectory")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-Score (Hybrid)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{subject_id}_smart_trajectory.png", dpi=cfg.dpi)
    plt.close()

# ----------------------------
# Orchestrator
# ----------------------------

def analyze_directory(edf_dir: Path, out_dir: Path, cfg: PipelineConfig):
    ensure_dir(out_dir)
    edf_files = sorted(list(edf_dir.glob("*.edf")))
    
    all_data = []

    # 1. Process Files
    for fpath in edf_files:
        subj, sess = parse_subject_session(fpath.stem)
        print(f"Loading: {subj} | Session {sess}")
        
        try:
            raw = load_raw_edf(fpath)
            raw_clean, _ = preprocess_with_plots(raw, cfg)
            df_win = compute_bandpower_windowed(raw_clean, cfg, DEFAULT_EEG_PICKS)
            
            if not df_win.empty:
                df_win['subject_id'] = subj
                df_win['session_id'] = sess
                df_win['file'] = fpath.name
                all_data.append(df_win)
        except Exception as e:
            print(f"Error {fpath.name}: {e}")

    if not all_data: return
    
    master_df = pd.concat(all_data, ignore_index=True)
    
    # 2. Apply Smart Normalization (The new step)
    master_df_norm = apply_smart_normalization(master_df)
    
    # Save processed data
    ensure_dir(out_dir / "tables")
    master_df_norm.to_csv(out_dir / "tables" / "smart_normalized_data.csv", index=False)
    
    # 3. Generate New Plots
    print("Generating Smart Plots...")
    plot_smart_comparison(master_df_norm, out_dir / "group_smart", cfg)
    
    indiv_dir = out_dir / "individual_smart"
    ensure_dir(indiv_dir)
    for subj, grp in master_df_norm.groupby('subject_id'):
        plot_individual_smart(grp, subj, indiv_dir, cfg)
        
    print(f"Done. Results in {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    
    analyze_directory(Path(args.edf_dir), Path(args.out_dir), PipelineConfig())