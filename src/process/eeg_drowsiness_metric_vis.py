"""
DEBUG VERSION with Step-by-Step Visualization
Generates plots at every processing stage to identify issues
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler

from Zh.src.utils.file_folder import get_file_paths

# Suppress MNE info messages unless verbose
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


# =============================================================================
# Configuration Classes (same as before)
# =============================================================================


@dataclass(frozen=True)
class ChannelConfig:
    """EEG channel configuration for DROZY dataset"""
    eeg_channels: Tuple[str, ...] = ("Fz", "Cz", "C3", "C4", "Pz")
    eog_channels: Tuple[str, ...] = ("EOG-V", "EOG-H")
    montage_name: str = "standard_1020"
    reference: str = "average"


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing pipeline configuration"""
    l_freq: float = 1.0
    h_freq: float = 40.0
    notch_freq: Optional[float] = 50.0
    notch_harmonics: int = 2
    apply_ica: bool = True
    ica_n_components: int = 20
    ica_method: str = "fastica"
    rejection_threshold_uv: float = 150.0


@dataclass(frozen=True)
class SegmentationConfig:
    """Window segmentation configuration"""
    window_length_s: float = 8.0
    overlap_s: float = 4.0
    reject_criteria: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reject_criteria is None:
            object.__setattr__(self, 'reject_criteria', {'eeg': 150e-6})


@dataclass(frozen=True)
class SpectralConfig:
    """Spectral analysis configuration"""
    theta_band: Tuple[float, float] = (4.0, 7.0)
    alpha_band: Tuple[float, float] = (8.0, 13.0)
    beta_band: Tuple[float, float] = (13.0, 30.0)
    psd_method: str = "welch"
    psd_fmin: float = 1.0
    psd_fmax: float = 40.0
    n_fft: int = 512
    n_overlap: int = 256


@dataclass(frozen=True)
class TirednessConfig:
    """Subject-specific tiredness index configuration"""
    extreme_percentile: float = 20.0
    features_for_ranking: Tuple[str, ...] = ("frontal_theta", "alpha_beta_ratio")
    clip_min: float = 0.0
    clip_max: float = 1.5
    normalize_method: str = "percentile"
    percentile_low: float = 5.0
    percentile_high: float = 95.0
    smoothing_window: int = 3


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration"""
    channels: ChannelConfig = ChannelConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    spectral: SpectralConfig = SpectralConfig()
    tiredness: TirednessConfig = TirednessConfig()


# =============================================================================
# Visualization Functions
# =============================================================================


def create_debug_dir(base_dir: str, subject_id: str, session_id: str) -> Path:
    """Create directory for debug plots"""
    debug_dir = Path(base_dir) / f"debug_plots/subject_{subject_id}/session_{session_id}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def plot_raw_data(raw: mne.io.BaseRaw, title: str, output_file: Path):
    """Plot raw EEG data"""
    fig, axes = plt.subplots(len(raw.ch_names), 1, figsize=(16, 3 * len(raw.ch_names)), sharex=True)
    
    if len(raw.ch_names) == 1:
        axes = [axes]
    
    data = raw.get_data() * 1e6  # Convert to µV
    times = raw.times
    
    for idx, (ax, ch_name) in enumerate(zip(axes, raw.ch_names)):
        ax.plot(times, data[idx], linewidth=0.5)
        ax.set_ylabel(f'{ch_name} (µV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], min(times[-1], 60))  # Show first 60 seconds
        
        # Add statistics
        mean_val = data[idx].mean()
        std_val = data[idx].std()
        ax.text(0.02, 0.95, f'Mean: {mean_val:.2f} µV\nStd: {std_val:.2f} µV',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_psd_comparison(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, 
                       title: str, output_file: Path):
    """Plot PSD before and after filtering"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    eeg_channels = [ch for ch in raw_before.ch_names if raw_before.get_channel_types([ch])[0] == 'eeg']
    
    for idx, ch in enumerate(eeg_channels[:3]):  # Plot first 3 channels
        # Before
        ax = axes[0, idx]
        spectrum_before = raw_before.copy().pick([ch]).compute_psd(fmax=100, verbose=False)
        spectrum_before.plot(axes=ax, show=False, average=True)
        ax.set_title(f'{ch} - Before Filtering', fontsize=12, fontweight='bold')
        ax.set_ylim([1e-2, 1e3])
        
        # After
        ax = axes[1, idx]
        spectrum_after = raw_after.copy().pick([ch]).compute_psd(fmax=100, verbose=False)
        spectrum_after.plot(axes=ax, show=False, average=True)
        ax.set_title(f'{ch} - After Filtering (1-40 Hz)', fontsize=12, fontweight='bold')
        ax.set_ylim([1e-2, 1e3])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_epochs_overview(epochs: mne.Epochs, output_file: Path):
    """Plot epochs overview"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Average across epochs for each channel
    ax = axes[0, 0]
    data = epochs.get_data() * 1e6  # Convert to µV
    for idx, ch in enumerate(epochs.ch_names):
        ax.plot(epochs.times, data[:, idx, :].mean(axis=0), label=ch, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Average Epoch Across All Epochs', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. First 5 epochs for first channel
    ax = axes[0, 1]
    for i in range(min(5, len(epochs))):
        ax.plot(epochs.times, data[i, 0, :], alpha=0.6, label=f'Epoch {i}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{epochs.ch_names[0]} (µV)')
    ax.set_title(f'First 5 Epochs - {epochs.ch_names[0]}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Amplitude distribution
    ax = axes[1, 0]
    ax.hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Amplitude (µV)')
    ax.set_ylabel('Count')
    ax.set_title('Amplitude Distribution (All Epochs)', fontweight='bold')
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    ax.axvline(data.mean() - data.std(), color='orange', linestyle='--', alpha=0.5)
    ax.axvline(data.mean() + data.std(), color='orange', linestyle='--', alpha=0.5, 
               label=f'Std: {data.std():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Epoch statistics
    ax = axes[1, 1]
    epoch_means = data.mean(axis=(1, 2))
    epoch_stds = data.std(axis=(1, 2))
    ax.scatter(range(len(epochs)), epoch_means, alpha=0.6, label='Mean')
    ax.scatter(range(len(epochs)), epoch_stds, alpha=0.6, label='Std')
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Per-Epoch Statistics', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_psd_per_epoch(psds: np.ndarray, freqs: np.ndarray, eeg_ch_names: List[str], 
                       cfg: SpectralConfig, output_file: Path):
    """Plot PSD for sample epochs (only EEG channels)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot first 6 epochs
    for i, ax in enumerate(axes):
        if i >= len(psds):
            break
        
        # Only plot EEG channels (psds only contains EEG channels)
        for ch_idx, ch_name in enumerate(eeg_ch_names):
            if ch_idx < psds.shape[1]:  # Make sure index is valid
                ax.semilogy(freqs, psds[i, ch_idx, :], label=ch_name, alpha=0.7)
        
        # Mark frequency bands
        ax.axvspan(cfg.theta_band[0], cfg.theta_band[1], alpha=0.2, color='blue', label='Theta')
        ax.axvspan(cfg.alpha_band[0], cfg.alpha_band[1], alpha=0.2, color='green', label='Alpha')
        ax.axvspan(cfg.beta_band[0], cfg.beta_band[1], alpha=0.2, color='red', label='Beta')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (V²/Hz)')
        ax.set_title(f'Epoch {i} PSD', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_band_powers(theta_power: np.ndarray, alpha_power: np.ndarray, beta_power: np.ndarray,
                     epochs: mne.Epochs, output_file: Path):
    """Plot band powers across epochs"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Average across channels
    theta_mean = theta_power.mean(axis=1)
    alpha_mean = alpha_power.mean(axis=1)
    beta_mean = beta_power.mean(axis=1)
    
    # 1. Band powers over epochs
    ax = axes[0, 0]
    ax.plot(theta_mean, label='Theta', marker='o', alpha=0.7, markersize=3)
    ax.plot(alpha_mean, label='Alpha', marker='s', alpha=0.7, markersize=3)
    ax.plot(beta_mean, label='Beta', marker='^', alpha=0.7, markersize=3)
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Power')
    ax.set_title('Band Powers Across Epochs (Mean Channels)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Band power distributions
    ax = axes[0, 1]
    ax.hist([theta_mean, alpha_mean, beta_mean], bins=30, label=['Theta', 'Alpha', 'Beta'],
            alpha=0.6, edgecolor='black')
    ax.set_xlabel('Power')
    ax.set_ylabel('Count')
    ax.set_title('Band Power Distributions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3-5. Per-channel band powers for first 3 channels
    for idx in range(min(3, theta_power.shape[1])):
        ax = axes[idx // 2 + 1, idx % 2]
        ch_name = epochs.ch_names[idx] if idx < len(epochs.ch_names) else f'Ch{idx}'
        
        ax.plot(theta_power[:, idx], label='Theta', alpha=0.7, linewidth=1)
        ax.plot(alpha_power[:, idx], label='Alpha', alpha=0.7, linewidth=1)
        ax.plot(beta_power[:, idx], label='Beta', alpha=0.7, linewidth=1)
        ax.set_xlabel('Epoch Index')
        ax.set_ylabel('Power')
        ax.set_title(f'Band Powers - {ch_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Theta/Alpha ratio
    ax = axes[2, 1]
    ratio = theta_mean / (alpha_mean + 1e-10)
    ax.plot(ratio, color='purple', marker='o', alpha=0.7, markersize=3)
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Theta/Alpha Ratio')
    ax.set_title('Theta/Alpha Ratio Across Epochs', fontweight='bold')
    ax.axhline(ratio.mean(), color='red', linestyle='--', label=f'Mean: {ratio.mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_relative_band_powers(rel_theta: np.ndarray, rel_alpha: np.ndarray, rel_beta: np.ndarray,
                              epochs: mne.Epochs, output_file: Path):
    """Plot relative band powers"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Average across channels
    rel_theta_mean = rel_theta.mean(axis=1)
    rel_alpha_mean = rel_alpha.mean(axis=1)
    rel_beta_mean = rel_beta.mean(axis=1)
    
    # 1. Stacked area plot
    ax = axes[0, 0]
    ax.stackplot(range(len(rel_theta_mean)), rel_theta_mean, rel_alpha_mean, rel_beta_mean,
                 labels=['Theta', 'Alpha', 'Beta'], alpha=0.7)
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Relative Power')
    ax.set_title('Relative Band Powers (Stacked)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Individual lines
    ax = axes[0, 1]
    ax.plot(rel_theta_mean, label='Theta', marker='o', alpha=0.7, markersize=3)
    ax.plot(rel_alpha_mean, label='Alpha', marker='s', alpha=0.7, markersize=3)
    ax.plot(rel_beta_mean, label='Beta', marker='^', alpha=0.7, markersize=3)
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Relative Power')
    ax.set_title('Relative Band Powers', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution boxplot
    ax = axes[1, 0]
    data_to_plot = [rel_theta_mean, rel_alpha_mean, rel_beta_mean]
    bp = ax.boxplot(data_to_plot, labels=['Theta', 'Alpha', 'Beta'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'green', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('Relative Power')
    ax.set_title('Relative Band Power Distributions', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Correlation matrix
    ax = axes[1, 1]
    corr_data = np.column_stack([rel_theta_mean, rel_alpha_mean, rel_beta_mean])
    corr_matrix = np.corrcoef(corr_data.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Theta', 'Alpha', 'Beta'])
    ax.set_yticklabels(['Theta', 'Alpha', 'Beta'])
    ax.set_title('Band Power Correlations', fontweight='bold')
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=12)
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_features(features: pd.DataFrame, output_file: Path):
    """Plot extracted features"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # 1. Feature distributions
    ax = axes[0, 0]
    key_features = ['frontal_theta', 'posterior_alpha', 'frontal_central_beta']
    for feat in key_features:
        if feat in features.columns:
            ax.hist(features[feat], bins=30, alpha=0.5, label=feat, edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title('Key Feature Distributions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Feature time series
    ax = axes[0, 1]
    for feat in key_features:
        if feat in features.columns:
            ax.plot(features[feat], label=feat, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Value')
    ax.set_title('Key Features Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Theta/Alpha ratio
    ax = axes[1, 0]
    if 'theta_alpha_ratio' in features.columns:
        ax.plot(features['theta_alpha_ratio'], color='purple', marker='o', 
                alpha=0.7, markersize=3, linewidth=1)
        ax.set_xlabel('Epoch Index')
        ax.set_ylabel('Theta/Alpha Ratio')
        ax.set_title('Theta/Alpha Ratio', fontweight='bold')
        ax.axhline(features['theta_alpha_ratio'].mean(), color='red', linestyle='--',
                  label=f"Mean: {features['theta_alpha_ratio'].mean():.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Alpha/Beta ratio
    ax = axes[1, 1]
    if 'alpha_beta_ratio' in features.columns:
        ax.plot(features['alpha_beta_ratio'], color='green', marker='s',
                alpha=0.7, markersize=3, linewidth=1)
        ax.set_xlabel('Epoch Index')
        ax.set_ylabel('Alpha/Beta Ratio')
        ax.set_title('Alpha/Beta Ratio', fontweight='bold')
        ax.axhline(features['alpha_beta_ratio'].mean(), color='red', linestyle='--',
                  label=f"Mean: {features['alpha_beta_ratio'].mean():.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Feature correlation heatmap
    ax = axes[2, 0]
    feature_cols = [col for col in features.columns if any(x in col for x in ['theta', 'alpha', 'beta'])]
    if len(feature_cols) > 1:
        corr = features[feature_cols[:10]].corr()  # Limit to first 10 features
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(corr)))
        ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title('Feature Correlations', fontweight='bold')
        plt.colorbar(im, ax=ax)
    
    # 6. Feature statistics table
    ax = axes[2, 1]
    ax.axis('off')
    stats_data = []
    for feat in key_features + ['theta_alpha_ratio', 'alpha_beta_ratio']:
        if feat in features.columns:
            stats_data.append([
                feat,
                f"{features[feat].mean():.4f}",
                f"{features[feat].std():.4f}",
                f"{features[feat].min():.4f}",
                f"{features[feat].max():.4f}"
            ])
    
    if stats_data:
        table = ax.table(cellText=stats_data,
                        colLabels=['Feature', 'Mean', 'Std', 'Min', 'Max'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title('Feature Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_extreme_windows(features: pd.DataFrame, alert_mask: np.ndarray, 
                        fatigued_mask: np.ndarray, fatigue_marker: np.ndarray,
                        output_file: Path):
    """Plot extreme window identification"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Fatigue marker distribution with thresholds
    ax = axes[0, 0]
    ax.hist(fatigue_marker, bins=50, alpha=0.7, edgecolor='black', color='gray')
    ax.axvline(fatigue_marker[alert_mask].max(), color='green', linestyle='--', 
              linewidth=2, label=f'Alert threshold')
    ax.axvline(fatigue_marker[fatigued_mask].min(), color='red', linestyle='--',
              linewidth=2, label=f'Fatigue threshold')
    ax.set_xlabel('Fatigue Marker')
    ax.set_ylabel('Count')
    ax.set_title('Fatigue Marker Distribution with Thresholds', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Fatigue marker over time
    ax = axes[0, 1]
    ax.plot(fatigue_marker, color='gray', alpha=0.5, linewidth=1)
    ax.scatter(np.where(alert_mask)[0], fatigue_marker[alert_mask], 
              color='green', label=f'Alert ({alert_mask.sum()})', s=50, alpha=0.7)
    ax.scatter(np.where(fatigued_mask)[0], fatigue_marker[fatigued_mask],
              color='red', label=f'Fatigued ({fatigued_mask.sum()})', s=50, alpha=0.7)
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Fatigue Marker')
    ax.set_title('Fatigue Marker Over Time with Extreme Windows', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Feature comparison: alert vs fatigued
    ax = axes[1, 0]
    key_features = ['frontal_theta', 'posterior_alpha', 'alpha_beta_ratio']
    x_pos = np.arange(len(key_features))
    
    alert_means = [features.loc[alert_mask, feat].mean() if feat in features.columns else 0 
                   for feat in key_features]
    fatigued_means = [features.loc[fatigued_mask, feat].mean() if feat in features.columns else 0
                      for feat in key_features]
    
    width = 0.35
    ax.bar(x_pos - width/2, alert_means, width, label='Alert', alpha=0.7, color='green')
    ax.bar(x_pos + width/2, fatigued_means, width, label='Fatigued', alpha=0.7, color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(key_features, rotation=15, ha='right')
    ax.set_ylabel('Mean Value')
    ax.set_title('Feature Comparison: Alert vs Fatigued', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter plot of two key features
    ax = axes[1, 1]
    if 'frontal_theta' in features.columns and 'alpha_beta_ratio' in features.columns:
        # All epochs
        ax.scatter(features['frontal_theta'], features['alpha_beta_ratio'],
                  color='gray', alpha=0.3, s=30, label='Other')
        # Alert
        ax.scatter(features.loc[alert_mask, 'frontal_theta'],
                  features.loc[alert_mask, 'alpha_beta_ratio'],
                  color='green', alpha=0.7, s=50, label='Alert', edgecolors='black')
        # Fatigued
        ax.scatter(features.loc[fatigued_mask, 'frontal_theta'],
                  features.loc[fatigued_mask, 'alpha_beta_ratio'],
                  color='red', alpha=0.7, s=50, label='Fatigued', edgecolors='black')
        ax.set_xlabel('Frontal Theta')
        ax.set_ylabel('Alpha/Beta Ratio')
        ax.set_title('Extreme Windows in Feature Space', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_projection_analysis(features: pd.DataFrame, alert_mask: np.ndarray,
                             fatigued_mask: np.ndarray, raw_scores: np.ndarray,
                             output_file: Path):
    """Plot projection analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Raw projection scores distribution
    ax = axes[0, 0]
    ax.hist(raw_scores, bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax.axvline(raw_scores.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {raw_scores.mean():.3f}')
    ax.axvline(raw_scores.mean() - raw_scores.std(), color='orange', linestyle='--', alpha=0.5)
    ax.axvline(raw_scores.mean() + raw_scores.std(), color='orange', linestyle='--', alpha=0.5,
              label=f'Std: {raw_scores.std():.3f}')
    ax.set_xlabel('Raw Projection Score')
    ax.set_ylabel('Count')
    ax.set_title('Raw Projection Score Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Projection scores over time
    ax = axes[0, 1]
    ax.plot(raw_scores, color='purple', alpha=0.5, linewidth=1)
    ax.scatter(np.where(alert_mask)[0], raw_scores[alert_mask],
              color='green', label='Alert ref', s=30, alpha=0.7)
    ax.scatter(np.where(fatigued_mask)[0], raw_scores[fatigued_mask],
              color='red', label='Fatigued ref', s=30, alpha=0.7)
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Raw Projection Score')
    ax.set_title('Projection Scores Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Projection score percentiles
    ax = axes[1, 0]
    percentiles = np.arange(0, 101, 5)
    perc_values = [np.percentile(raw_scores, p) for p in percentiles]
    ax.plot(percentiles, perc_values, marker='o', color='purple')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Projection Score')
    ax.set_title('Projection Score Percentiles', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add key percentiles
    for p in [5, 25, 50, 75, 95]:
        val = np.percentile(raw_scores, p)
        ax.axhline(val, color='red', linestyle='--', alpha=0.3)
        ax.text(102, val, f'{p}th: {val:.3f}', fontsize=8)
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_scores = np.sort(raw_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cumulative, color='purple', linewidth=2)
    ax.set_xlabel('Projection Score')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Projection Scores', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark 5th and 95th percentiles
    p5 = np.percentile(raw_scores, 5)
    p95 = np.percentile(raw_scores, 95)
    ax.axvline(p5, color='green', linestyle='--', label=f'5th: {p5:.3f}')
    ax.axvline(p95, color='red', linestyle='--', label=f'95th: {p95:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


def plot_normalization_analysis(raw_scores: np.ndarray, ti_normalized: np.ndarray,
                                alert_mask: np.ndarray, fatigued_mask: np.ndarray,
                                output_file: Path):
    """Plot normalization analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Before and after normalization
    ax = axes[0, 0]
    ax.scatter(raw_scores, ti_normalized, alpha=0.5, s=20, color='purple')
    ax.scatter(raw_scores[alert_mask], ti_normalized[alert_mask],
              alpha=0.7, s=50, color='green', label='Alert ref', edgecolors='black')
    ax.scatter(raw_scores[fatigued_mask], ti_normalized[fatigued_mask],
              alpha=0.7, s=50, color='red', label='Fatigued ref', edgecolors='black')
    ax.set_xlabel('Raw Projection Score')
    ax.set_ylabel('Normalized TI (0-1)')
    ax.set_title('Normalization Transformation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Normalized TI distribution
    ax = axes[0, 1]
    ax.hist(ti_normalized, bins=50, alpha=0.7, edgecolor='black', color='blue')
    ax.axvline(ti_normalized.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {ti_normalized.mean():.3f}')
    ax.axvline(0.3, color='green', linestyle='--', label='Alert threshold (0.3)')
    ax.axvline(0.7, color='red', linestyle='--', label='Fatigue threshold (0.7)')
    ax.set_xlabel('Tiredness Index')
    ax.set_ylabel('Count')
    ax.set_title('Normalized Tiredness Index Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. TI over time
    ax = axes[1, 0]
    ax.plot(ti_normalized, color='blue', alpha=0.7, linewidth=1.5)
    ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Alert')
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Fatigue')
    ax.fill_between(range(len(ti_normalized)), 0, ti_normalized, alpha=0.3, color='blue')
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Tiredness Index')
    ax.set_title('Tiredness Index Over Time', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Category distribution
    ax = axes[1, 1]
    alert_count = (ti_normalized <= 0.3).sum()
    moderate_count = ((ti_normalized > 0.3) & (ti_normalized <= 0.7)).sum()
    fatigued_count = (ti_normalized > 0.7).sum()
    
    categories = ['Alert\n(≤0.3)', 'Moderate\n(0.3-0.7)', 'Fatigued\n(>0.7)']
    counts = [alert_count, moderate_count, fatigued_count]
    colors_cat = ['green', 'yellow', 'red']
    
    bars = ax.bar(categories, counts, color=colors_cat, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Epochs')
    ax.set_title('Tiredness Category Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = len(ti_normalized)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {output_file.name}")


# =============================================================================
# File Path Utilities
# =============================================================================


def extract_subject_session_id(filepath: str | Path) -> Tuple[str, str]:
    """Extract subject_id and session_id from filename"""
    filename = Path(filepath).stem
    
    if "-" in filename:
        parts = filename.split("-", 1)
        subject_id = parts[0].strip()
        session_id = parts[1].strip()
    elif "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2:
            subject_id = parts[0].strip()
            session_id = parts[1].strip()
        else:
            subject_id = filename
            session_id = "unknown"
    else:
        subject_id = filename
        session_id = "unknown"
    
    return subject_id, session_id


def subject_id_from_path(filepath: Path) -> str:
    """Extract only subject ID from filepath"""
    subject_id, _ = extract_subject_session_id(filepath)
    return subject_id


# =============================================================================
# Processing Functions with Visualization
# =============================================================================


def load_raw_edf(filepath: Path, cfg: ChannelConfig, debug_dir: Path, verbose: bool = False) -> mne.io.BaseRaw:
    """Load EDF file with visualization"""
    print(f"  [LOAD] Loading {filepath.name}...")
    
    raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose="ERROR")
    
    # Plot 1: Raw data before any processing
    plot_raw_data(raw, "Raw EEG Data (Original)", debug_dir / "01_raw_original.png")
    
    available_channels = set(raw.ch_names)
    channel_types = {}
    
    for ch in cfg.eeg_channels:
        if ch in available_channels:
            channel_types[ch] = "eeg"
    
    for ch in cfg.eog_channels:
        if ch in available_channels:
            channel_types[ch] = "eog"
    
    if channel_types:
        raw.set_channel_types(channel_types)
    
    channels_to_pick = [ch for ch in raw.ch_names if ch in channel_types]
    if not channels_to_pick:
        raise ValueError(f"No valid EEG channels found in {filepath.name}")
    
    raw.pick(channels_to_pick)
    
    try:
        montage = mne.channels.make_standard_montage(cfg.montage_name)
        raw.set_montage(montage, on_missing="warn")
    except Exception as e:
        pass
    
    # Plot 2: After channel selection
    plot_raw_data(raw, "After Channel Selection", debug_dir / "02_after_channel_selection.png")
    
    print(f"  [LOAD] ✓ Load complete")
    return raw


def preprocess_raw(raw: mne.io.BaseRaw, cfg: PreprocessConfig, channel_cfg: ChannelConfig,
                  debug_dir: Path, verbose: bool = False) -> mne.io.BaseRaw:
    """Preprocess with visualization"""
    print(f"  [PREPROCESS] Starting preprocessing...")
    
    raw_before = raw.copy()
    raw = raw.copy()
    
    # Filter
    print(f"  [PREPROCESS] Filtering: {cfg.l_freq}-{cfg.h_freq} Hz...")
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks="eeg", verbose="ERROR")
    
    # Plot 3: Before and after filtering (PSD comparison)
    plot_psd_comparison(raw_before, raw, "Filtering Effect (PSD)",
                       debug_dir / "03_filtering_psd_comparison.png")
    
    # Plot 4: After filtering (time domain)
    plot_raw_data(raw, f"After Band-Pass Filter ({cfg.l_freq}-{cfg.h_freq} Hz)",
                 debug_dir / "04_after_filtering.png")
    
    # Notch filter
    if cfg.notch_freq is not None:
        notch_freqs = [cfg.notch_freq * (i + 1) for i in range(cfg.notch_harmonics)]
        fs = raw.info["sfreq"]
        notch_freqs = [f for f in notch_freqs if f < fs / 2]
        if notch_freqs:
            print(f"  [PREPROCESS] Notch filtering: {notch_freqs} Hz...")
            raw.notch_filter(freqs=notch_freqs, picks="eeg", verbose="ERROR")
            
            # Plot 5: After notch filtering
            plot_raw_data(raw, f"After Notch Filter ({notch_freqs} Hz)",
                         debug_dir / "05_after_notch.png")
    
    # Re-referencing
    print(f"  [PREPROCESS] Re-referencing: {channel_cfg.reference}...")
    if channel_cfg.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    
    # Plot 6: After re-referencing
    plot_raw_data(raw, f"After Re-Referencing ({channel_cfg.reference})",
                 debug_dir / "06_after_rereferencing.png")
    
    # ICA
    if cfg.apply_ica:
        print(f"  [PREPROCESS] Running ICA...")
        n_eeg_channels = len([ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg'])
        max_components = n_eeg_channels - 1
        n_components = min(cfg.ica_n_components, max_components)
        
        if n_components >= 1:
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                method=cfg.ica_method,
                random_state=42,
                max_iter=500,
                verbose="ERROR"
            )
            
            ica.fit(raw, picks="eeg", verbose="ERROR")
            
            eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
            if eog_channels:
                eog_indices, eog_scores = ica.find_bads_eog(
                    raw,
                    ch_name=eog_channels[0],
                    threshold=3.0,
                    verbose="ERROR"
                )
                ica.exclude = eog_indices
            
            ica.apply(raw, verbose="ERROR")
            
            # Plot 7: After ICA
            plot_raw_data(raw, "After ICA Artifact Removal",
                         debug_dir / "07_after_ica.png")
    
    print(f"  [PREPROCESS] ✓ Preprocessing complete")
    return raw


def create_epochs(raw: mne.io.BaseRaw, cfg: SegmentationConfig, debug_dir: Path,
                 verbose: bool = False) -> mne.Epochs:
    """Create epochs with visualization"""
    print(f"  [EPOCH] Creating epochs: {cfg.window_length_s}s windows, {cfg.overlap_s}s overlap...")
    
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=cfg.window_length_s,
        overlap=cfg.overlap_s,
        preload=True,
        verbose="ERROR"
    )
    
    print(f"  [EPOCH] Created {len(epochs)} epochs initially")
    
    # Plot 8: Epochs overview before rejection
    plot_epochs_overview(epochs, debug_dir / "08_epochs_before_rejection.png")
    
    if cfg.reject_criteria:
        n_before = len(epochs)
        epochs.drop_bad(reject=cfg.reject_criteria, verbose="ERROR")
        n_after = len(epochs)
        print(f"  [EPOCH] Rejected {n_before - n_after}/{n_before} epochs")
        
        if n_after > 0:
            # Plot 9: Epochs overview after rejection
            plot_epochs_overview(epochs, debug_dir / "09_epochs_after_rejection.png")
    
    return epochs


def extract_spectral_features(epochs: mne.Epochs, cfg: SpectralConfig, channel_cfg: ChannelConfig,
                              debug_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Extract features with visualization"""
    print("  [FEATURES] Computing spectral features...")
    
    # Get only EEG channels for PSD computation
    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    epochs_eeg = epochs.copy().pick(eeg_picks)
    
    # Compute PSD (only on EEG channels)
    try:
        spectrum = epochs_eeg.compute_psd(
            method='welch',
            fmin=cfg.psd_fmin,
            fmax=cfg.psd_fmax,
            n_fft=cfg.n_fft,
            n_overlap=cfg.n_overlap,
            verbose="ERROR"
        )
        psds = spectrum.get_data()
        freqs = spectrum.freqs
    except AttributeError:
        psds, freqs = epochs_eeg.compute_psd(
            method='welch',
            fmin=cfg.psd_fmin,
            fmax=cfg.psd_fmax,
            n_fft=cfg.n_fft,
            n_overlap=cfg.n_overlap,
            verbose="ERROR"
        )
    
    # Plot 10: PSD for sample epochs (pass only EEG channel names)
    eeg_ch_names = epochs_eeg.ch_names
    plot_psd_per_epoch(psds, freqs, eeg_ch_names, cfg,
                      debug_dir / "10_psd_per_epoch.png")
    
    n_epochs = psds.shape[0]
    total_power = np.trapz(psds, freqs, axis=2)
    
    # Compute band powers
    def compute_band_power(psd, freqs, band):
        freq_mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(psd[:, :, freq_mask], freqs[freq_mask], axis=2)
    
    theta_power = compute_band_power(psds, freqs, cfg.theta_band)
    alpha_power = compute_band_power(psds, freqs, cfg.alpha_band)
    beta_power = compute_band_power(psds, freqs, cfg.beta_band)
    
    # Plot 11: Band powers (use epochs_eeg for channel names)
    plot_band_powers(theta_power, alpha_power, beta_power, epochs_eeg,
                    debug_dir / "11_band_powers.png")
    
    # Relative band powers
    eps = 1e-10
    rel_theta = theta_power / (total_power + eps)
    rel_alpha = alpha_power / (total_power + eps)
    rel_beta = beta_power / (total_power + eps)
    
    # Plot 12: Relative band powers
    plot_relative_band_powers(rel_theta, rel_alpha, rel_beta, epochs_eeg,
                             debug_dir / "12_relative_band_powers.png")
    
    # Build features
    ch_names = epochs_eeg.ch_names
    ch_indices = {ch: ch_names.index(ch) for ch in channel_cfg.eeg_channels if ch in ch_names}
    
    features = {}
    for ch_name in channel_cfg.eeg_channels:
        if ch_name in ch_indices:
            idx = ch_indices[ch_name]
            features[f"{ch_name}_theta_rel"] = rel_theta[:, idx]
            features[f"{ch_name}_alpha_rel"] = rel_alpha[:, idx]
            features[f"{ch_name}_beta_rel"] = rel_beta[:, idx]
    
    if "Fz" in ch_indices:
        features["frontal_theta"] = rel_theta[:, ch_indices["Fz"]]
    
    if "Pz" in ch_indices:
        features["posterior_alpha"] = rel_alpha[:, ch_indices["Pz"]]
    
    fc_channels = ["Fz", "Cz", "C3", "C4"]
    fc_indices = [ch_indices[ch] for ch in fc_channels if ch in ch_indices]
    if fc_indices:
        features["frontal_central_beta"] = rel_beta[:, fc_indices].mean(axis=1)
    
    if "Fz" in ch_indices and "Pz" in ch_indices:
        features["theta_alpha_ratio"] = (
            rel_theta[:, ch_indices["Fz"]] / (rel_alpha[:, ch_indices["Pz"]] + eps)
        )
    
    if "Pz" in ch_indices and fc_indices:
        features["alpha_beta_ratio"] = (
            rel_alpha[:, ch_indices["Pz"]] / (rel_beta[:, fc_indices].mean(axis=1) + eps)
        )
    
    df = pd.DataFrame(features)
    
    # Plot 13: Feature overview
    plot_features(df, debug_dir / "13_features.png")
    
    print(f"  [FEATURES] ✓ Extracted {len(df.columns)} features from {n_epochs} epochs")
    
    return df


def identify_extreme_windows(features: pd.DataFrame, cfg: TirednessConfig,
                             debug_dir: Path, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify extremes with visualization"""
    print("  [EXTREME] Identifying extreme windows...")
    
    ranking_features = [f for f in cfg.features_for_ranking if f in features.columns]
    
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(features[ranking_features])
    fatigue_marker = z_scores.sum(axis=1)
    
    low_threshold = np.percentile(fatigue_marker, cfg.extreme_percentile)
    high_threshold = np.percentile(fatigue_marker, 100 - cfg.extreme_percentile)
    
    alert_mask = fatigue_marker <= low_threshold
    fatigued_mask = fatigue_marker >= high_threshold
    
    # Plot 14: Extreme windows
    plot_extreme_windows(features, alert_mask, fatigued_mask, fatigue_marker,
                        debug_dir / "14_extreme_windows.png")
    
    print(f"  [EXTREME] ✓ Alert: {alert_mask.sum()}, Fatigued: {fatigued_mask.sum()}")
    
    return alert_mask, fatigued_mask, fatigue_marker


def compute_tiredness_projection(features: pd.DataFrame, alert_mask: np.ndarray,
                                 fatigued_mask: np.ndarray, cfg: TirednessConfig,
                                 debug_dir: Path, verbose: bool = False) -> np.ndarray:
    """Compute projection with visualization"""
    print("  [PROJECT] Computing tiredness projection...")
    
    X = features.values
    mu_alert = X[alert_mask].mean(axis=0)
    mu_fatigued = X[fatigued_mask].mean(axis=0)
    
    v = mu_fatigued - mu_alert
    v_norm_sq = np.dot(v, v)
    
    if v_norm_sq < 1e-10:
        print("  [PROJECT] ✗ WARNING: No variation!")
        return np.zeros(len(X))
    
    scores = np.dot(X - mu_alert, v) / v_norm_sq
    scores = np.clip(scores, cfg.clip_min, cfg.clip_max)
    
    # Plot 15: Projection analysis
    plot_projection_analysis(features, alert_mask, fatigued_mask, scores,
                            debug_dir / "15_projection_analysis.png")
    
    print(f"  [PROJECT] ✓ Scores: min={scores.min():.3f}, max={scores.max():.3f}")
    
    return scores


def normalize_tiredness_index(scores: np.ndarray, alert_mask: np.ndarray,
                              fatigued_mask: np.ndarray, cfg: TirednessConfig,
                              debug_dir: Path, verbose: bool = False) -> np.ndarray:
    """Normalize with visualization"""
    print("  [NORMALIZE] Normalizing to 0-1 scale...")
    
    if cfg.normalize_method == "percentile":
        s_min = np.percentile(scores, cfg.percentile_low)
        s_max = np.percentile(scores, cfg.percentile_high)
    else:
        s_min = scores.min()
        s_max = scores.max()
    
    if s_max - s_min < 1e-10:
        print("  [NORMALIZE] ✗ WARNING: No variation!")
        return np.zeros_like(scores)
    
    ti = (scores - s_min) / (s_max - s_min)
    ti = np.clip(ti, 0.0, 1.0)
    
    # Plot 16: Normalization analysis
    plot_normalization_analysis(scores, ti, alert_mask, fatigued_mask,
                                debug_dir / "16_normalization_analysis.png")
    
    print(f"  [NORMALIZE] ✓ TI: mean={ti.mean():.3f}, std={ti.std():.3f}")
    
    return ti


def smooth_tiredness_index(ti: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply moving average smoothing"""
    if window_size <= 1 or len(ti) < window_size:
        return ti
    
    kernel = np.ones(window_size) / window_size
    ti_smooth = np.convolve(ti, kernel, mode='same')
    
    return ti_smooth


# Continue in next message due to length...


# =============================================================================
# Main Processing Function
# =============================================================================


def process_single_file_with_plots(filepath: Path, cfg: PipelineConfig, output_base_dir: str):
    """Process a single file with complete visualization"""
    
    subject_id, session_id = extract_subject_session_id(filepath)
    
    print(f"\n{'='*60}")
    print(f"Processing: Subject {subject_id}, Session {session_id}")
    print(f"File: {filepath}")
    print(f"{'='*60}\n")
    
    # Create debug directory
    debug_dir = create_debug_dir(output_base_dir, subject_id, session_id)
    print(f"Debug plots will be saved to: {debug_dir}\n")
    
    # Process with visualization at each step
    raw = load_raw_edf(filepath, cfg.channels, debug_dir)
    raw = preprocess_raw(raw, cfg.preprocess, cfg.channels, debug_dir)
    epochs = create_epochs(raw, cfg.segmentation, debug_dir)
    
    if len(epochs) == 0:
        print("  [ERROR] No valid epochs!")
        return None
    
    features = extract_spectral_features(epochs, cfg.spectral, cfg.channels, debug_dir)
    alert_mask, fatigued_mask, fatigue_marker = identify_extreme_windows(features, cfg.tiredness, debug_dir)
    raw_scores = compute_tiredness_projection(features, alert_mask, fatigued_mask, cfg.tiredness, debug_dir)
    ti_normalized = normalize_tiredness_index(raw_scores, alert_mask, fatigued_mask, cfg.tiredness, debug_dir)
    
    # Apply smoothing
    if cfg.tiredness.smoothing_window > 1:
        ti_smooth = smooth_tiredness_index(ti_normalized, cfg.tiredness.smoothing_window)
    else:
        ti_smooth = ti_normalized
    
    # Create results
    epoch_starts = np.arange(len(epochs)) * cfg.segmentation.overlap_s
    
    results = []
    for i in range(len(features)):
        results.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "file_path": str(filepath),
            "epoch_index": i,
            "start_time_sec": epoch_starts[i],
            "tiredness_index": ti_smooth[i],
            "tiredness_index_raw": ti_normalized[i],
            "projection_score": raw_scores[i],
            "is_alert_ref": alert_mask[i],
            "is_fatigued_ref": fatigued_mask[i]
        })
    
    print(f"\n✓ Processing complete!")
    print(f"  Generated 16+ debug plots in: {debug_dir}")
    print(f"  {len(results)} epochs processed")
    print(f"  Mean TI: {ti_smooth.mean():.3f}, Std: {ti_smooth.std():.3f}\n")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Debug version with step-by-step visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Single EDF file to process with full visualization"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="debug_output",
        help="Output directory for plots and CSV (default: debug_output)"
    )
    parser.add_argument(
        "--no-ica",
        action="store_true",
        help="Disable ICA"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    cfg = PipelineConfig(
        preprocess=PreprocessConfig(apply_ica=not args.no_ica)
    )
    
    # Process file
    filepath = Path(args.input_file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return
    
    result_df = process_single_file_with_plots(filepath, cfg, args.output_dir)
    
    if result_df is not None:
        # Save CSV
        output_csv = Path(args.output_dir) / "results.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()