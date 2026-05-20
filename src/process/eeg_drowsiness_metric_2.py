"""
Subject-Specific Tiredness Index Pipeline (FIXED VERSION)

This version computes tiredness indices independently for each subject,
ensuring that each person's alert and fatigued states are relative to their
own baseline, not the population average.

Key improvements:
1. Subject-specific alert/fatigued references (not population-wide)
2. Better handling of edge cases (zero variance, short sessions)
3. Optional z-score normalization within subjects
4. Comprehensive diagnostics and error handling
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
from scipy import signal
from sklearn.preprocessing import StandardScaler

from Zh.src.utils.file_folder import get_file_paths

# Suppress MNE info messages unless verbose
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass(frozen=True)
class ChannelConfig:
    """EEG channel configuration for DROZY dataset"""
    eeg_channels: Tuple[str, ...] = ("Fz", "Cz", "C3", "C4", "Pz")
    eog_channels: Tuple[str, ...] = ("EOG-V", "EOG-H")
    montage_name: str = "standard_1020"
    reference: str = "average"  # 'average' or 'mastoids'


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
    min_epochs_required: int = 10  # Minimum epochs to compute TI


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration"""
    channels: ChannelConfig = ChannelConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    spectral: SpectralConfig = SpectralConfig()
    tiredness: TirednessConfig = TirednessConfig()


# =============================================================================
# File Path Utilities
# =============================================================================


def extract_subject_session_id(filepath: str | Path) -> Tuple[str, str]:
    """Extract subject_id and session_id from filename."""
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
# Step 1: Data Loading and Channel Setup
# =============================================================================


def load_raw_edf(
    filepath: Path,
    cfg: ChannelConfig,
    verbose: bool = False
) -> mne.io.BaseRaw:
    """Load EDF file and set up channel information."""
    if verbose:
        print(f"  [LOAD] Loading {filepath.name}...")
    
    try:
        raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose="ERROR")
        if verbose:
            print(f"  [LOAD] ✓ Duration: {raw.times[-1]:.1f}s, Fs: {raw.info['sfreq']} Hz")
    except Exception as e:
        print(f"  [LOAD] ✗ ERROR loading file: {e}")
        raise
    
    available_channels = set(raw.ch_names)
    channel_types = {}
    
    # EEG channels
    for ch in cfg.eeg_channels:
        if ch in available_channels:
            channel_types[ch] = "eeg"
        elif verbose:
            print(f"  [LOAD] ⚠ Missing EEG channel: {ch}")
    
    # EOG channels (optional)
    for ch in cfg.eog_channels:
        if ch in available_channels:
            channel_types[ch] = "eog"
    
    if not channel_types:
        raise ValueError(f"No valid EEG channels found in {filepath.name}")
    
    raw.set_channel_types(channel_types)
    channels_to_pick = [ch for ch in raw.ch_names if ch in channel_types]
    raw.pick(channels_to_pick)
    
    # Set montage
    try:
        montage = mne.channels.make_standard_montage(cfg.montage_name)
        raw.set_montage(montage, on_missing="warn")
    except Exception as e:
        if verbose:
            print(f"  [LOAD] ⚠ Could not set montage: {e}")
    
    if verbose:
        print(f"  [LOAD] ✓ Loaded {len(channels_to_pick)} channels")
    
    return raw


# =============================================================================
# Step 2: Preprocessing
# =============================================================================


def preprocess_raw(
    raw: mne.io.BaseRaw,
    cfg: PreprocessConfig,
    channel_cfg: ChannelConfig,
    verbose: bool = False
) -> mne.io.BaseRaw:
    """Apply preprocessing pipeline."""
    if verbose:
        print(f"  [PREPROCESS] Starting...")
    
    raw = raw.copy()
    
    # 1. Band-pass filter
    try:
        raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks="eeg", verbose="ERROR")
        if verbose:
            print(f"  [PREPROCESS] ✓ Filtered {cfg.l_freq}-{cfg.h_freq} Hz")
    except Exception as e:
        print(f"  [PREPROCESS] ✗ ERROR during filtering: {e}")
        raise
    
    # 2. Notch filter
    if cfg.notch_freq is not None:
        notch_freqs = [cfg.notch_freq * (i + 1) for i in range(cfg.notch_harmonics)]
        fs = raw.info["sfreq"]
        notch_freqs = [f for f in notch_freqs if f < fs / 2]
        if notch_freqs:
            try:
                raw.notch_filter(freqs=notch_freqs, picks="eeg", verbose="ERROR")
                if verbose:
                    print(f"  [PREPROCESS] ✓ Notch filtered {notch_freqs} Hz")
            except Exception as e:
                print(f"  [PREPROCESS] ✗ ERROR during notch filtering: {e}")
                raise
    
    # 3. Re-referencing
    try:
        if channel_cfg.reference == "average":
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        elif channel_cfg.reference == "mastoids":
            if "A1" in raw.ch_names and "A2" in raw.ch_names:
                raw.set_eeg_reference(["A1", "A2"], verbose="ERROR")
            else:
                if verbose:
                    print("  [PREPROCESS] ⚠ Mastoids unavailable, using average")
                raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        if verbose:
            print(f"  [PREPROCESS] ✓ Re-referenced to {channel_cfg.reference}")
    except Exception as e:
        print(f"  [PREPROCESS] ✗ ERROR during re-referencing: {e}")
        raise
    
    # 4. ICA for artifact removal
    if cfg.apply_ica:
        try:
            n_eeg_channels = len([ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg'])
            max_components = n_eeg_channels - 1
            n_components = min(cfg.ica_n_components, max_components)
            
            if n_components < 1:
                if verbose:
                    print(f"  [PREPROCESS] ⚠ Skipping ICA: only {n_eeg_channels} channels")
            else:
                ica = mne.preprocessing.ICA(
                    n_components=n_components,
                    method=cfg.ica_method,
                    random_state=42,
                    max_iter=500,
                    verbose="ERROR"
                )
                
                ica.fit(raw, picks="eeg", verbose="ERROR")
                
                # Find EOG-related components
                eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
                if eog_channels:
                    eog_indices, eog_scores = ica.find_bads_eog(
                        raw,
                        ch_name=eog_channels[0],
                        threshold=3.0,
                        verbose="ERROR"
                    )
                    ica.exclude = eog_indices
                    if verbose and eog_indices:
                        print(f"  [PREPROCESS] Excluding ICA components: {eog_indices}")
                
                ica.apply(raw, verbose="ERROR")
                if verbose:
                    print(f"  [PREPROCESS] ✓ ICA applied ({n_components} components)")
        except Exception as e:
            print(f"  [PREPROCESS] ✗ ERROR during ICA: {e}")
            raise
    
    return raw


# =============================================================================
# Step 3: Segmentation into Analysis Windows
# =============================================================================


def create_epochs(
    raw: mne.io.BaseRaw,
    cfg: SegmentationConfig,
    verbose: bool = False
) -> mne.Epochs:
    """Segment continuous data into fixed-length epochs with overlap."""
    if verbose:
        print(f"  [EPOCH] Creating {cfg.window_length_s}s windows, {cfg.overlap_s}s overlap...")
    
    try:
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=cfg.window_length_s,
            overlap=cfg.overlap_s,
            preload=True,
            verbose="ERROR"
        )
        if verbose:
            print(f"  [EPOCH] ✓ Created {len(epochs)} epochs")
    except Exception as e:
        print(f"  [EPOCH] ✗ ERROR creating epochs: {e}")
        raise
    
    # Drop bad epochs
    if cfg.reject_criteria:
        try:
            n_before = len(epochs)
            epochs.drop_bad(reject=cfg.reject_criteria, verbose="ERROR")
            n_after = len(epochs)
            n_rejected = n_before - n_after
            if verbose:
                print(f"  [EPOCH] Rejected {n_rejected}/{n_before} epochs ({n_rejected/n_before*100:.1f}%)")
            
            if n_after == 0:
                print(f"  [EPOCH] ✗ WARNING: ALL EPOCHS REJECTED!")
        except Exception as e:
            print(f"  [EPOCH] ✗ ERROR during epoch rejection: {e}")
            raise
    
    return epochs


# =============================================================================
# Step 4: Spectral Feature Extraction
# =============================================================================


def compute_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    band: Tuple[float, float]
) -> np.ndarray:
    """Compute power in a frequency band."""
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.trapz(psd[:, :, freq_mask], freqs[freq_mask], axis=2)
    return band_power


def extract_spectral_features(
    epochs: mne.Epochs,
    cfg: SpectralConfig,
    channel_cfg: ChannelConfig,
    verbose: bool = False
) -> pd.DataFrame:
    """Extract spectral features from epochs."""
    if verbose:
        print("  [FEATURES] Computing spectral features...")
    
    try:
        # Compute PSD - handle both old and new MNE versions
        try:
            if cfg.psd_method == "welch":
                spectrum = epochs.compute_psd(
                    method='welch',
                    fmin=cfg.psd_fmin,
                    fmax=cfg.psd_fmax,
                    n_fft=cfg.n_fft,
                    n_overlap=cfg.n_overlap,
                    verbose="ERROR"
                )
            else:
                spectrum = epochs.compute_psd(
                    method='multitaper',
                    fmin=cfg.psd_fmin,
                    fmax=cfg.psd_fmax,
                    verbose="ERROR"
                )
            psds = spectrum.get_data()
            freqs = spectrum.freqs
        except AttributeError:
            # Fall back to old MNE API
            if cfg.psd_method == "welch":
                psds, freqs = epochs.compute_psd(
                    method='welch',
                    fmin=cfg.psd_fmin,
                    fmax=cfg.psd_fmax,
                    n_fft=cfg.n_fft,
                    n_overlap=cfg.n_overlap,
                    verbose="ERROR"
                )
            else:
                psds, freqs = epochs.compute_psd(
                    method='multitaper',
                    fmin=cfg.psd_fmin,
                    fmax=cfg.psd_fmax,
                    verbose="ERROR"
                )
        
        if verbose:
            print(f"  [FEATURES] ✓ PSD computed, shape: {psds.shape}")
        
        # Check for zero PSDs
        if np.all(psds == 0):
            print(f"  [FEATURES] ✗ WARNING: All PSD values are ZERO!")
            print(f"  [FEATURES] Data may have been filtered to nothing")
        
    except Exception as e:
        print(f"  [FEATURES] ✗ ERROR computing PSD: {e}")
        raise
    
    # Compute total power
    total_power = np.trapz(psds, freqs, axis=2)
    
    # Compute band powers
    theta_power = compute_band_power(psds, freqs, cfg.theta_band)
    alpha_power = compute_band_power(psds, freqs, cfg.alpha_band)
    beta_power = compute_band_power(psds, freqs, cfg.beta_band)
    
    # Relative band powers
    eps = 1e-10
    rel_theta = theta_power / (total_power + eps)
    rel_alpha = alpha_power / (total_power + eps)
    rel_beta = beta_power / (total_power + eps)
    
    # Get channel indices
    ch_names = epochs.ch_names
    ch_indices = {ch: ch_names.index(ch) for ch in channel_cfg.eeg_channels if ch in ch_names}
    
    # Initialize features dictionary
    features = {}
    
    # Per-channel relative band powers
    for ch_name in channel_cfg.eeg_channels:
        if ch_name in ch_indices:
            idx = ch_indices[ch_name]
            features[f"{ch_name}_theta_rel"] = rel_theta[:, idx]
            features[f"{ch_name}_alpha_rel"] = rel_alpha[:, idx]
            features[f"{ch_name}_beta_rel"] = rel_beta[:, idx]
    
    # Regional features
    if "Fz" in ch_indices:
        features["frontal_theta"] = rel_theta[:, ch_indices["Fz"]]
    
    if "Pz" in ch_indices:
        features["posterior_alpha"] = rel_alpha[:, ch_indices["Pz"]]
    
    # Frontal-central beta
    fc_channels = ["Fz", "Cz", "C3", "C4"]
    fc_indices = [ch_indices[ch] for ch in fc_channels if ch in ch_indices]
    if fc_indices:
        features["frontal_central_beta"] = rel_beta[:, fc_indices].mean(axis=1)
    
    # Ratios
    if "Fz" in ch_indices and "Pz" in ch_indices:
        features["theta_alpha_ratio"] = (
            rel_theta[:, ch_indices["Fz"]] / (rel_alpha[:, ch_indices["Pz"]] + eps)
        )
    
    if "Pz" in ch_indices and fc_indices:
        features["alpha_beta_ratio"] = (
            rel_alpha[:, ch_indices["Pz"]] / (rel_beta[:, fc_indices].mean(axis=1) + eps)
        )
    
    df = pd.DataFrame(features)
    
    if verbose:
        print(f"  [FEATURES] ✓ Extracted {len(df.columns)} features from {len(df)} epochs")
    
    return df


# =============================================================================
# Step 5: Subject-Internal Alert vs Fatigued Reference
# =============================================================================


def identify_extreme_windows(
    features: pd.DataFrame,
    cfg: TirednessConfig,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify alert and fatigued windows based on EEG features."""
    if verbose:
        print(f"  [EXTREME] Identifying extreme windows (n={len(features)})...")
    
    # Select features for ranking
    ranking_features = [f for f in cfg.features_for_ranking if f in features.columns]
    
    if not ranking_features:
        raise ValueError(f"None of the ranking features {cfg.features_for_ranking} found")
    
    try:
        # Compute z-scores
        scaler = StandardScaler()
        z_scores = scaler.fit_transform(features[ranking_features])
        
        # Combine into fatigue marker (higher = more fatigued)
        fatigue_marker = z_scores.sum(axis=1)
        
        # Compute percentile thresholds
        low_threshold = np.percentile(fatigue_marker, cfg.extreme_percentile)
        high_threshold = np.percentile(fatigue_marker, 100 - cfg.extreme_percentile)
        
        # Identify alert and fatigued windows
        alert_mask = fatigue_marker <= low_threshold
        fatigued_mask = fatigue_marker >= high_threshold
        
        n_alert = alert_mask.sum()
        n_fatigued = fatigued_mask.sum()
        
        if verbose:
            print(f"  [EXTREME] ✓ Alert: {n_alert} ({n_alert/len(features)*100:.1f}%)")
            print(f"  [EXTREME] ✓ Fatigued: {n_fatigued} ({n_fatigued/len(features)*100:.1f}%)")
        
        if n_alert == 0 or n_fatigued == 0:
            print(f"  [EXTREME] ⚠ WARNING: Insufficient extreme windows")
        
    except Exception as e:
        print(f"  [EXTREME] ✗ ERROR: {e}")
        raise
    
    return alert_mask, fatigued_mask


# =============================================================================
# Step 6: Learn Subject-Specific Tiredness Axis
# =============================================================================


def compute_tiredness_projection(
    features: pd.DataFrame,
    alert_mask: np.ndarray,
    fatigued_mask: np.ndarray,
    cfg: TirednessConfig,
    verbose: bool = False,
    baseline_features: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Compute projection of all epochs onto the alert-to-fatigued axis.
    
    Args:
        features: Feature DataFrame (all epochs to project)
        alert_mask: Boolean mask for alert windows (from baseline if provided)
        fatigued_mask: Boolean mask for fatigued windows (from baseline if provided)
        cfg: Tiredness configuration
        verbose: Whether to print verbose output
        baseline_features: If provided, compute axis from this baseline data only
        
    Returns:
        Raw projection scores (n_epochs,)
    """
    if verbose:
        print("  [PROJECT] Computing tiredness projection...")
    
    try:
        # If baseline features are provided, use them to compute the axis
        if baseline_features is not None:
            X_baseline = baseline_features.values
            X_all = features.values
            
            # Check for NaN or Inf
            if np.any(np.isnan(X_baseline)) or np.any(np.isinf(X_baseline)):
                print(f"  [PROJECT] ⚠ WARNING: NaN or Inf in baseline, replacing with 0")
                X_baseline = np.nan_to_num(X_baseline, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.any(np.isnan(X_all)) or np.any(np.isinf(X_all)):
                print(f"  [PROJECT] ⚠ WARNING: NaN or Inf in features, replacing with 0")
                X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute mean feature vectors FROM BASELINE ONLY
            alert_epochs = X_baseline[alert_mask]
            fatigued_epochs = X_baseline[fatigued_mask]
            
            if verbose:
                print(f"  [PROJECT] Using baseline: alert={len(alert_epochs)}, fatigued={len(fatigued_epochs)} epochs")
            
            mu_alert = alert_epochs.mean(axis=0)
            mu_fatigued = fatigued_epochs.mean(axis=0)
            
            # Direction vector (alert -> fatigued)
            v = mu_fatigued - mu_alert
            v_norm_sq = np.dot(v, v)
            
            if v_norm_sq < 1e-10:
                print("  [PROJECT] ⚠ WARNING: Alert and fatigued means are too similar")
                print("  [PROJECT] Returning zero projection - no tiredness variation")
                return np.zeros(len(X_all))
            
            # Project ALL epochs onto this axis
            X_centered = X_all - mu_alert
            scores_raw = np.dot(X_centered, v) / v_norm_sq
            
        else:
            # Original behavior: use all features
            X = features.values
            
            # Check for NaN or Inf
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print(f"  [PROJECT] ⚠ WARNING: NaN or Inf in features, replacing with 0")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute mean feature vectors
            alert_epochs = X[alert_mask]
            fatigued_epochs = X[fatigued_mask]
            
            mu_alert = alert_epochs.mean(axis=0)
            mu_fatigued = fatigued_epochs.mean(axis=0)
            
            # Direction vector (alert -> fatigued)
            v = mu_fatigued - mu_alert
            v_norm_sq = np.dot(v, v)
            
            if v_norm_sq < 1e-10:
                print("  [PROJECT] ⚠ WARNING: Alert and fatigued means are too similar")
                print("  [PROJECT] Returning zero projection - no tiredness variation")
                return np.zeros(len(X))
            
            # Project all epochs onto this axis
            X_centered = X - mu_alert
            scores_raw = np.dot(X_centered, v) / v_norm_sq
        
        # Clip to reasonable range
        scores = np.clip(scores_raw, cfg.clip_min, cfg.clip_max)
        
        if verbose:
            n_clipped = np.sum((scores_raw < cfg.clip_min) | (scores_raw > cfg.clip_max))
            print(f"  [PROJECT] ✓ Projection complete (clipped {n_clipped} outliers)")
        
    except Exception as e:
        print(f"  [PROJECT] ✗ ERROR: {e}")
        raise
    
    return scores


# =============================================================================
# Step 7: Normalize to 0-1 Tiredness Index
# =============================================================================


def normalize_tiredness_index(
    scores: np.ndarray,
    cfg: TirednessConfig,
    verbose: bool = False
) -> np.ndarray:
    """Normalize projection scores to 0-1 tiredness index."""
    if verbose:
        print(f"  [NORMALIZE] Normalizing to 0-1 (method: {cfg.normalize_method})...")
    
    # Check for NaN or Inf
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        print(f"  [NORMALIZE] ⚠ WARNING: NaN or Inf in scores, replacing with 0")
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    try:
        if cfg.normalize_method == "percentile":
            s_min = np.percentile(scores, cfg.percentile_low)
            s_max = np.percentile(scores, cfg.percentile_high)
        else:  # minmax
            s_min = scores.min()
            s_max = scores.max()
        
        # Normalize
        if s_max - s_min < 1e-10:
            print("  [NORMALIZE] ⚠ WARNING: No variation in scores")
            print("  [NORMALIZE] Returning 0.5 for all epochs")
            return np.ones_like(scores) * 0.5
        
        ti = (scores - s_min) / (s_max - s_min)
        ti = np.clip(ti, 0.0, 1.0)
        
        if verbose:
            print(f"  [NORMALIZE] ✓ TI: mean={ti.mean():.3f}, std={ti.std():.3f}")
        
    except Exception as e:
        print(f"  [NORMALIZE] ✗ ERROR: {e}")
        raise
    
    return ti


def smooth_tiredness_index(
    ti: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """Apply moving average smoothing to tiredness index."""
    if window_size <= 1 or len(ti) < window_size:
        return ti
    
    kernel = np.ones(window_size) / window_size
    ti_smooth = np.convolve(ti, kernel, mode='same')
    
    return ti_smooth


# =============================================================================
# Complete Pipeline for Single Subject
# =============================================================================


def process_subject_sessions(
    session_files: List[Path],
    subject_id: str,
    cfg: PipelineConfig,
    verbose: bool = False,
    use_first_session_baseline: bool = True
) -> pd.DataFrame:
    """
    Process all sessions for a single subject and compute tiredness index.
    
    This is the core function that implements SUBJECT-SPECIFIC processing:
    - Processes all sessions for one subject
    - If use_first_session_baseline=True: Uses ONLY first session to define alert/fatigued
    - If use_first_session_baseline=False: Uses all sessions to define alert/fatigued
    - Computes tiredness axis specific to that subject
    - Returns tiredness index normalized to that subject's range
    """
    print(f"\n{'='*60}")
    print(f"[SUBJECT] Processing Subject: {subject_id}")
    print(f"[SUBJECT] Sessions: {len(session_files)}")
    if use_first_session_baseline:
        print(f"[SUBJECT] Mode: Using FIRST SESSION as baseline")
    else:
        print(f"[SUBJECT] Mode: Using ALL SESSIONS for normalization")
    print(f"{'='*60}")
    
    # Store data for each session
    all_epochs_data = []
    session_metadata = []
    
    # Process each session
    for session_idx, filepath in enumerate(session_files, 1):
        _, session_id = extract_subject_session_id(filepath)
        
        print(f"\n[SESSION] {session_idx}/{len(session_files)}: {session_id}")
        
        try:
            # Load and preprocess
            raw = load_raw_edf(filepath, cfg.channels, verbose=verbose)
            raw = preprocess_raw(raw, cfg.preprocess, cfg.channels, verbose=verbose)
            
            # Create epochs
            epochs = create_epochs(raw, cfg.segmentation, verbose=verbose)
            
            if len(epochs) == 0:
                print(f"[SESSION] ✗ No valid epochs - SKIPPING")
                continue
            
            # Extract features
            features = extract_spectral_features(
                epochs,
                cfg.spectral,
                cfg.channels,
                verbose=verbose
            )
            
            # Store for later processing
            all_epochs_data.append(features)
            
            # Compute timing information
            epoch_starts = np.arange(len(epochs)) * cfg.segmentation.overlap_s
            
            session_metadata.append({
                "session_id": session_id,
                "session_number": session_idx,
                "n_epochs": len(epochs),
                "epoch_starts": epoch_starts,
                "filepath": str(filepath)
            })
            
            print(f"[SESSION] ✓ Processed: {len(features)} epochs")
            
        except Exception as e:
            print(f"[SESSION] ✗ ERROR: {e}")
            continue
    
    if not all_epochs_data:
        print(f"\n[SUBJECT] ✗ No valid data for subject {subject_id}")
        return pd.DataFrame()
    
    # Concatenate features from all sessions of THIS SUBJECT ONLY
    print(f"\n{'='*60}")
    print("[ANALYSIS] Computing Subject-Specific Tiredness Index")
    print(f"{'='*60}")
    
    all_features = pd.concat(all_epochs_data, axis=0, ignore_index=True)
    total_epochs = len(all_features)
    
    print(f"[ANALYSIS] Total epochs: {total_epochs}")
    
    # Check if we have enough epochs
    if total_epochs < cfg.tiredness.min_epochs_required:
        print(f"[ANALYSIS] ✗ WARNING: Only {total_epochs} epochs (minimum: {cfg.tiredness.min_epochs_required})")
        print(f"[ANALYSIS] Tiredness index may be unreliable")
    
    # Decide which data to use for identifying extremes
    if use_first_session_baseline and len(all_epochs_data) > 0:
        # Use ONLY the first session to identify alert/fatigued states
        baseline_features = all_epochs_data[0]
        print(f"[ANALYSIS] Using FIRST SESSION ONLY as baseline ({len(baseline_features)} epochs)")
        
        # Identify extreme windows from first session
        alert_mask_baseline, fatigued_mask_baseline = identify_extreme_windows(
            baseline_features,
            cfg.tiredness,
            verbose=verbose
        )
        
        # Compute tiredness projection using first session's alert/fatigued states
        raw_scores = compute_tiredness_projection(
            all_features,  # Apply to ALL sessions
            alert_mask_baseline,
            fatigued_mask_baseline,
            cfg.tiredness,
            verbose=verbose,
            baseline_features=baseline_features  # Pass baseline features
        )
        
        # Create full masks for all epochs (needed for output DataFrame)
        # These indicate which epochs in the baseline were used as references
        n_baseline = len(baseline_features)
        alert_mask = np.zeros(total_epochs, dtype=bool)
        fatigued_mask = np.zeros(total_epochs, dtype=bool)
        alert_mask[:n_baseline] = alert_mask_baseline
        fatigued_mask[:n_baseline] = fatigued_mask_baseline
        
    else:
        # Use all sessions to identify extremes (original behavior)
        print(f"[ANALYSIS] Using ALL SESSIONS for baseline ({len(all_features)} epochs)")
        
        alert_mask, fatigued_mask = identify_extreme_windows(
            all_features,
            cfg.tiredness,
            verbose=verbose
        )
        
        # Compute tiredness projection using THIS SUBJECT's alert/fatigued states
        raw_scores = compute_tiredness_projection(
            all_features,
            alert_mask,
            fatigued_mask,
            cfg.tiredness,
            verbose=verbose
        )
    
    # Normalize to 0-1 using THIS SUBJECT's range
    ti_normalized = normalize_tiredness_index(
        raw_scores,
        cfg.tiredness,
        verbose=verbose
    )
    
    # Split back into sessions and apply smoothing
    print(f"\n[RESULTS] Creating output DataFrame...")
    results = []
    epoch_offset = 0
    
    for meta in session_metadata:
        n_epochs = meta["n_epochs"]
        session_ti = ti_normalized[epoch_offset:epoch_offset + n_epochs]
        
        # Apply temporal smoothing within session
        if cfg.tiredness.smoothing_window > 1:
            session_ti_smooth = smooth_tiredness_index(
                session_ti,
                cfg.tiredness.smoothing_window
            )
        else:
            session_ti_smooth = session_ti
        
        # Create result rows for this session
        for i in range(n_epochs):
            results.append({
                "subject_id": subject_id,
                "session_id": meta["session_id"],
                "file_path": meta["filepath"],
                "epoch_index": i,
                "start_time_sec": meta["epoch_starts"][i],
                "tiredness_index": session_ti_smooth[i],
                "tiredness_index_raw": session_ti[i],
                "projection_score": raw_scores[epoch_offset + i],
                "is_alert_ref": alert_mask[epoch_offset + i],
                "is_fatigued_ref": fatigued_mask[epoch_offset + i]
            })
        
        epoch_offset += n_epochs
    
    result_df = pd.DataFrame(results)
    print(f"[RESULTS] ✓ Created DataFrame: {len(result_df)} rows")
    print(f"[SUBJECT] ✓ Subject {subject_id} complete")
    
    return result_df


# =============================================================================
# Directory Processing - SUBJECT-SPECIFIC VERSION
# =============================================================================


def process_directory(
    input_dir: str,
    output_csv: str,
    depth: int = 1,
    verbose: bool = False,
    cfg: Optional[PipelineConfig] = None,
    exts: Optional[Sequence[str]] = None,
    use_first_session_baseline: bool = True
) -> None:
    """
    Process all EDF files with SUBJECT-SPECIFIC tiredness indices.
    
    Each subject's tiredness index is computed independently:
    - Alert/fatigued states are identified within each subject's data
    - Tiredness axis is specific to each subject's physiology
    - Normalization is done per-subject
    
    Args:
        input_dir: Directory containing EDF files
        output_csv: Path for output CSV
        depth: Directory search depth
        verbose: Enable verbose output
        cfg: Pipeline configuration
        exts: File extensions to search
        use_first_session_baseline: If True, use only first session to define
                                     alert/fatigued states (preserves temporal trends)
    
    This ensures that TI=0 means "alert for this person" and TI=1 means
    "fatigued for this person", not relative to the population.
    """
    if cfg is None:
        cfg = PipelineConfig()
    
    if exts is None:
        exts = [".edf", ".EDF"]
    
    print(f"\n{'='*60}")
    print("Subject-Specific Tiredness Index Pipeline")
    if use_first_session_baseline:
        print("MODE: FIRST SESSION BASELINE (Preserves temporal trends)")
    else:
        print("MODE: ALL SESSIONS BASELINE (Subject-specific normalization)")
    print(f"{'='*60}")
    print(f"[MAIN] Input: {input_dir}")
    print(f"[MAIN] Output: {output_csv}")
    
    # Find all EDF files
    print(f"\n[MAIN] Searching for files...")
    try:
        files = get_file_paths(
            input_path=input_dir,
            return_full_path=True,
            depth=depth,
            EXT_LIST=list(exts)
        )
    except Exception as e:
        print(f"[MAIN] ✗ ERROR during file search: {e}")
        return
    
    if not files:
        print(f"\n[MAIN] ✗ No EDF files found in {input_dir}")
        return
    
    print(f"[MAIN] ✓ Found {len(files)} files")
    
    # Group files by subject
    print(f"\n[MAIN] Grouping by subject...")
    subject_files: Dict[str, List[Path]] = {}
    for filepath in files:
        subject_id = subject_id_from_path(Path(filepath))
        subject_files.setdefault(subject_id, []).append(Path(filepath))
    
    print(f"[MAIN] ✓ Found {len(subject_files)} subjects")
    
    # Sort for consistent processing
    subject_ids = sorted(subject_files.keys())
    
    for sid in subject_ids:
        print(f"[MAIN]   - {sid}: {len(subject_files[sid])} sessions")
    
    # Process each subject INDEPENDENTLY
    all_results = []
    
    for subj_idx, subject_id in enumerate(subject_ids, 1):
        print(f"\n{'#'*60}")
        print(f"[MAIN] Subject {subj_idx}/{len(subject_ids)}: {subject_id}")
        print(f"{'#'*60}")
        
        try:
            # Process this subject's sessions
            result_df = process_subject_sessions(
                session_files=sorted(subject_files[subject_id]),
                subject_id=subject_id,
                cfg=cfg,
                verbose=verbose,
                use_first_session_baseline=use_first_session_baseline
            )
            
            if not result_df.empty:
                all_results.append(result_df)
                print(f"[MAIN] ✓ Subject {subject_id}: {len(result_df)} epochs processed")
            else:
                print(f"[MAIN] ✗ Subject {subject_id}: No valid data")
                
        except Exception as e:
            print(f"[MAIN] ✗ ERROR processing subject {subject_id}: {e}")
            continue
    
    # Combine all results
    if not all_results:
        print("\n[MAIN] ✗ No results to save")
        return
    
    print(f"\n{'='*60}")
    print("[MAIN] Combining results and saving...")
    print(f"{'='*60}")
    
    df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    try:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[MAIN] ✓ Saved to: {output_csv}")
    except Exception as e:
        print(f"[MAIN] ✗ ERROR saving: {e}")
        return
    
    # Print summary
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Output: {output_csv}")
    print(f"Total epochs: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Sessions: {len(df.groupby(['subject_id', 'session_id']))}")
    
    print("\nTiredness Index Summary:")
    print(f"  Mean: {df['tiredness_index'].mean():.3f}")
    print(f"  Std: {df['tiredness_index'].std():.3f}")
    print(f"  Min: {df['tiredness_index'].min():.3f}")
    print(f"  Max: {df['tiredness_index'].max():.3f}")
    
    print("\nPer-Subject Summary:")
    subject_summary = df.groupby("subject_id").agg({
        "session_id": "nunique",
        "epoch_index": "count",
        "tiredness_index": ["mean", "std", "min", "max"]
    })
    subject_summary.columns = ["n_sessions", "n_epochs", "TI_mean", "TI_std", "TI_min", "TI_max"]
    print(subject_summary.to_string())


# =============================================================================
# Command Line Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compute subject-specific tiredness index from EEG (FIXED VERSION)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the FIXED version that computes tiredness indices with proper temporal awareness.

Two baseline modes:

1. FIRST SESSION BASELINE (default, recommended for sleep deprivation studies):
   - Uses ONLY the first session to identify alert/fatigued states
   - Later sessions are projected onto this baseline axis
   - Preserves temporal trends (Session 3 should be higher than Session 1)
   - Best for: sleep deprivation, longitudinal studies

2. ALL SESSIONS BASELINE (use --use-all-sessions-baseline):
   - Uses all sessions to identify extremes
   - Normalizes each subject's range independently
   - May not preserve temporal trends across sessions
   - Best for: cross-sectional comparisons within subject

Pipeline:
1. Load EDF files (5 EEG channels: Fz, Cz, C3, C4, Pz)
2. Preprocess: filter 1-40 Hz, notch 50/60 Hz, re-reference, ICA
3. Segment into 8s windows with 50% overlap
4. Extract spectral features (theta/alpha/beta ratios)
5. FOR EACH SUBJECT INDEPENDENTLY:
   - Identify alert vs fatigued windows (from first session or all sessions)
   - Compute tiredness axis
   - Project all epochs and normalize to 0-1
6. Smooth within sessions

Expected filename: subject_id-session_id.edf (e.g., "1-1.edf")

Example usage:
  # Default (first session baseline)
  python script.py /path/to/data --output results.csv
  
  # Use all sessions for baseline
  python script.py /path/to/data --use-all-sessions-baseline
        """
    )
    
    parser.add_argument("input_dir", type=str, help="Directory containing EDF files")
    parser.add_argument("--output", "-o", type=str, default="tiredness_index_subject_specific.csv",
                       help="Output CSV file (default: tiredness_index_subject_specific.csv)")
    parser.add_argument("--depth", "-d", type=int, default=1,
                       help="Directory search depth (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    # Preprocessing options
    preprocess_group = parser.add_argument_group("Preprocessing")
    preprocess_group.add_argument("--no-ica", action="store_true",
                                 help="Disable ICA artifact removal")
    preprocess_group.add_argument("--notch-freq", type=float, default=50.0,
                                 help="Line noise frequency (default: 50 Hz)")
    preprocess_group.add_argument("--filter-low", type=float, default=1.0,
                                 help="High-pass cutoff (default: 1.0 Hz)")
    preprocess_group.add_argument("--filter-high", type=float, default=40.0,
                                 help="Low-pass cutoff (default: 40.0 Hz)")
    
    # Segmentation options
    segment_group = parser.add_argument_group("Segmentation")
    segment_group.add_argument("--window-length", type=float, default=8.0,
                              help="Window length in seconds (default: 8.0)")
    segment_group.add_argument("--window-overlap", type=float, default=4.0,
                              help="Window overlap in seconds (default: 4.0)")
    segment_group.add_argument("--reject-threshold", type=float, default=150.0,
                              help="Artifact rejection threshold in µV (default: 150.0)")
    
    # Tiredness index options
    tiredness_group = parser.add_argument_group("Tiredness Index")
    tiredness_group.add_argument("--extreme-percentile", type=float, default=10.0,
                                help="Percentile for alert/fatigued (default: 20.0)")
    tiredness_group.add_argument("--smoothing-window", type=int, default=3,
                                help="Smoothing window size (default: 3)")
    tiredness_group.add_argument("--no-smoothing", action="store_true",
                                help="Disable smoothing")
    tiredness_group.add_argument("--min-epochs", type=int, default=10,
                                help="Minimum epochs required per subject (default: 10)")
    tiredness_group.add_argument("--use-all-sessions-baseline", action="store_true",
                                help="Use all sessions for baseline (default: use only first session)")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input_dir).is_dir():
        print(f"[MAIN] ✗ ERROR: Input directory does not exist: {args.input_dir}")
        return
    
    # Build configuration
    cfg = PipelineConfig(
        channels=ChannelConfig(),
        preprocess=PreprocessConfig(
            l_freq=args.filter_low,
            h_freq=args.filter_high,
            notch_freq=args.notch_freq if args.notch_freq > 0 else None,
            apply_ica=not args.no_ica
        ),
        segmentation=SegmentationConfig(
            window_length_s=args.window_length,
            overlap_s=args.window_overlap,
            reject_criteria={'eeg': args.reject_threshold * 1e-6}
        ),
        tiredness=TirednessConfig(
            extreme_percentile=args.extreme_percentile,
            smoothing_window=1 if args.no_smoothing else args.smoothing_window,
            min_epochs_required=args.min_epochs
        )
    )
    
    # Run pipeline
    process_directory(
        input_dir=args.input_dir,
        output_csv=args.output,
        depth=args.depth,
        verbose=args.verbose,
        cfg=cfg,
        use_first_session_baseline=not args.use_all_sessions_baseline
    )


if __name__ == "__main__":
    main()