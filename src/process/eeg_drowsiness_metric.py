from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np

# =============================================================================
# Types / Protocols (Dependency Injection)
# =============================================================================


@dataclass(frozen=True)
class RecordingRef:
    """Lightweight pointer to a recording."""

    recording_id: str
    subject_id: str
    session_id: str  # e.g., "PVT1", "PVT2", "PVT3" or "S1", "S2", "S3"


class EEGDataProvider(Protocol):
    """
    Dataset adapter. Implement this for DROZY or any other dataset.
    Must return data for requested channels aligned in time for the recording segment.
    """

    def load_eeg(
        self,
        ref: RecordingRef,
        channels: Sequence[str],
    ) -> Tuple[np.ndarray, float]:
        """
        Returns:
            X: shape (n_channels, n_samples) in Volts (or consistent units)
            fs: sampling rate in Hz
        Raises:
            KeyError if channels unavailable for this recording.
        """
        ...


class ArtifactMasker(Protocol):
    """
    Optional artifact/quality masker. Keep this simple first: amplitude threshold,
    flatline detection, etc. It operates at epoch level.
    """

    def valid_epoch_mask(
        self,
        epochs: np.ndarray,
        fs: float,
    ) -> np.ndarray:
        """
        Args:
            epochs: shape (n_epochs, n_channels, n_epoch_samples)
        Returns:
            mask: shape (n_epochs,), True where epoch is valid
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class EpochConfig:
    length_s: float
    step_s: float
    detrend: bool = True


@dataclass(frozen=True)
class BandConfig:
    theta: Tuple[float, float] = (4.0, 7.0)
    alpha: Tuple[float, float] = (8.0, 12.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    fmin: float = 0.5
    fmax: float = 30.0


@dataclass(frozen=True)
class BaselineConfig:
    quantile_qb: float = 0.20
    spread: str = "sd"  # "sd" or "mad"
    eps: float = 1e-12


@dataclass(frozen=True)
class SessionLabelConfig:
    quantile: float = 0.90  # p90


@dataclass(frozen=True)
class MetricConfig:
    channels: Tuple[str, ...] = ("Oz", "Pz", "Cz")
    epoch: EpochConfig = EpochConfig(length_s=4.0, step_s=1.0)
    bands: BandConfig = BandConfig()
    baseline: BaselineConfig = BaselineConfig(quantile_qb=0.20, spread="sd")
    session_label: SessionLabelConfig = SessionLabelConfig(quantile=0.90)


# =============================================================================
# Utilities
# =============================================================================


@dataclass(frozen=True)
class SimpleAmplitudeArtifactMasker:
    max_abs_uV: float = 200.0
    min_std_uV: float = 0.5

    def valid_epoch_mask(self, epochs: np.ndarray, fs: float) -> np.ndarray:
        # epochs: (E,C,T), assume Volts -> convert to µV for thresholds
        x_uV = epochs * 1e6
        max_abs = np.max(np.abs(x_uV), axis=(1, 2))
        std = np.std(x_uV, axis=(1, 2))
        return (max_abs <= self.max_abs_uV) & (std >= self.min_std_uV)


def _sliding_epoch_view(
    X: np.ndarray, fs: float, length_s: float, step_s: float
) -> np.ndarray:
    """
    Args:
        X: (n_channels, n_samples)
    Returns:
        epochs: (n_epochs, n_channels, n_epoch_samples)
    """
    n_ch, n = X.shape
    L = int(round(length_s * fs))
    step = int(round(step_s * fs))
    if L <= 0 or step <= 0:
        raise ValueError("Epoch length and step must be positive.")
    if n < L:
        return np.empty((0, n_ch, L), dtype=X.dtype)

    starts = np.arange(0, n - L + 1, step, dtype=int)
    # Create epochs
    epochs = np.stack([X[:, s : s + L] for s in starts], axis=0)
    return epochs


def _detrend_epochwise(epochs: np.ndarray) -> np.ndarray:
    """Remove per-epoch linear trend per channel (simple least squares)."""
    # epochs: (E, C, T)
    E, C, T = epochs.shape
    t = np.linspace(-1.0, 1.0, T, dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T  # (T, 2)
    # Solve per (E,C)
    out = epochs.astype(np.float64, copy=True)
    for e in range(E):
        for c in range(C):
            y = out[e, c]
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            trend = A @ coef
            out[e, c] = y - trend
    return out.astype(epochs.dtype, copy=False)


def _welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal Welch PSD using numpy FFT (no scipy dependency).
    Args:
        x: 1D signal
    Returns:
        freqs, psd
    Notes:
        For production, scipy.signal.welch is preferred, but this is portable.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if nperseg is None:
        nperseg = min(256, n)
    nperseg = int(nperseg)
    if nperseg <= 0:
        raise ValueError("nperseg must be positive.")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be < nperseg.")

    step = nperseg - noverlap
    if n < nperseg:
        # single segment
        segs = [x]
    else:
        starts = range(0, n - nperseg + 1, step)
        segs = [x[s : s + nperseg] for s in starts]

    window = np.hanning(nperseg)
    win_norm = (window**2).sum()

    psd_acc = None
    for seg in segs:
        if seg.shape[0] != nperseg:
            continue
        seg = seg - seg.mean()
        seg_win = seg * window
        fft = np.fft.rfft(seg_win)
        pxx = (np.abs(fft) ** 2) / (fs * win_norm)
        psd_acc = pxx if psd_acc is None else (psd_acc + pxx)

    if psd_acc is None:
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        return freqs, np.zeros_like(freqs)

    psd = psd_acc / max(1, len(segs))
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


def _bandpower(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    lo, hi = band
    idx = (freqs >= lo) & (freqs < hi)
    if not np.any(idx):
        return 0.0
    # Integrate PSD over band (trapezoid)
    return float(np.trapz(psd[idx], freqs[idx]))


def _robust_spread(values: np.ndarray, mode: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if mode == "sd":
        return float(values.std(ddof=1) if values.size > 1 else 0.0)
    if mode == "mad":
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        # Consistent estimator for normal distribution
        return 1.4826 * mad
    raise ValueError(f"Unknown spread mode: {mode}")


# =============================================================================
# Core computations
# =============================================================================


def compute_epoch_index_ratio_of_avg_powers(
    epochs: np.ndarray,
    fs: float,
    bands: BandConfig,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Computes I_raw(e) = mean_theta / (mean_alpha + mean_beta + eps)
    where mean_* are averages across channels of bandpower in that epoch.

    Args:
        epochs: (E, C, T)
    Returns:
        I_raw: (E,)
    """
    E, C, _ = epochs.shape
    out = np.zeros(E, dtype=np.float64)

    for e in range(E):
        theta_sum = 0.0
        alpha_sum = 0.0
        beta_sum = 0.0
        for c in range(C):
            freqs, psd = _welch_psd(epochs[e, c], fs=fs)
            # Restrict to useful band if desired (not strictly required)
            theta_sum += _bandpower(freqs, psd, bands.theta)
            alpha_sum += _bandpower(freqs, psd, bands.alpha)
            beta_sum += _bandpower(freqs, psd, bands.beta)

        theta_mean = theta_sum / C
        alpha_mean = alpha_sum / C
        beta_mean = beta_sum / C
        out[e] = theta_mean / (alpha_mean + beta_mean + eps)
    return out


@dataclass(frozen=True)
class SubjectBaselineStats:
    mu: float
    spread: float  # SD or MAD scale


def fit_subject_baseline(
    indices_raw: np.ndarray,
    qb: float,
    spread_mode: str,
    eps: float = 1e-12,
) -> SubjectBaselineStats:
    """
    Baseline = lowest qb quantile of indices_raw.
    mu computed as mean of baseline set (you can switch to median later if desired).
    """
    x = np.asarray(indices_raw, dtype=np.float64)
    if x.size == 0:
        return SubjectBaselineStats(mu=float("nan"), spread=float("nan"))
    thr = np.quantile(x, qb)
    base = x[x <= thr]
    mu = float(base.mean()) if base.size > 0 else float(x.mean())
    spread = _robust_spread(base if base.size > 1 else x, mode=spread_mode)
    spread = max(spread, eps)
    return SubjectBaselineStats(mu=mu, spread=spread)


def normalize_option_a(
    indices_raw: np.ndarray,
    baseline: SubjectBaselineStats,
) -> np.ndarray:
    """Z_A = (I_raw - mu_s) / spread_s"""
    x = np.asarray(indices_raw, dtype=np.float64)
    return (x - baseline.mu) / baseline.spread


def normalize_option_b(
    indices_raw: np.ndarray,
    baseline_mu: float,
    global_sd_train: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Z_B = (I_raw - mu_s) / global_sd_train"""
    g = max(float(global_sd_train), eps)
    x = np.asarray(indices_raw, dtype=np.float64)
    return (x - float(baseline_mu)) / g


def aggregate_session_p90(values: np.ndarray, q: float = 0.90) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


# =============================================================================
# High-level orchestration
# =============================================================================


@dataclass(frozen=True)
class SessionMetricResult:
    recording_id: str
    subject_id: str
    session_id: str
    y_a: float  # Option A session score (p90)
    y_b: float  # Option B session score (p90) - requires global SD fit
    debug: Dict[str, float]


class DrowsinessMetricComputer:
    """
    Coordinates computing session-level labels from EEG.
    Designed for reuse: data access, artifact logic, and CV fitting are injected.
    """

    def __init__(
        self,
        cfg: MetricConfig,
        provider: EEGDataProvider,
        artifact_masker: Optional[ArtifactMasker] = None,
    ) -> None:
        self.cfg = cfg
        self.provider = provider
        self.artifact_masker = artifact_masker

    def compute_indices_for_recording(
        self, ref: RecordingRef
    ) -> Tuple[np.ndarray, float]:
        X, fs = self.provider.load_eeg(ref, channels=self.cfg.channels)
        epochs = _sliding_epoch_view(
            X,
            fs=fs,
            length_s=self.cfg.epoch.length_s,
            step_s=self.cfg.epoch.step_s,
        )
        if epochs.shape[0] == 0:
            return np.empty((0,), dtype=np.float64), fs

        if self.cfg.epoch.detrend:
            epochs = _detrend_epochwise(epochs)

        if self.artifact_masker is not None:
            mask = self.artifact_masker.valid_epoch_mask(epochs, fs=fs)
            epochs = epochs[mask]

        indices_raw = compute_epoch_index_ratio_of_avg_powers(
            epochs=epochs,
            fs=fs,
            bands=self.cfg.bands,
            eps=self.cfg.baseline.eps,
        )
        return indices_raw, fs

    def fit_subject_baselines(
        self,
        refs: Sequence[RecordingRef],
    ) -> Dict[str, SubjectBaselineStats]:
        """
        Fit mu_s and spread_s per subject using all provided refs for that subject.
        In CV, pass only training refs for baselines if you want strictness.
        """
        by_subject: Dict[str, List[np.ndarray]] = {}
        for ref in refs:
            idx_raw, _ = self.compute_indices_for_recording(ref)
            by_subject.setdefault(ref.subject_id, []).append(idx_raw)

        baselines: Dict[str, SubjectBaselineStats] = {}
        for sid, chunks in by_subject.items():
            x = (
                np.concatenate([c for c in chunks if c.size > 0], axis=0)
                if chunks
                else np.empty((0,))
            )
            baselines[sid] = fit_subject_baseline(
                indices_raw=x,
                qb=self.cfg.baseline.quantile_qb,
                spread_mode=self.cfg.baseline.spread,
                eps=self.cfg.baseline.eps,
            )
        return baselines

    def fit_global_sd_train(
        self,
        refs_train: Sequence[RecordingRef],
        baselines: Mapping[str, SubjectBaselineStats],
    ) -> float:
        """
        Option B: global SD computed from training data only.
        Uses baseline-centered indices: D = I_raw - mu_s.
        """
        centered_all: List[np.ndarray] = []
        for ref in refs_train:
            idx_raw, _ = self.compute_indices_for_recording(ref)
            if idx_raw.size == 0:
                continue
            mu_s = baselines[ref.subject_id].mu
            centered_all.append(idx_raw - mu_s)

        if not centered_all:
            return float("nan")

        D = np.concatenate(centered_all, axis=0).astype(np.float64)
        # ddof=1 makes it a sample SD
        sd = float(D.std(ddof=1) if D.size > 1 else D.std(ddof=0))
        return max(sd, self.cfg.baseline.eps)

    def compute_session_scores(
        self,
        ref: RecordingRef,
        baselines: Mapping[str, SubjectBaselineStats],
        global_sd_train: Optional[float],
    ) -> SessionMetricResult:
        idx_raw, _ = self.compute_indices_for_recording(ref)
        b = baselines[ref.subject_id]

        z_a = normalize_option_a(idx_raw, baseline=b)
        y_a = aggregate_session_p90(z_a, q=self.cfg.session_label.quantile)

        if global_sd_train is None or not np.isfinite(global_sd_train):
            y_b = float("nan")
        else:
            z_b = normalize_option_b(
                idx_raw,
                baseline_mu=b.mu,
                global_sd_train=global_sd_train,
                eps=self.cfg.baseline.eps,
            )
            y_b = aggregate_session_p90(z_b, q=self.cfg.session_label.quantile)

        debug = {
            "n_epochs": float(idx_raw.size),
            "baseline_mu": float(b.mu),
            "baseline_spread": float(b.spread),
            "global_sd_train": (
                float(global_sd_train) if global_sd_train is not None else float("nan")
            ),
        }

        return SessionMetricResult(
            recording_id=ref.recording_id,
            subject_id=ref.subject_id,
            session_id=ref.session_id,
            y_a=y_a,
            y_b=y_b,
            debug=debug,
        )


def run_fold(
    metric: DrowsinessMetricComputer,
    train_refs: Sequence[RecordingRef],
    test_refs: Sequence[RecordingRef],
) -> Tuple[List[SessionMetricResult], List[SessionMetricResult]]:
    # Baselines: choose your strictness.
    # Recommended for your label-construction research phase:
    #   - baseline per subject computed from that subject's available refs within the fold group.
    # Strict CV (harder): compute baselines using only each subject's training sessions.
    baselines_train = metric.fit_subject_baselines(train_refs)

    # Fit global SD using training refs only (Option B)
    global_sd_train = metric.fit_global_sd_train(train_refs, baselines=baselines_train)

    # Compute train session labels (optional if you want to store them)
    train_results = [
        metric.compute_session_scores(
            ref, baselines=baselines_train, global_sd_train=global_sd_train
        )
        for ref in train_refs
    ]

    # For test subjects:
    # - Option A and B both require mu_s for that subject.
    # If test subjects are disjoint (subject-wise CV), baselines_train won't contain them.
    # In that case, compute baselines for test subjects from their own available recordings (label construction).
    baselines_test = metric.fit_subject_baselines(test_refs)

    test_results = [
        metric.compute_session_scores(
            ref, baselines=baselines_test, global_sd_train=global_sd_train
        )
        for ref in test_refs
    ]
    return train_results, test_results


import mne


class MNEFileProvider:
    def __init__(self, path_resolver: Callable[[RecordingRef], str]) -> None:
        self.path_resolver = path_resolver

    def load_eeg(
        self, ref: RecordingRef, channels: Sequence[str]
    ) -> Tuple[np.ndarray, float]:
        path = self.path_resolver(ref)
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.pick_channels(list(channels))
        fs = float(raw.info["sfreq"])
        X = raw.get_data()  # (n_channels, n_samples)
        return X, fs
