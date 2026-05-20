"""
EEG-based drowsiness label (session-level, one score per recording file)
- Input: directory of PREPROCESSED files (recommended: .fif / .fif.gz saved by your preprocessing step)
- Channels: configurable (default: Fz, Pz, Cz)
- Epoching: sliding window (length_s, step_s)
- Index: (theta + alpha) / (beta + eps)  (kept as in your current code)
- Baseline: lowest q_b quantile of I_raw (within subject scope; by default per-subject across files)
- Option A: (I_raw - mu_s) / spread_s   (SD or MAD)
- Option B: (I_raw - mu_s) / global_sd_train   (global_sd_train computed from baseline-only epochs)

This file is written to be reusable:
- You can inject how to extract subject_id from file paths.
- You can inject how to load recordings (default: read_raw_fif).
- You can keep baseline per-file or per-subject.

Important:
- All signal preprocessing already done elsewhere must NOT be repeated here.
- The only "cleaning" done here is optional epoch quality filtering (amplitude and flatness),
  because you explicitly said you may handle epoch dropping here.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd

from Zh.src.utils.file_folder import get_file_paths

# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class EpochConfig:
    length_s: float = 4.0
    step_s: float = 1.0
    detrend: bool = True


@dataclass(frozen=True)
class BandConfig:
    theta: Tuple[float, float] = (4.0, 7.0)
    alpha: Tuple[float, float] = (8.0, 12.0)
    beta: Tuple[float, float] = (13.0, 30.0)


@dataclass(frozen=True)
class EpochRejectConfig:
    """
    Epoch-level quality gating (optional).
    Units are microvolts for max_abs, and microvolts for min_std.
    Set enable=False to skip rejection entirely.
    """
    enable: bool = True
    max_abs_uV: float = 150.0
    min_std_uV: float = 0.5


@dataclass(frozen=True)
class BaselineConfig:
    quantile_qb: float = 0.10
    spread: str = "mad"  # "sd" or "mad"
    eps: float = 1e-12


@dataclass(frozen=True)
class SessionAggConfig:
    """
    Aggregation over epoch-level normalized scores to produce a session-level label.
    """
    method: str = "quantile"  # "quantile", "mean", "median", "trimmed_mean", "winsor_mean"
    q: float = 0.90           # used if method="quantile"
    trim_frac: float = 0.10   # used if method in {"trimmed_mean","winsor_mean"}


@dataclass(frozen=True)
class MetricConfig:
    selected_eeg_channels: Tuple[str, ...] = ("Fz", "Pz", "Cz")
    epoch: EpochConfig = EpochConfig()
    bands: BandConfig = BandConfig()
    epoch_reject: EpochRejectConfig = EpochRejectConfig()
    baseline: BaselineConfig = BaselineConfig()
    session_label: SessionAggConfig = SessionAggConfig()


@dataclass(frozen=True)
class BaselineStats:
    mu: float
    spread: float  # SD or MAD (scaled MAD)


# =============================================================================
# Core DSP helpers
# =============================================================================


def _epoch_signal(X: np.ndarray, fs: float, length_s: float, step_s: float) -> np.ndarray:
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
    epochs = np.stack([X[:, s : s + L] for s in starts], axis=0)
    return epochs


def _detrend_epochs(epochs: np.ndarray) -> np.ndarray:
    """
    Remove a per-epoch linear trend per channel.
    epochs: (E, C, T)
    """
    E, C, T = epochs.shape
    t = np.linspace(-1.0, 1.0, T, dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T  # (T, 2)

    out = epochs.astype(np.float64, copy=True)
    for e in range(E):
        for c in range(C):
            y = out[e, c]
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            out[e, c] = y - (A @ coef)
    return out.astype(epochs.dtype, copy=False)


def _simple_epoch_quality_mask(epochs: np.ndarray, max_abs_uV: float, min_std_uV: float) -> np.ndarray:
    """
    Basic artifact mask based on amplitude and variance.
    """
    x_uV = epochs * 1e6
    max_abs = np.max(np.abs(x_uV), axis=(1, 2))
    std = np.std(x_uV, axis=(1, 2))
    return (max_abs <= max_abs_uV) & (std >= min_std_uV)


def _welch_psd_numpy(
    x: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
):
    """
    Minimal Welch PSD using numpy FFT.
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
        segs = [x]
    else:
        segs = [x[s : s + nperseg] for s in range(0, n - nperseg + 1, step)]

    window = np.hanning(nperseg)
    win_norm = (window**2).sum()

    psd_acc = None
    used = 0
    for seg in segs:
        if seg.shape[0] != nperseg:
            continue
        seg = seg - seg.mean()
        seg_win = seg * window
        fft = np.fft.rfft(seg_win)
        pxx = (np.abs(fft) ** 2) / (fs * win_norm)
        psd_acc = pxx if psd_acc is None else (psd_acc + pxx)
        used += 1

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    if psd_acc is None or used == 0:
        return freqs, np.zeros_like(freqs)
    return freqs, psd_acc / used


def _bandpower(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    lo, hi = band
    idx = (freqs >= lo) & (freqs < hi)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def compute_index_ratio_of_avg_powers(
    epochs: np.ndarray,
    fs: float,
    bands: BandConfig,
    eps: float,
) -> np.ndarray:
    """
    Ratio of averaged band powers across channels.

    NOTE: This is kept exactly consistent with your current implementation:
      I_raw(e) = (theta_mean + alpha_mean) / (beta_mean + eps)

    epochs: (E, C, T)
    returns: (E,)
    """
    E, C, _ = epochs.shape
    out = np.zeros(E, dtype=np.float64)

    for e in range(E):
        theta_sum = 0.0
        alpha_sum = 0.0
        beta_sum = 0.0
        for c in range(C):
            freqs, psd = _welch_psd_numpy(epochs[e, c], fs=fs)
            theta_sum += _bandpower(freqs, psd, bands.theta)
            alpha_sum += _bandpower(freqs, psd, bands.alpha)
            beta_sum += _bandpower(freqs, psd, bands.beta)

        theta_mean = theta_sum / C
        alpha_mean = alpha_sum / C
        beta_mean = beta_sum / C
        out[e] = (theta_mean) / (alpha_mean + beta_mean + eps)

    return out


def _spread(values: np.ndarray, mode: str) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    if mode == "sd":
        return float(v.std(ddof=1) if v.size > 1 else 0.0)
    if mode == "mad":
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        return 1.4826 * mad
    raise ValueError(f"Unknown spread mode: {mode}")


def fit_baseline_from_indices(indices_raw: np.ndarray, qb: float, spread_mode: str, eps: float) -> BaselineStats:
    x = np.asarray(indices_raw, dtype=np.float64)
    if x.size == 0:
        return BaselineStats(mu=float("nan"), spread=float("nan"))
    thr = float(np.quantile(x, qb))
    base = x[x <= thr]
    mu = float(base.mean()) if base.size > 0 else float(x.mean())
    sp = _spread(base if base.size > 1 else x, mode=spread_mode)
    sp = max(float(sp), eps)
    return BaselineStats(mu=mu, spread=sp)


def normalize_option_a(indices_raw: np.ndarray, baseline: BaselineStats) -> np.ndarray:
    x = np.asarray(indices_raw, dtype=np.float64)
    return (x - baseline.mu) / baseline.spread


def normalize_option_b(indices_raw: np.ndarray, baseline_mu: float, global_sd_train: float, eps: float) -> np.ndarray:
    x = np.asarray(indices_raw, dtype=np.float64)
    g = max(float(global_sd_train), eps)
    return (x - float(baseline_mu)) / g


def aggregate_epoch_scores(z: np.ndarray, cfg: SessionAggConfig) -> float:
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return float("nan")

    m = cfg.method.lower()

    if m == "quantile":
        return float(np.quantile(z, cfg.q))
    if m == "mean":
        return float(z.mean())
    if m == "median":
        return float(np.median(z))

    if m == "trimmed_mean":
        a = float(cfg.trim_frac)
        a = min(max(a, 0.0), 0.49)
        lo = np.quantile(z, a)
        hi = np.quantile(z, 1.0 - a)
        zz = z[(z >= lo) & (z <= hi)]
        return float(zz.mean()) if zz.size else float(z.mean())

    if m == "winsor_mean":
        a = float(cfg.trim_frac)
        a = min(max(a, 0.0), 0.49)
        lo = np.quantile(z, a)
        hi = np.quantile(z, 1.0 - a)
        zw = np.clip(z, lo, hi)
        return float(zw.mean())

    raise ValueError(f"Unknown SessionAggConfig.method: {cfg.method}")


# =============================================================================
# Recording-level orchestration (preprocessed input)
# =============================================================================


def default_loader_preprocessed(path: Path, *, verbose: bool = False) -> mne.io.BaseRaw:
    """
    Default loader for preprocessed recordings saved in FIF format.
    """
    return mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR" if not verbose else None)


def compute_I_raw_from_preprocessed(
    recording_path: str | Path,
    cfg: MetricConfig,
    *,
    loader: Callable[[Path], mne.io.BaseRaw],
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    PREPROCESSED file -> pick channels -> epoch -> optional epoch reject -> I_raw.

    Returns:
        (I_raw, fs)
    """
    recording_path = Path(recording_path)

    raw = loader(recording_path)
    fs = float(raw.info["sfreq"])

    # Pick channels only (no filtering, no notch, no reref, no ICA here)
    wanted = list(cfg.selected_eeg_channels)
    present = set(raw.ch_names)
    missing = [ch for ch in wanted if ch not in present]
    if missing:
        raise RuntimeError(f"Missing required channels in {recording_path.name}: {missing}")

    raw.pick(wanted)
    X = raw.get_data()  # (C, N)

    epochs = _epoch_signal(X, fs=fs, length_s=cfg.epoch.length_s, step_s=cfg.epoch.step_s)
    if epochs.shape[0] == 0:
        return np.empty((0,), dtype=np.float64), fs

    if cfg.epoch.detrend:
        epochs = _detrend_epochs(epochs)

    if cfg.epoch_reject.enable:
        mask = _simple_epoch_quality_mask(
            epochs,
            max_abs_uV=cfg.epoch_reject.max_abs_uV,
            min_std_uV=cfg.epoch_reject.min_std_uV,
        )
        epochs = epochs[mask]

    if epochs.shape[0] == 0:
        return np.empty((0,), dtype=np.float64), fs

    I_raw = compute_index_ratio_of_avg_powers(
        epochs=epochs,
        fs=fs,
        bands=cfg.bands,
        eps=cfg.baseline.eps,
    )
    return I_raw.astype(np.float64), fs


def fit_subject_baselines(
    recording_paths: Sequence[str | Path],
    cfg: MetricConfig,
    subject_id_from_path: Callable[[Path], str],
    *,
    loader: Callable[[Path], mne.io.BaseRaw],
    verbose: bool = False,
) -> Dict[str, BaselineStats]:
    """
    Fit one baseline (mu + spread) per subject using concatenated I_raw from all that subject's files.
    """
    by_subj: Dict[str, List[Path]] = {}
    for p in map(Path, recording_paths):
        by_subj.setdefault(subject_id_from_path(p), []).append(p)

    out: Dict[str, BaselineStats] = {}
    for sid, files in by_subj.items():
        all_raw: List[np.ndarray] = []
        for fp in files:
            I_raw, _ = compute_I_raw_from_preprocessed(fp, cfg, loader=loader, verbose=verbose)
            if I_raw.size:
                all_raw.append(I_raw)

        if not all_raw:
            out[sid] = BaselineStats(mu=float("nan"), spread=float("nan"))
            continue

        cat = np.concatenate(all_raw, axis=0)
        out[sid] = fit_baseline_from_indices(
            cat,
            qb=cfg.baseline.quantile_qb,
            spread_mode=cfg.baseline.spread,
            eps=cfg.baseline.eps,
        )

    return out


def compute_global_sd_from_baseline_epochs(
    recording_paths: Sequence[str | Path],
    cfg: MetricConfig,
    subject_id_from_path: Callable[[Path], str],
    subject_baselines: Dict[str, BaselineStats],
    *,
    loader: Callable[[Path], mne.io.BaseRaw],
    sd_mode: str = "subject_baseline",  # "subject_baseline" or "global_baseline"
    verbose: bool = False,
) -> float:
    """
    Compute global SD for Option B using baseline-only epochs, after centering by subject mu.

    sd_mode:
      - "subject_baseline": per file, take the lowest q_b epochs (within file), center by subject mu,
                            aggregate across all files to compute SD.
      - "global_baseline": center all epochs by subject mu, then take globally lowest q_b epochs to compute SD.
    """
    paths = [Path(p) for p in recording_paths]
    centered_baseline_chunks: List[np.ndarray] = []

    if sd_mode == "subject_baseline":
        for fp in paths:
            sid = subject_id_from_path(fp)
            stats = subject_baselines.get(sid, BaselineStats(mu=float("nan"), spread=float("nan")))
            if not np.isfinite(stats.mu):
                continue

            I_raw, _ = compute_I_raw_from_preprocessed(fp, cfg, loader=loader, verbose=verbose)
            if I_raw.size == 0:
                continue

            thr = float(np.quantile(I_raw, cfg.baseline.quantile_qb))
            base = I_raw[I_raw <= thr]
            if base.size == 0:
                continue

            centered_baseline_chunks.append((base - stats.mu).astype(np.float64))

    elif sd_mode == "global_baseline":
        centered_all: List[np.ndarray] = []
        for fp in paths:
            sid = subject_id_from_path(fp)
            stats = subject_baselines.get(sid, BaselineStats(mu=float("nan"), spread=float("nan")))
            if not np.isfinite(stats.mu):
                continue

            I_raw, _ = compute_I_raw_from_preprocessed(fp, cfg, loader=loader, verbose=verbose)
            if I_raw.size == 0:
                continue

            centered_all.append((I_raw - stats.mu).astype(np.float64))

        if not centered_all:
            return float("nan")

        D = np.concatenate(centered_all, axis=0)
        thr = float(np.quantile(D, cfg.baseline.quantile_qb))
        base = D[D <= thr]
        if base.size == 0:
            return float("nan")
        centered_baseline_chunks = [base]

    else:
        raise ValueError("sd_mode must be 'subject_baseline' or 'global_baseline'.")

    if not centered_baseline_chunks:
        return float("nan")

    B = np.concatenate(centered_baseline_chunks, axis=0)
    sd = float(B.std(ddof=1) if B.size > 1 else B.std(ddof=0))
    return max(sd, cfg.baseline.eps)


def compute_session_label_from_preprocessed(
    recording_path: str | Path,
    cfg: MetricConfig,
    *,
    loader: Callable[[Path], mne.io.BaseRaw],
    global_sd_train: Optional[float] = None,
    baseline_stats: Optional[BaselineStats] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Computes one session-level label for this preprocessed recording file.

    By default, baseline is fitted from this file's own epochs (lowest q_b of I_raw).
    If you want subject-level baseline pooled across multiple sessions, pass baseline_stats.
    """
    recording_path = Path(recording_path)

    I_raw, _ = compute_I_raw_from_preprocessed(recording_path, cfg, loader=loader, verbose=verbose)

    if I_raw.size == 0:
        return {
            "y_a": float("nan"),
            "y_b": float("nan"),
            "n_epochs": 0.0,
            "baseline_mu": float("nan"),
            "baseline_spread": float("nan"),
            "global_sd_train": (float(global_sd_train) if global_sd_train is not None else float("nan")),
        }

    if baseline_stats is None:
        baseline_stats = fit_baseline_from_indices(
            I_raw,
            qb=cfg.baseline.quantile_qb,
            spread_mode=cfg.baseline.spread,
            eps=cfg.baseline.eps,
        )

    Z_a = normalize_option_a(I_raw, baseline_stats)
    y_a = aggregate_epoch_scores(Z_a, cfg.session_label)

    if global_sd_train is None or not np.isfinite(global_sd_train):
        y_b = float("nan")
    else:
        Z_b = normalize_option_b(
            I_raw,
            baseline_mu=baseline_stats.mu,
            global_sd_train=global_sd_train,
            eps=cfg.baseline.eps,
        )
        y_b = aggregate_epoch_scores(Z_b, cfg.session_label)

    return {
        "y_a": float(y_a),
        "y_b": float(y_b),
        "n_epochs": float(I_raw.size),
        "baseline_mu": float(baseline_stats.mu),
        "baseline_spread": float(baseline_stats.spread),
        "global_sd_train": (float(global_sd_train) if global_sd_train is not None else float("nan")),
    }


# =============================================================================
# Path parsing (default DROZY: "subject-session.ext")
# =============================================================================


def extract_subject_session_id(filepath: str | Path) -> tuple[str, str]:
    """
    Extract subject_id and session_id from filename.
    Expected format: "subject_id-session_id.ext"
    """
    filename = Path(filepath).stem
    parts = filename.split("-", 1)

    if len(parts) == 2:
        subject_id = parts[0].strip()
        session_id = parts[1].strip()
    else:
        subject_id = filename
        session_id = "unknown"

    return subject_id, session_id


def subject_id_from_path(filepath: Path) -> str:
    subject_id, _ = extract_subject_session_id(filepath)
    return subject_id


# =============================================================================
# Directory processing
# =============================================================================


def process_preprocessed_directory(
    input_dir: str,
    output_csv: str,
    depth: int = 1,
    baseline_scope: str = "subject",
    verbose: bool = False,
    *,
    cfg: Optional[MetricConfig] = None,
    loader: Optional[Callable[[Path], mne.io.BaseRaw]] = None,
    subject_id_fn: Optional[Callable[[Path], str]] = None,
    exts: Optional[Sequence[str]] = None,
) -> None:
    """
    Process all preprocessed files in a directory and compute drowsiness labels.

    Customization points:
    - cfg: MetricConfig (channels, epoching, bands, baseline, aggregation, epoch rejection)
    - loader: how to load a file into MNE Raw (default: read_raw_fif)
    - subject_id_fn: subject id extraction from Path (default: subject_id_from_path)
    - exts: which file extensions to include (default: [".fif", ".fif.gz"])
    """
    if cfg is None:
        cfg = MetricConfig(
            selected_eeg_channels=("Fz", "Pz", "Cz"),
            epoch=EpochConfig(length_s=4.0, step_s=1.0, detrend=True),
            epoch_reject=EpochRejectConfig(enable=True, max_abs_uV=150.0, min_std_uV=0.5),
            baseline=BaselineConfig(quantile_qb=0.20, spread="sd"),
            session_label=SessionAggConfig(method="quantile", q=0.90, trim_frac=0.10),
        )

    if loader is None:
        def _ld(p: Path) -> mne.io.BaseRaw:
            return default_loader_preprocessed(p, verbose=verbose)
        loader = _ld

    if subject_id_fn is None:
        subject_id_fn = subject_id_from_path

    if exts is None:
        exts = [".fif", ".fif.gz", ".FIF", ".FIF.GZ"]

    print(f"Searching for preprocessed files in: {input_dir}")
    print(f"Search depth: {depth}")
    print(f"Extensions: {list(exts)}")

    files = get_file_paths(
        input_path=input_dir,
        return_full_path=True,
        depth=depth,
        EXT_LIST=list(exts),
    )

    if not files:
        print("No preprocessed files found!")
        return

    print(f"Found {len(files)} files")

    file_info = []
    subjects_found = set()
    for fp in files:
        subject_id, session_id = extract_subject_session_id(fp)
        file_info.append({"path": fp, "subject_id": subject_id, "session_id": session_id})
        subjects_found.add(subject_id)

    print(f"Found {len(subjects_found)} unique subjects: {sorted(subjects_found)}")

    subject_baselines: Optional[Dict[str, BaselineStats]] = None

    if baseline_scope == "subject":
        print("\nFitting subject-level baseline stats (mu/spread) ...")
        subject_baselines = fit_subject_baselines(
            recording_paths=files,
            cfg=cfg,
            subject_id_from_path=subject_id_fn,
            loader=loader,
            verbose=verbose,
        )

        sd_mode = "subject_baseline"
        print(f"\nComputing global SD from baseline epochs (mode={sd_mode}) ...")
        global_sd_train = compute_global_sd_from_baseline_epochs(
            recording_paths=files,
            cfg=cfg,
            subject_id_from_path=subject_id_fn,
            subject_baselines=subject_baselines,
            loader=loader,
            sd_mode=sd_mode,
            verbose=verbose,
        )
    elif baseline_scope == "file":
        subject_baselines = None
        print("\nComputing global SD from per-file baselines (file scope) ...")

        centered_all: List[np.ndarray] = []
        for fp in map(Path, files):
            I_raw, _ = compute_I_raw_from_preprocessed(fp, cfg, loader=loader, verbose=verbose)
            if I_raw.size == 0:
                continue
            stats = fit_baseline_from_indices(
                I_raw,
                qb=cfg.baseline.quantile_qb,
                spread_mode=cfg.baseline.spread,
                eps=cfg.baseline.eps,
            )
            if not np.isfinite(stats.mu):
                continue
            centered_all.append(I_raw - stats.mu)

        if centered_all:
            D = np.concatenate(centered_all, axis=0).astype(np.float64)
            sd = float(D.std(ddof=1) if D.size > 1 else D.std(ddof=0))
            global_sd_train = max(sd, cfg.baseline.eps)
        else:
            global_sd_train = float("nan")
    else:
        raise ValueError("baseline_scope must be 'file' or 'subject'.")

    print(f"Global SD (train): {float(global_sd_train):.6f}")

    print("\nComputing session-level drowsiness labels for each recording...")

    rows = []
    for i, info in enumerate(file_info, 1):
        fp = Path(info["path"])
        sid = info["subject_id"]
        session_id = info["session_id"]

        if verbose:
            print(f"  [{i}/{len(file_info)}] Processing {sid}-{session_id} ...")

        baseline_stats = None
        if baseline_scope == "subject" and subject_baselines is not None:
            baseline_stats = subject_baselines.get(sid, BaselineStats(mu=float("nan"), spread=float("nan")))

        res = compute_session_label_from_preprocessed(
            fp,
            cfg,
            loader=loader,
            global_sd_train=global_sd_train,
            baseline_stats=baseline_stats,
            verbose=verbose,
        )

        rows.append(
            {
                "subject_id": sid,
                "session_id": session_id,
                "file_path": str(fp),
                "recording_id": fp.stem,
                "y_a": res["y_a"],
                "y_b": res["y_b"],
                "n_epochs": res["n_epochs"],
                "baseline_mu": res["baseline_mu"],
                "baseline_spread": res["baseline_spread"],
                "global_sd_train": res["global_sd_train"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_id", "session_id"]).reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nResults written to: {output_csv}")
    print(f"\nProcessed {len(df)} recordings from {len(subjects_found)} subjects")

    print("\nOutput columns:")
    print("  - subject_id: Subject identifier (extracted from filename)")
    print("  - session_id: Session identifier (extracted from filename)")
    print("  - file_path: Full path to the preprocessed file")
    print("  - recording_id: Filename stem (subject_id-session_id)")
    print("  - y_a: Option A normalized drowsiness score (aggregated over epochs)")
    print("  - y_b: Option B normalized drowsiness score (aggregated over epochs)")
    print("  - n_epochs: Number of valid epochs used")
    print("  - baseline_mu: Baseline mean (subject-level if baseline_scope=subject)")
    print("  - baseline_spread: Baseline spread (SD or MAD)")
    print("  - global_sd_train: Global SD used for Option B")

    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"Option A (y_a) - Mean: {df['y_a'].mean():.4f}, Std: {df['y_a'].std():.4f}")
    print(f"Option B (y_b) - Mean: {df['y_b'].mean():.4f}, Std: {df['y_b'].std():.4f}")
    print(f"Average epochs per recording: {df['n_epochs'].mean():.1f}")

    print("\nPer-Subject Summary:")
    print("-" * 60)
    subject_summary = (
        df.groupby("subject_id")
        .agg({"session_id": "count", "y_a": "mean", "y_b": "mean", "n_epochs": "mean"})
        .rename(columns={"session_id": "n_sessions"})
    )
    print(subject_summary.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Compute EEG-based drowsiness labels for preprocessed files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Filename format expected (default parser): subject_id-session_id.fif
Example: 1-1_preprocessed.fif

The script will:
1. Find all preprocessed files in the input directory
2. Extract subject_id and session_id from filenames
3. Fit subject baselines (if baseline_scope=subject)
4. Compute global SD (Option B) from baseline-only epochs
5. Generate drowsiness labels for each recording
        """,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing preprocessed files (recommended: .fif / .fif.gz).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="drowsiness_labels.csv",
        help="Output CSV file path (default: drowsiness_labels.csv)",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=1,
        help="Maximum directory depth to search (default: 1)",
    )
    parser.add_argument(
        "--baseline-scope",
        type=str,
        choices=["file", "subject"],
        default="subject",
        help="Baseline scope: 'file' (per-session) or 'subject' (per-subject, default)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    # Minimal, safe knobs that do not reintroduce preprocessing.
    parser.add_argument("--epoch-length", type=float, default=8.0)
    parser.add_argument("--epoch-step", type=float, default=2.0)
    parser.add_argument("--no-detrend", action="store_true", help="Disable epoch detrending")

    parser.add_argument("--reject-epochs", action="store_true", help="Enable epoch rejection")
    parser.add_argument("--reject-max-abs-uv", type=float, default=200.0)
    parser.add_argument("--reject-min-std-uv", type=float, default=0.5)

    parser.add_argument("--baseline-qb", type=float, default=0.20)
    parser.add_argument("--baseline-spread", type=str, choices=["sd", "mad"], default="mad")

    parser.add_argument(
        "--session-agg",
        type=str,
        choices=["quantile", "mean", "median", "trimmed_mean", "winsor_mean"],
        default="quantile",
    )
    parser.add_argument("--session-q", type=float, default=0.90)
    parser.add_argument("--session-trim-frac", type=float, default=0.10)

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return

    cfg = MetricConfig(
        selected_eeg_channels=("Fz", "Pz", "Cz", "C3", "C4"),
        epoch=EpochConfig(
            length_s=float(args.epoch_length),
            step_s=float(args.epoch_step),
            detrend=(not bool(args.no_detrend)),
        ),
        epoch_reject=EpochRejectConfig(
            enable=bool(args.reject_epochs),
            max_abs_uV=float(args.reject_max_abs_uv),
            min_std_uV=float(args.reject_min_std_uv),
        ),
        baseline=BaselineConfig(
            quantile_qb=float(args.baseline_qb),
            spread=str(args.baseline_spread),
        ),
        session_label=SessionAggConfig(
            method=str(args.session_agg),
            q=float(args.session_q),
            trim_frac=float(args.session_trim_frac),
        ),
    )

    process_preprocessed_directory(
        input_dir=args.input_dir,
        output_csv=args.output,
        depth=args.depth,
        baseline_scope=args.baseline_scope,
        verbose=args.verbose,
        cfg=cfg,
        loader=None,          # default: read_raw_fif
        subject_id_fn=None,   # default: subject_id_from_path
        exts=None,            # default: fif/fif.gz
    )


if __name__ == "__main__":
    main()