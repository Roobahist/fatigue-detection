from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file_with_stim_times(
    path: Path,
) -> Tuple[int, int, List[Tuple[float, float, float]]]:
    """
    Returns:
        subject_id, session_id, rows as (stim_time_sec_since_start, rt_sec, stim_abs_seconds)

    stim_abs_seconds is seconds since epoch, used only to preserve exact ordering if needed.
    """
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename format (expected number-number.csv): {path.name}")

    subject_id_str, session_id_str = path.stem.split("-")
    subject_id = int(subject_id_str)
    session_id = int(session_id_str)

    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    session_start = parse_timestamp(lines[0])

    rows: List[Tuple[float, float, float]] = []
    for line in lines[1:]:
        stim_ts, react_ts = line.split(";")
        stim_time = parse_timestamp(stim_ts)
        react_time = parse_timestamp(react_ts)

        stim_since_start = (stim_time - session_start).total_seconds()
        rt = (react_time - stim_time).total_seconds()

        stim_abs = stim_time.timestamp()
        rows.append((float(stim_since_start), float(rt), float(stim_abs)))

    # Sort by stim time
    rows.sort(key=lambda x: x[0])
    return subject_id, session_id, rows


def load_subject_session_rows(
    folder: Path,
) -> Dict[Tuple[int, int], List[Tuple[float, float, float]]]:
    """
    Returns:
        (subject_id, session_id) -> list of (stim_t, rt, stim_abs)
    """
    out: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
    for csv_file in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(csv_file.name):
            continue
        subject_id, session_id, rows = load_session_file_with_stim_times(csv_file)
        if rows:
            out[(subject_id, session_id)] = rows
    return out


def compute_isi_rt_pairs(
    rows: List[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From rows (stim_t, rt, stim_abs), compute:
        isi[i] = stim_t[i] - stim_t[i-1], for i >= 1
        rt[i]  = rt[i], aligned to the same i

    Returns:
        isi, rt_aligned
    """
    stim_t = np.asarray([r[0] for r in rows], dtype=float)
    rt = np.asarray([r[1] for r in rows], dtype=float)

    ok = np.isfinite(stim_t) & np.isfinite(rt)
    stim_t = stim_t[ok]
    rt = rt[ok]

    if stim_t.size < 2:
        return np.asarray([]), np.asarray([])

    isi = stim_t[1:] - stim_t[:-1]
    rt_aligned = rt[1:]
    ok2 = np.isfinite(isi) & np.isfinite(rt_aligned)
    return isi[ok2], rt_aligned[ok2]


def compute_prev_rt_pairs(
    rows: List[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        prev_rt[i] = rt[i-1], rt[i], for i >= 1
    """
    rt = np.asarray([r[1] for r in rows], dtype=float)
    rt = rt[np.isfinite(rt)]
    if rt.size < 2:
        return np.asarray([]), np.asarray([])
    prev = rt[:-1]
    cur = rt[1:]
    ok = np.isfinite(prev) & np.isfinite(cur)
    return prev[ok], cur[ok]


def autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Autocorrelation for lags 0..max_lag.
    Returns array length max_lag+1 with acf[0] = 1 if variance > 0 else 0.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.zeros(max_lag + 1, dtype=float)

    x = x - x.mean()
    var = float(np.dot(x, x))
    if var <= 0.0:
        out = np.zeros(max_lag + 1, dtype=float)
        out[0] = 0.0
        return out

    out = np.zeros(max_lag + 1, dtype=float)
    out[0] = 1.0
    n = x.size
    for lag in range(1, max_lag + 1):
        if lag >= n:
            out[lag] = np.nan
            continue
        num = float(np.dot(x[:-lag], x[lag:]))
        out[lag] = num / var
    return out


def aggregate_acf_by_session(
    subj_sess_rows: Dict[Tuple[int, int], List[Tuple[float, float, float]]],
    max_lag: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    For each session:
      compute per subject-session acf
      aggregate via nanmean and nanquantiles (25, 75)

    Returns:
      session_id -> (mean_acf, band_acf) where band_acf shape is (2, max_lag+1)
    """
    per_session: Dict[int, List[np.ndarray]] = {}
    for (_subj, sess), rows in subj_sess_rows.items():
        rt = np.asarray([r[1] for r in rows], dtype=float)
        acf = autocorr_1d(rt, max_lag=max_lag)
        per_session.setdefault(sess, []).append(acf)

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sess, acfs in per_session.items():
        M = np.vstack(acfs) if acfs else np.zeros((0, max_lag + 1), dtype=float)
        if M.size == 0:
            continue
        mean = np.nanmean(M, axis=0)
        q25 = np.nanquantile(M, 0.25, axis=0)
        q75 = np.nanquantile(M, 0.75, axis=0)
        out[sess] = (mean, np.vstack([q25, q75]))
    return out


def save_stimulus_timing_effect_plots(
    subj_sess_rows: Dict[Tuple[int, int], List[Tuple[float, float, float]]],
    output_dir: Path,
    dpi: int = 300,
    isi_clip_sec: Tuple[float, float] | None = (0.0, 60.0),
    rt_ylim: Tuple[float, float] = (0.1, 1.0),
    max_lag: int = 30,
) -> None:
    """
    Saves:
      1) rt_vs_isi.png
      2) rt_conditional_on_prev_rt.png
      3) rt_autocorrelation.png
    """
    session_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pool pairs per session
    isi_by_session: Dict[int, List[np.ndarray]] = {}
    rtisi_by_session: Dict[int, List[np.ndarray]] = {}
    prev_by_session: Dict[int, List[np.ndarray]] = {}
    cur_by_session: Dict[int, List[np.ndarray]] = {}

    for (_subj, sess), rows in subj_sess_rows.items():
        isi, rt_aligned = compute_isi_rt_pairs(rows)
        if isi.size:
            if isi_clip_sec is not None:
                lo, hi = float(isi_clip_sec[0]), float(isi_clip_sec[1])
                keep = (isi >= lo) & (isi <= hi)
                isi = isi[keep]
                rt_aligned = rt_aligned[keep]
            if isi.size:
                isi_by_session.setdefault(sess, []).append(isi)
                rtisi_by_session.setdefault(sess, []).append(rt_aligned)

        prev_rt, cur_rt = compute_prev_rt_pairs(rows)
        if prev_rt.size:
            prev_by_session.setdefault(sess, []).append(prev_rt)
            cur_by_session.setdefault(sess, []).append(cur_rt)

    # Plot 1: RT vs ISI
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for sess in sorted(isi_by_session.keys()):
        if sess not in session_colors:
            continue
        isi_all = np.concatenate(isi_by_session[sess]) if isi_by_session[sess] else np.asarray([])
        rt_all = np.concatenate(rtisi_by_session[sess]) if rtisi_by_session[sess] else np.asarray([])
        if isi_all.size == 0:
            continue

        ax.scatter(
            isi_all,
            rt_all,
            s=18,
            alpha=0.18,
            color=session_colors[sess],
            label=f"Session {sess}",
        )

        # Optional: binned median line for readability
        try:
            bins = np.quantile(isi_all, np.linspace(0.0, 1.0, 12))
            bins = np.unique(bins)
            if bins.size >= 4:
                centers = []
                meds = []
                for i in range(bins.size - 1):
                    left = bins[i]
                    right = bins[i + 1]
                    mask = (isi_all >= left) & (isi_all < right) if i < bins.size - 2 else (isi_all >= left) & (isi_all <= right)
                    if mask.any():
                        centers.append(0.5 * (left + right))
                        meds.append(float(np.median(rt_all[mask])))
                if centers:
                    ax.plot(np.asarray(centers), np.asarray(meds), linewidth=2.5, color=session_colors[sess], alpha=0.95)
        except Exception:
            pass

    ax.set_xlabel("Inter-stimulus interval ISI (seconds)")
    ax.set_ylabel("Reaction time (seconds)")
    ax.set_ylim(rt_ylim[0], rt_ylim[1])
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "rt_vs_isi.png", dpi=dpi)
    plt.close(fig)

    # Plot 2: Conditional RT given previous RT
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for sess in sorted(prev_by_session.keys()):
        if sess not in session_colors:
            continue
        prev_all = np.concatenate(prev_by_session[sess]) if prev_by_session[sess] else np.asarray([])
        cur_all = np.concatenate(cur_by_session[sess]) if cur_by_session[sess] else np.asarray([])
        if prev_all.size == 0:
            continue

        ax.scatter(
            prev_all,
            cur_all,
            s=18,
            alpha=0.18,
            color=session_colors[sess],
            label=f"Session {sess}",
        )

        # Simple least squares line for summary
        if prev_all.size >= 2 and float(np.var(prev_all)) > 0.0:
            slope, intercept = np.polyfit(prev_all, cur_all, deg=1)
            x_line = np.linspace(prev_all.min(), prev_all.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linewidth=2.5, color=session_colors[sess], alpha=0.95)

    ax.set_xlabel("Previous RT (seconds)")
    ax.set_ylabel("Current RT (seconds)")
    ax.set_xlim(rt_ylim[0], rt_ylim[1])
    ax.set_ylim(rt_ylim[0], rt_ylim[1])
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "rt_conditional_on_prev_rt.png", dpi=dpi)
    plt.close(fig)

    # Plot 3: Autocorrelation of RT series
    acf_by_session = aggregate_acf_by_session(subj_sess_rows, max_lag=max_lag)
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = np.arange(0, max_lag + 1, dtype=int)

    for sess in sorted(acf_by_session.keys()):
        if sess not in session_colors:
            continue
        mean_acf, band = acf_by_session[sess]
        color = session_colors[sess]

        ax.plot(lags, mean_acf, linewidth=2.6, color=color, label=f"Session {sess}")
        ax.fill_between(lags, band[0], band[1], color=color, alpha=0.18, linewidth=0.0)

    ax.axhline(0.0, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Lag (trials)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "rt_autocorrelation.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    output_dir = Path("/home/mojtaba.kananisarche/Zh/plots/rt_stimulus_timing_effects")

    subj_sess_rows = load_subject_session_rows(data_folder)

    save_stimulus_timing_effect_plots(
        subj_sess_rows=subj_sess_rows,
        output_dir=output_dir,
        dpi=300,
        isi_clip_sec=(0.0, 60.0),
        rt_ylim=(0.1, 1.0),
        max_lag=30,
    )


if __name__ == "__main__":
    main()