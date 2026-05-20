from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, int, List[Tuple[float, float]]]:
    """
    Returns:
        subject_id, session_id, points where points are (time_since_start_sec, reaction_time_sec)
    """
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename format (expected number-number.csv): {path.name}")

    subject_id_str, session_id_str = path.stem.split("-")
    subject_id = int(subject_id_str)
    session_id = int(session_id_str)

    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    session_start = parse_timestamp(lines[0])

    points: List[Tuple[float, float]] = []
    for line in lines[1:]:
        stim_ts, react_ts = line.split(";")
        stim_time = parse_timestamp(stim_ts)
        react_time = parse_timestamp(react_ts)

        t = (stim_time - session_start).total_seconds()
        rt = (react_time - stim_time).total_seconds()
        points.append((t, rt))

    return subject_id, session_id, points


def load_subject_session_points(folder: Path) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """
    Returns:
        (subject_id, session_id) -> list of (t, rt)
    """
    out: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    for csv_file in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(csv_file.name):
            continue
        subject_id, session_id, points = load_session_file(csv_file)
        if points:
            out[(subject_id, session_id)] = points

    return out


def _square_legend(ax, items: List[Tuple[str, str]], fontsize: int = 14) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=12,
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
        )
        for color, label in items
    ]
    if handles:
        ax.legend(handles=handles, fontsize=fontsize, handlelength=1.0)


def compute_normalized_series_per_subject_session(
    points: List[Tuple[float, float]],
    baseline_frac: float = 0.2,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per subject-session normalized representations.

    Returns:
        t_norm: time normalized to [0, 1] using within-session min/max of stimulus times
        rt_raw: raw reaction time
        rt_z: z-scored RT (meanǹ: mean/std computed within the session)
        rt_pct: percent change from early baseline mean (baseline defined by earliest baseline_frac of the session)
        meta: array [rt_mean, rt_std, baseline_mean]
    """
    times = np.asarray([t for t, _ in points], dtype=float)
    rts = np.asarray([rt for _, rt in points], dtype=float)

    ok = np.isfinite(times) & np.isfinite(rts)
    times = times[ok]
    rts = rts[ok]

    if times.size == 0:
        return (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([np.nan, np.nan, np.nan]),
        )

    order = np.argsort(times)
    times = times[order]
    rts = rts[order]

    t0 = float(times.min())
    t1 = float(times.max())
    denom = (t1 - t0) if (t1 > t0) else 1.0
    t_norm = (times - t0) / denom

    rt_mean = float(rts.mean())
    rt_std = float(rts.std(ddof=1)) if rts.size >= 2 else 0.0

    rt_z = (rts - rt_mean) / (rt_std + eps)

    # Early baseline: earliest baseline_frac portion of time
    cutoff = float(baseline_frac)
    baseline_mask = t_norm <= cutoff
    if not baseline_mask.any():
        baseline_mean = rt_mean
    else:
        baseline_mean = float(rts[baseline_mask].mean())

    rt_pct = 100.0 * (rts - baseline_mean) / (abs(baseline_mean) + eps)

    meta = np.asarray([rt_mean, rt_std, baseline_mean], dtype=float)
    return t_norm, rts, rt_z, rt_pct, meta


def bin_and_aggregate(
    t_norm: np.ndarray,
    values: np.ndarray,
    n_bins: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin values by t_norm into n_bins equally spaced bins in [0, 1].
    Aggregate with median and IQR (25th, 75th).

    Returns:
        centers, median, q25, q75
    """
    if t_norm.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    med = np.full(n_bins, np.nan, dtype=float)
    q25 = np.full(n_bins, np.nan, dtype=float)
    q75 = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        mask = (t_norm >= left) & (t_norm < right) if i < n_bins - 1 else (t_norm >= left) & (t_norm <= right)

        if not mask.any():
            continue

        v = values[mask]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue

        med[i] = float(np.median(v))
        q25[i] = float(np.quantile(v, 0.25))
        q75[i] = float(np.quantile(v, 0.75))

    return centers, med, q25, q75


def save_within_subject_normalization_plots(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]],
    output_dir: Path,
    dpi: int = 300,
    baseline_frac: float = 0.2,
    n_bins: int = 40,
    raw_ylim: Tuple[float, float] = (0.1, 1.0),
    z_ylim: Tuple[float, float] | None = None,
    pct_ylim: Tuple[float, float] | None = None,
) -> None:
    """
    Produces three plots, all on subject-aligned time axis [0, 1]:

      1) Raw RT vs normalized time, aggregated across subjects per session
      2) Z-scored RT vs normalized time, aggregated across subjects per session
      3) Percent change from early baseline vs normalized time, aggregated across subjects per session

    Aggregation: median line + IQR band (25th to 75th) per bin.
    """
    session_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-session arrays of (t_norm, series) across all subject-sessions
    per_session_raw: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    per_session_z: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    per_session_pct: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for (subject_id, session_id), points in subj_sess_points.items():
        t_norm, rt_raw, rt_z, rt_pct, _ = compute_normalized_series_per_subject_session(
            points,
            baseline_frac=baseline_frac,
        )
        if t_norm.size == 0:
            continue

        per_session_raw.setdefault(session_id, []).append((t_norm, rt_raw))
        per_session_z.setdefault(session_id, []).append((t_norm, rt_z))
        per_session_pct.setdefault(session_id, []).append((t_norm, rt_pct))

    def plot_agg(
        series_by_session: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
        y_label: str,
        out_name: str,
        ylim: Tuple[float, float] | None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(11, 6))
        legend_items: List[Tuple[str, str]] = []

        for session_id in sorted(series_by_session.keys()):
            if session_id not in session_colors:
                continue
            pairs = series_by_session[session_id]
            if not pairs:
                continue

            # Concatenate all subject points then bin
            t_all = np.concatenate([t for t, _ in pairs])
            v_all = np.concatenate([v for _, v in pairs])

            ok = np.isfinite(t_all) & np.isfinite(v_all)
            t_all = t_all[ok]
            v_all = v_all[ok]
            if t_all.size == 0:
                continue

            centers, med, q25, q75 = bin_and_aggregate(t_all, v_all, n_bins=n_bins)
            valid = np.isfinite(med)
            if not valid.any():
                continue

            color = session_colors[session_id]

            ax.plot(centers[valid], med[valid], color=color, linewidth=2.6)
            ax.fill_between(centers[valid], q25[valid], q75[valid], color=color, alpha=0.18, linewidth=0.0)

            legend_items.append((color, f"Session {session_id}"))

        ax.set_xlabel("Normalized session time (0 to 1)")
        ax.set_ylabel(y_label)
        ax.grid(True)
        if ylim is not None:
            ax.set_ylim(*ylim)

        _square_legend(ax, legend_items, fontsize=14)

        fig.tight_layout()
        fig.savefig(output_dir / out_name, dpi=dpi)
        plt.close(fig)

    plot_agg(per_session_raw, "Reaction time (seconds)", "rt_raw_aligned_time.png", raw_ylim)
    plot_agg(per_session_z, "Z-scored reaction time (within subject-session)", "rt_zscore_aligned_time.png", z_ylim)
    plot_agg(
        per_session_pct,
        f"Percent change from early baseline (first {baseline_frac:.0%})",
        "rt_percent_change_aligned_time.png",
        pct_ylim,
    )


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    output_dir = Path("/home/mojtaba.kananisarche/Zh/plots/rt_within_subject_normalization")

    subj_sess_points = load_subject_session_points(data_folder)

    save_within_subject_normalization_plots(
        subj_sess_points=subj_sess_points,
        output_dir=output_dir,
        dpi=300,
        baseline_frac=0.2,
        n_bins=40,
        raw_ylim=(0.1, 1.0),
        z_ylim=None,
        pct_ylim=None,
    )


if __name__ == "__main__":
    main()