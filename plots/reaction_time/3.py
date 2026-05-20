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
        out[(subject_id, session_id)] = points

    return out


def rolling_mean_std_cv(
    points: List[Tuple[float, float]],
    window_sec: float,
    step_sec: float,
    ddof: int = 1,
    cv_eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling window stats over time.

    Returns:
        centers, means, stds, cvs
    """
    times = np.asarray([t for t, _ in points], dtype=float)
    rts = np.asarray([rt for _, rt in points], dtype=float)

    ok = np.isfinite(times) & np.isfinite(rts)
    times = times[ok]
    rts = rts[ok]

    order = np.argsort(times)
    times = times[order]
    rts = rts[order]

    if times.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

    centers: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    cvs: List[float] = []

    left = float(times.min())
    right_max = float(times.max())

    while left + window_sec <= right_max:
        right = left + window_sec
        mask = (times >= left) & (times < right)

        if mask.any():
            w = rts[mask]
            m = float(w.mean())
            s = float(w.std(ddof=ddof)) if w.size >= 2 else 0.0
            c = float(s / (abs(m) + cv_eps))

            centers.append(left + window_sec / 2.0)
            means.append(m)
            stds.append(s)
            cvs.append(c)

        left += step_sec

    return np.asarray(centers), np.asarray(means), np.asarray(stds), np.asarray(cvs)


def aggregate_session_rolling_stats(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]],
    window_sec: float,
    step_sec: float,
) -> Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    For each session, compute rolling stats per subject.
    Returns a structure that preserves per-subject trajectories:

        session_id -> {
            "mean": [(centers, values), ...],
            "std":  [(centers, values), ...],
            "cv":   [(centers, values), ...],
        }
    """
    out: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {}

    for (subject_id, session_id), points in subj_sess_points.items():
        centers, means, stds, cvs = rolling_mean_std_cv(points, window_sec, step_sec)
        if centers.size == 0:
            continue

        sess = out.setdefault(session_id, {"mean": [], "std": [], "cv": []})
        sess["mean"].append((centers, means))
        sess["std"].append((centers, stds))
        sess["cv"].append((centers, cvs))

    return out


def compute_subject_level_mean_std(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]]
) -> Dict[int, List[Tuple[float, float]]]:
    """
    For each session, compute (mean_rt, std_rt) per subject-session.
    Returns:
        session_id -> [(mean, std), ...]
    """
    out: Dict[int, List[Tuple[float, float]]] = {}

    for (subject_id, session_id), points in subj_sess_points.items():
        rts = np.asarray([rt for _, rt in points], dtype=float)
        rts = rts[np.isfinite(rts)]
        if rts.size == 0:
            continue

        mean_rt = float(rts.mean())
        std_rt = float(rts.std(ddof=1)) if rts.size >= 2 else 0.0
        out.setdefault(session_id, []).append((mean_rt, std_rt))

    return out


def _session_colors():
    return {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }


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


def save_variability_dynamics_plots(
    session_roll: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]],
    session_mean_std: Dict[int, List[Tuple[float, float]]],
    output_dir: Path,
    dpi: int = 300,
    time_grid_sec: float = 10.0,
) -> None:
    """
    Produces:
      1) Rolling window std over time (aggregated across subjects per session)
      2) Rolling CV over time (aggregated across subjects per session)
      3) Variability vs mean scatter per subject-session
    """
    colors = _session_colors()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper: interpolate each subject trajectory to a common grid then aggregate
    def aggregate_to_grid(trajs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not trajs:
            return np.asarray([]), np.asarray([]), np.asarray([])

        t_min = min(float(t[0].min()) for t, _ in trajs)
        t_max = max(float(t[0].max()) for t, _ in trajs)
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            return np.asarray([]), np.asarray([]), np.asarray([])

        grid = np.arange(t_min, t_max + 1e-9, float(time_grid_sec))

        mat = []
        for t, v in trajs:
            if t.size < 2:
                continue
            vv = np.interp(grid, t, v, left=np.nan, right=np.nan)
            mat.append(vv)

        if not mat:
            return np.asarray([]), np.asarray([]), np.asarray([])

        M = np.vstack(mat)
        med = np.nanmedian(M, axis=0)
        q25 = np.nanquantile(M, 0.25, axis=0)
        q75 = np.nanquantile(M, 0.75, axis=0)
        return grid, med, np.vstack([q25, q75])

    # ----------------------------
    # Plot 1: Rolling std over time
    # ----------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    legend_items: List[Tuple[str, str]] = []

    for session_id in sorted(session_roll.keys()):
        if session_id not in colors:
            continue
        trajs = session_roll[session_id].get("std", [])
        grid, med, band = aggregate_to_grid(trajs)
        if grid.size == 0:
            continue

        color = colors[session_id]
        ax.plot(grid, med, color=color, linewidth=2.5)
        ax.fill_between(grid, band[0], band[1], color=color, alpha=0.18, linewidth=0.0)
        legend_items.append((color, f"Session {session_id}"))

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Rolling std of RT (seconds)")
    ax.grid(True)
    _square_legend(ax, legend_items, fontsize=14)

    fig.tight_layout()
    fig.savefig(output_dir / "rt_rolling_std_over_time.png", dpi=dpi)
    plt.close(fig)

    # ----------------------------
    # Plot 2: Rolling CV over time
    # ----------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    legend_items = []

    for session_id in sorted(session_roll.keys()):
        if session_id not in colors:
            continue
        trajs = session_roll[session_id].get("cv", [])
        grid, med, band = aggregate_to_grid(trajs)
        if grid.size == 0:
            continue

        color = colors[session_id]
        ax.plot(grid, med, color=color, linewidth=2.5)
        ax.fill_between(grid, band[0], band[1], color=color, alpha=0.18, linewidth=0.0)
        legend_items.append((color, f"Session {session_id}"))

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Rolling CV (std / mean)")
    ax.grid(True)
    _square_legend(ax, legend_items, fontsize=14)

    fig.tight_layout()
    fig.savefig(output_dir / "rt_rolling_cv_over_time.png", dpi=dpi)
    plt.close(fig)

    # ----------------------------------------
    # Plot 3: Variability vs mean per subject
    # ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    legend_items = []

    for session_id in sorted(session_mean_std.keys()):
        if session_id not in colors:
            continue
        pts = session_mean_std[session_id]
        if not pts:
            continue

        means = np.asarray([m for m, _ in pts], dtype=float)
        stds = np.asarray([s for _, s in pts], dtype=float)

        ok = np.isfinite(means) & np.isfinite(stds)
        means = means[ok]
        stds = stds[ok]
        if means.size == 0:
            continue

        color = colors[session_id]
        ax.scatter(means, stds, s=28, alpha=0.35, color=color)
        legend_items.append((color, f"Session {session_id} (n={means.size})"))

    ax.set_xlabel("Mean RT per subject-session (seconds)")
    ax.set_ylabel("Std RT per subject-session (seconds)")
    ax.grid(True)
    _square_legend(ax, legend_items, fontsize=14)

    fig.tight_layout()
    fig.savefig(output_dir / "rt_variability_vs_mean_scatter.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    output_dir = Path("/home/mojtaba.kananisarche/Zh/plots/rt_variability_dynamics")

    window_sec = 60.0
    step_sec = 15.0

    subj_sess_points = load_subject_session_points(data_folder)

    session_roll = aggregate_session_rolling_stats(
        subj_sess_points,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    session_mean_std = compute_subject_level_mean_std(subj_sess_points)

    save_variability_dynamics_plots(
        session_roll=session_roll,
        session_mean_std=session_mean_std,
        output_dir=output_dir,
        dpi=300,
        time_grid_sec=10.0,
    )


if __name__ == "__main__":
    main()