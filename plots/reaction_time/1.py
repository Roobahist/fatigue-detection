from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")


# ==============================
# IO and parsing
# ==============================

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Returns:
        session_id
        list of (time_since_start_sec, reaction_time_sec)
    """
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename format: {path.name}")

    _, session_id_str = path.stem.split("-")
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

    return session_id, points


def load_all_sessions(folder: Path) -> Dict[int, List[Tuple[float, float]]]:
    data: Dict[int, List[Tuple[float, float]]] = {}

    for csv_file in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(csv_file.name):
            continue

        session_id, points = load_session_file(csv_file)
        data.setdefault(session_id, []).extend(points)

    return data


# ==============================
# Analysis helpers
# ==============================

def rolling_quantiles(
    points: List[Tuple[float, float]],
    window_sec: float,
    step_sec: float,
    quantiles: List[float],
):
    times = np.asarray([t for t, _ in points], dtype=float)
    rts = np.asarray([rt for _, rt in points], dtype=float)

    order = np.argsort(times)
    times = times[order]
    rts = rts[order]

    centers = []
    qvals = {q: [] for q in quantiles}

    left = times.min()
    right_max = times.max()

    while left + window_sec <= right_max:
        right = left + window_sec
        mask = (times >= left) & (times < right)

        if mask.any():
            centers.append(left + window_sec / 2.0)
            for q in quantiles:
                qvals[q].append(np.quantile(rts[mask], q))

        left += step_sec

    return np.asarray(centers), {q: np.asarray(v) for q, v in qvals.items()}


def rolling_tail_mass(
    points: List[Tuple[float, float]],
    threshold: float,
    window_sec: float,
    step_sec: float,
):
    times = np.asarray([t for t, _ in points], dtype=float)
    rts = np.asarray([rt for _, rt in points], dtype=float)

    order = np.argsort(times)
    times = times[order]
    rts = rts[order]

    centers = []
    fractions = []

    left = times.min()
    right_max = times.max()

    while left + window_sec <= right_max:
        right = left + window_sec
        mask = (times >= left) & (times < right)

        if mask.any():
            centers.append(left + window_sec / 2.0)
            fractions.append(float(np.mean(rts[mask] > threshold)))

        left += step_sec

    return np.asarray(centers), np.asarray(fractions)


def survival_curve(rts: np.ndarray):
    rts = np.sort(rts)
    surv = 1.0 - np.arange(1, rts.size + 1) / rts.size
    return rts, surv


# ==============================
# Plotting
# ==============================

def save_tail_analysis_plots(
    data: Dict[int, List[Tuple[float, float]]],
    out_dir: Path,
    quantiles: List[float] = [0.5, 0.75, 0.9],
    rt_threshold: float = 0.5,
    window_sec: float = 60.0,
    step_sec: float = 15.0,
    rt_ylim: Tuple[float, float] = (0.1, 1.0),
    dpi: int = 300,
):
    session_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Plot 1: Quantile trajectories --------
    fig, ax = plt.subplots(figsize=(11, 6))

    for sid, points in data.items():
        if sid not in session_colors or not points:
            continue

        centers, qvals = rolling_quantiles(
            points, window_sec, step_sec, quantiles
        )

        color = session_colors[sid]
        for q in quantiles:
            ax.plot(
                centers,
                qvals[q],
                color=color,
                linewidth=2.5 if q == 0.5 else 1.8,
                alpha=0.95 if q == 0.5 else 0.7,
                label=f"Session {sid} q={q}",
            )

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Reaction time quantile (seconds)")
    ax.set_ylim(*rt_ylim)
    ax.grid(True)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "rt_quantiles_over_time.png", dpi=dpi)
    plt.close(fig)

    # -------- Plot 2: Survival curves --------
    fig, ax = plt.subplots(figsize=(7, 6))

    for sid, points in data.items():
        if sid not in session_colors or not points:
            continue

        rts = np.asarray([rt for _, rt in points], dtype=float)
        x, y = survival_curve(rts)

        ax.plot(
            x,
            y,
            color=session_colors[sid],
            linewidth=2.5,
            label=f"Session {sid}",
        )

    ax.set_xlabel("Reaction time threshold (seconds)")
    ax.set_ylabel("P(RT > t)")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "rt_survival_curves.png", dpi=dpi)
    plt.close(fig)

    # -------- Plot 3: Tail mass over time --------
    fig, ax = plt.subplots(figsize=(11, 6))

    for sid, points in data.items():
        if sid not in session_colors or not points:
            continue

        centers, frac = rolling_tail_mass(
            points, rt_threshold, window_sec, step_sec
        )

        ax.plot(
            centers,
            frac,
            color=session_colors[sid],
            linewidth=2.5,
            label=f"Session {sid}",
        )

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel(f"Fraction of RT > {rt_threshold:.2f}s")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "rt_tail_mass_over_time.png", dpi=dpi)
    plt.close(fig)


# ==============================
# Main
# ==============================

def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    out_dir = Path("/home/mojtaba.kananisarche/Zh/plots/rt_tail_analysis")

    session_data = load_all_sessions(data_folder)

    save_tail_analysis_plots(
        session_data,
        out_dir,
        quantiles=[0.5, 0.75, 0.9],
        rt_threshold=0.5,
        window_sec=60.0,
        step_sec=15.0,
        rt_ylim=(0.1, 1.0),
    )


if __name__ == "__main__":
    main()