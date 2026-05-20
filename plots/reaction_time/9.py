from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")

LAPSE_THRESHOLD_SEC = 0.5


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, List[Tuple[float, float]]]:
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename: {path.name}")

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

    points.sort(key=lambda x: x[0])
    return session_id, points


def load_all_sessions(folder: Path) -> Dict[int, List[Tuple[float, float]]]:
    data: Dict[int, List[Tuple[float, float]]] = {}
    for f in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(f.name):
            continue
        sess, pts = load_session_file(f)
        data.setdefault(sess, []).extend(pts)
    return data


def gaussian_kernel(x: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-0.5 * (x / bandwidth) ** 2)


def smooth_lapse_rate_over_time(
    points: List[Tuple[float, float]],
    time_grid: np.ndarray,
    bandwidth_sec: float,
) -> np.ndarray:
    """
    Kernel-smoothed lapse probability P(lapse | time)
    """
    times = np.asarray([t for t, _ in points], dtype=float)
    lapses = np.asarray([rt > LAPSE_THRESHOLD_SEC for _, rt in points], dtype=float)

    smoothed = np.zeros_like(time_grid)

    for i, tg in enumerate(time_grid):
        weights = gaussian_kernel(times - tg, bandwidth_sec)
        wsum = weights.sum()
        if wsum > 0:
            smoothed[i] = np.sum(weights * lapses) / wsum
        else:
            smoothed[i] = np.nan

    return smoothed


def save_smooth_lapse_distribution_plot(
    data: Dict[int, List[Tuple[float, float]]],
    output_path: Path,
    bandwidth_sec: float = 60.0,
    n_grid: int = 400,
    dpi: int = 300,
) -> None:
    session_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    for sess in sorted(data.keys()):
        if sess not in session_colors:
            continue
        points = data[sess]
        if not points:
            continue

        times = np.asarray([t for t, _ in points], dtype=float)
        t_min, t_max = times.min(), times.max()
        time_grid = np.linspace(t_min, t_max, n_grid)

        lapse_curve = smooth_lapse_rate_over_time(
            points,
            time_grid=time_grid,
            bandwidth_sec=bandwidth_sec,
        )

        ax.plot(
            time_grid,
            lapse_curve,
            color=session_colors[sess],
            linewidth=3.0,
            label=f"Session {sess}",
        )

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("P(lapse)  (RT > 500 ms)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(fontsize=13)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    output_file = Path("/home/mojtaba.kananisarche/Zh/plots/lapse_distribution_smooth.png")

    session_data = load_all_sessions(data_folder)

    save_smooth_lapse_distribution_plot(
        session_data,
        output_file,
        bandwidth_sec=60.0,
        n_grid=400,
    )


if __name__ == "__main__":
    main()