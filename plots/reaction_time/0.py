from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[str, int, List[Tuple[float, float]]]:
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename format (expected number-number.csv): {path.name}")

    subject_id, session_id_str = path.stem.split("-")
    session_id = int(session_id_str)

    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    session_start = parse_timestamp(lines[0])

    points: List[Tuple[float, float]] = []
    for line in lines[1:]:
        stim_ts, react_ts = line.split(";")

        stim_time = parse_timestamp(stim_ts)
        react_time = parse_timestamp(react_ts)

        time_since_start = (stim_time - session_start).total_seconds()
        reaction_time = (react_time - stim_time).total_seconds()

        points.append((time_since_start, reaction_time))

    return subject_id, session_id, points


def load_all_sessions(folder: Path) -> Dict[int, List[Tuple[float, float]]]:
    data: Dict[int, List[Tuple[float, float]]] = {}

    for csv_file in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(csv_file.name):
            continue

        _, session_id, points = load_session_file(csv_file)
        data.setdefault(session_id, []).extend(points)

    return data


def _compute_kde_density(
    ys: List[float],
    y_grid,
    bandwidth_scale: float = 1.0,
):
    import numpy as np

    ys_arr = np.asarray(ys, dtype=float)
    ys_arr = ys_arr[np.isfinite(ys_arr)]

    if ys_arr.size < 2:
        return np.zeros_like(y_grid, dtype=float)

    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(ys_arr)
        kde.set_bandwidth(kde.factor * float(bandwidth_scale))
        return kde(y_grid)
    except Exception:
        bins = max(30, int(len(ys_arr) ** 0.5))
        hist, edges = np.histogram(ys_arr, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        dx = float(np.median(np.diff(centers))) if centers.size > 1 else 1.0
        sigma = max(dx, (ys_arr.std(ddof=1) + 1e-12) * 0.15 * float(bandwidth_scale))
        radius = int(max(3, round(4.0 * sigma / dx)))
        xs = np.arange(-radius, radius + 1) * dx
        kernel = np.exp(-0.5 * (xs / sigma) ** 2)
        kernel /= kernel.sum()

        smooth_hist = np.convolve(hist, kernel, mode="same")
        return np.interp(y_grid, centers, smooth_hist, left=0.0, right=0.0)


def save_reaction_time_plot(
    data: Dict[int, List[Tuple[float, float]]],
    output_path: Path,
    dpi: int = 300,
    rt_ylim: Tuple[float, float] = (0.0, 1.0),
    kde_points: int = 400,
    bandwidth_scale: float = 1.0,
) -> None:
    import numpy as np

    session_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    fig, ax = plt.subplots(figsize=(11.5, 6))

    divider = make_axes_locatable(ax)
    ax_dist = divider.append_axes("right", size="25%", pad=0.08, sharey=ax)

    y0, y1 = rt_ylim
    y_grid = np.linspace(y0, y1, int(kde_points))

    max_density = 0.0
    legend_items = []

    for session_id in sorted(data.keys()):
        if session_id not in session_colors:
            continue
        if not data[session_id]:
            continue

        xs, ys = zip(*data[session_id])
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        color = session_colors[session_id]

        ax.scatter(
            xs,
            ys,
            s=10,
            alpha=0.15,
            color=color,
        )

        if xs.size >= 2:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            x_line = np.linspace(xs.min(), xs.max(), 200)
            y_line = slope * x_line + intercept

            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=2.5,
            )

            legend_items.append(
                (
                    color,
                    f"Session {session_id}: y = {slope:.5f} x + {intercept:.2f}",
                )
            )

        dens = _compute_kde_density(list(ys), y_grid, bandwidth_scale)
        max_density = max(max_density, float(np.max(dens)) if dens.size else 0.0)

        ax_dist.fill_betweenx(
            y_grid,
            0.0,
            dens,
            alpha=0.25,
            color=color,
        )
        ax_dist.plot(
            dens,
            y_grid,
            color=color,
            linewidth=1.5,
        )

    ax.set_ylim(y0, y1)
    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Reaction time (seconds)")
    ax.grid(True)

    ax_dist.set_xlim(0.0, max_density * 1.05 if max_density > 0 else 1.0)
    ax_dist.set_xlabel("Density")
    ax_dist.grid(True)
    ax_dist.tick_params(axis="y", left=False, labelleft=False)
    ax_dist.spines["left"].set_visible(False)

    legend_handles = [
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
        for color, label in legend_items
    ]

    ax.legend(handles=legend_handles, fontsize=14)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    output_file = Path("/home/mojtaba.kananisarche/Zh/plots/reaction_times_scatter.png")

    session_data = load_all_sessions(data_folder)
    save_reaction_time_plot(
        session_data,
        output_file,
        rt_ylim=(0.1, 1.0),
        bandwidth_scale=1.0,
    )