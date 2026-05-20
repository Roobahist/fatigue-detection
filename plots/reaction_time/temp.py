from __future__ import annotations

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
    """
    Smooth density on y_grid using Gaussian KDE.

    Tries SciPy first. If SciPy is unavailable, falls back to a smoothed histogram.
    """
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


def _t_critical_approx(df: int, level: float) -> float:
    """
    Two-sided critical value for t distribution at confidence level.

    Uses SciPy if available, otherwise normal approximation (ok for large df).
    """
    import math

    alpha = 1.0 - float(level)
    try:
        from scipy.stats import t  # type: ignore

        return float(t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        # Normal approx
        # For 95% this is 1.96, for other levels we approximate using inverse error function
        # If math.erfcinv is unavailable (it is), keep a simple fallback.
        if abs(level - 0.95) < 1e-12:
            return 1.96
        if abs(level - 0.90) < 1e-12:
            return 1.645
        if abs(level - 0.99) < 1e-12:
            return 2.576
        return 1.96


def _linear_band(
    xs,
    ys,
    x_grid,
    slope: float,
    intercept: float,
    kind: str,
    level: float,
):
    """
    kind:
      - "mean": confidence band for mean regression line
      - "pred": prediction interval for a new observation
    """
    import numpy as np

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    n = xs.size
    if n < 3:
        return None

    y_hat_grid = slope * x_grid + intercept

    resid = ys - (slope * xs + intercept)
    s2 = float(np.sum(resid**2) / (n - 2))
    s = s2**0.5

    x_bar = float(xs.mean())
    sxx = float(np.sum((xs - x_bar) ** 2))
    if sxx <= 0:
        return None

    se_mean = s * np.sqrt((1.0 / n) + ((x_grid - x_bar) ** 2) / sxx)

    if kind == "pred":
        se = s * np.sqrt(1.0 + (1.0 / n) + ((x_grid - x_bar) ** 2) / sxx)
    else:
        se = se_mean

    tcrit = _t_critical_approx(df=n - 2, level=level)
    lo = y_hat_grid - tcrit * se
    hi = y_hat_grid + tcrit * se
    return lo, hi


def save_reaction_time_plot(
    data: Dict[int, List[Tuple[float, float]]],
    output_path: Path,
    dpi: int = 300,
    rt_ylim: Tuple[float, float] = (0.0, 1.0),
    kde_points: int = 400,
    bandwidth_scale: float = 1.0,
    band_kind: str = "mean",   # "mean" or "pred"
    band_level: float = 0.95,
    band_alpha: float = 0.18,
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
    legend_items: List[Tuple[str, str]] = []

    for session_id in sorted(data.keys()):
        if session_id not in session_colors:
            continue
        if not data[session_id]:
            continue

        xs, ys = zip(*data[session_id])
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        color = session_colors[session_id]

        # Light scatter (no legend entry)
        ax.scatter(xs, ys, s=20, alpha=0.15, color=color)

        # Regression line + band
        if xs.size >= 2:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            x_line = np.linspace(xs.min(), xs.max(), 200)
            y_line = slope * x_line + intercept

            ax.plot(x_line, y_line, color=color, linewidth=2.5)

            band = _linear_band(
                xs=xs,
                ys=ys,
                x_grid=x_line,
                slope=float(slope),
                intercept=float(intercept),
                kind=band_kind,
                level=band_level,
            )
            if band is not None:
                lo, hi = band
                ax.fill_between(x_line, lo, hi, color=color, alpha=band_alpha, linewidth=0.0)

            legend_items.append((color, f"Session {session_id}: y = {slope:.3e} x + {intercept:.3e}"))

        # KDE distribution on the right
        dens = _compute_kde_density(list(ys), y_grid, bandwidth_scale)
        max_density = max(max_density, float(np.max(dens)) if dens.size else 0.0)

        ax_dist.fill_betweenx(y_grid, 0.0, dens, alpha=0.25, color=color)
        ax_dist.plot(dens, y_grid, color=color, linewidth=1.5)

    ax.set_ylim(y0, y1)
    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Reaction time (seconds)")
    ax.grid(True)

    ax_dist.set_xlim(0.0, max_density * 1.05 if max_density > 0 else 1.0)
    ax_dist.set_xlabel("Density")
    ax_dist.grid(True)
    ax_dist.tick_params(axis="y", left=False, labelleft=False)
    ax_dist.spines["left"].set_visible(False)

    # Square legend handles in dark session colors
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
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=14, handlelength=1.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def compute_subject_session_stats(folder: Path) -> Dict[int, List[Tuple[float, float]]]:
    """
    Returns:
        session_id -> list of (mean_rt, std_rt) across subjects
        Each CSV contributes exactly one point to its session.
    """
    import numpy as np

    by_session: Dict[int, List[Tuple[float, float]]] = {}

    for csv_file in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(csv_file.name):
            continue

        _, session_id, points = load_session_file(csv_file)
        if not points:
            continue

        rts = np.asarray([rt for _, rt in points], dtype=float)
        rts = rts[np.isfinite(rts)]
        if rts.size == 0:
            continue

        mean_rt = float(rts.mean())
        std_rt = float(rts.std(ddof=1)) if rts.size >= 2 else 0.0

        by_session.setdefault(session_id, []).append((mean_rt, std_rt))

    return by_session


def save_mean_std_joint_plot(
    stats_by_session: Dict[int, List[Tuple[float, float]]],
    output_path: Path,
    dpi: int = 300,
    bins: int = 30,
    scatter_alpha: float = 0.35,
    point_size: int = 28,
) -> None:
    """
    Joint style plot:
      Center: mean RT vs std RT (per subject-session point)
      Top: distribution of mean RT
      Right: distribution of std RT
    """
    import numpy as np

    session_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    divider = make_axes_locatable(ax)

    ax_top = divider.append_axes("top", size="22%", pad=0.12, sharex=ax)
    ax_right = divider.append_axes("right", size="22%", pad=0.12, sharey=ax)

    legend_items: List[Tuple[str, str]] = []

    for session_id in sorted(stats_by_session.keys()):
        if session_id not in session_colors:
            continue

        pts = stats_by_session[session_id]
        if not pts:
            continue

        color = session_colors[session_id]
        means = np.asarray([m for m, _ in pts], dtype=float)
        stds = np.asarray([s for _, s in pts], dtype=float)

        means = means[np.isfinite(means)]
        stds = stds[np.isfinite(stds)]
        if means.size == 0 or stds.size == 0:
            continue

        ax.scatter(means, stds, s=point_size, alpha=scatter_alpha, color=color)

        ax_top.hist(means, bins=bins, density=True, alpha=0.45, color=color)
        ax_right.hist(stds, bins=bins, density=True, orientation="horizontal", alpha=0.45, color=color)

        legend_items.append((color, f"Session {session_id}"))

    ax.set_xlabel("Mean reaction time (seconds)")
    ax.set_ylabel("Std reaction time (seconds)")
    ax.grid(True)

    ax_top.set_ylabel("Density")
    ax_top.grid(True)

    ax_right.set_xlabel("Density")
    ax_right.grid(True)

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

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
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=14, handlelength=1.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")

    out_scatter = Path("/home/mojtaba.kananisarche/Zh/plots/reaction_times_scatter.png")
    out_joint = Path("/home/mojtaba.kananisarche/Zh/plots/rt_mean_vs_std_joint.png")

    # Plot 1: RT over time scatter + KDE right + regression + band
    session_data = load_all_sessions(data_folder)
    save_reaction_time_plot(
        session_data,
        out_scatter,
        rt_ylim=(0.1, 1.0),
        bandwidth_scale=1.0,
        band_kind="mean",
        band_level=0.95,
        band_alpha=0.18,
    )

    # Plot 2: Mean vs Std per subject-session, with marginal hists
    stats = compute_subject_session_stats(data_folder)
    save_mean_std_joint_plot(
        stats,
        out_joint,
        bins=30,
        scatter_alpha=0.35,
        point_size=28,
    )


if __name__ == "__main__":
    main()