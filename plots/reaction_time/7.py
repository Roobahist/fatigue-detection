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
        points.append((float(t), float(rt)))

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


def compute_subject_session_mean_std(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]]
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Returns:
        (subject_id, session_id) -> (mean_rt, std_rt)
    """
    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for key, points in subj_sess_points.items():
        rts = np.asarray([rt for _, rt in points], dtype=float)
        rts = rts[np.isfinite(rts)]
        if rts.size == 0:
            continue
        mean_rt = float(rts.mean())
        std_rt = float(rts.std(ddof=1)) if rts.size >= 2 else 0.0
        out[key] = (mean_rt, std_rt)
    return out


def pooled_rts_by_session(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]]
) -> Dict[int, np.ndarray]:
    """
    Returns:
        session_id -> pooled RT array
    """
    by_sess: Dict[int, List[float]] = {}
    for (_subj, sess), points in subj_sess_points.items():
        by_sess.setdefault(sess, []).extend([rt for _, rt in points])
    out: Dict[int, np.ndarray] = {}
    for sess, lst in by_sess.items():
        arr = np.asarray(lst, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[sess] = arr
    return out


def _compute_kde_density(
    ys: np.ndarray,
    y_grid: np.ndarray,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """
    Smooth density on y_grid using Gaussian KDE.

    Tries SciPy first. If SciPy is unavailable, falls back to a smoothed histogram.
    """
    ys = np.asarray(ys, dtype=float)
    ys = ys[np.isfinite(ys)]
    if ys.size < 2:
        return np.zeros_like(y_grid, dtype=float)

    try:
        from scipy.stats import gaussian_kde  # type: ignore

        kde = gaussian_kde(ys)
        kde.set_bandwidth(kde.factor * float(bandwidth_scale))
        return kde(y_grid)
    except Exception:
        bins = max(30, int(ys.size**0.5))
        hist, edges = np.histogram(ys, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        if centers.size < 2:
            return np.zeros_like(y_grid, dtype=float)

        dx = float(np.median(np.diff(centers)))
        sigma = max(dx, (float(np.std(ys, ddof=1)) + 1e-12) * 0.15 * float(bandwidth_scale))
        radius = int(max(3, round(4.0 * sigma / dx)))
        xs = np.arange(-radius, radius + 1) * dx
        kernel = np.exp(-0.5 * (xs / sigma) ** 2)
        kernel /= kernel.sum()

        smooth_hist = np.convolve(hist, kernel, mode="same")
        return np.interp(y_grid, centers, smooth_hist, left=0.0, right=0.0)


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 5000,
    level: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Returns:
        mean, lo, hi
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(v.mean())
    if v.size == 1:
        return mean, mean, mean

    boots = np.empty(n_boot, dtype=float)
    n = v.size
    for i in range(n_boot):
        sample = v[rng.integers(0, n, size=n)]
        boots[i] = float(sample.mean())

    alpha = 1.0 - float(level)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean, lo, hi


def save_between_session_transition_plots(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]],
    output_dir: Path,
    dpi: int = 300,
    rt_xlim: Tuple[float, float] = (0.1, 1.0),
    rt_ylim: Tuple[float, float] = (0.0, 0.35),
    kde_grid: Tuple[float, float] = (0.1, 1.0),
    kde_points: int = 500,
    bandwidth_scale: float = 1.0,
    n_boot: int = 5000,
) -> None:
    """
    Saves:
      1) arrows_session1_to_3_mean_std.png
      2) distribution_shape_across_sessions.png
      3) paired_differences_with_ci.png
    """
    session_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute per subject-session mean/std
    ms = compute_subject_session_mean_std(subj_sess_points)

    # --------------------------
    # Plot 1: Per subject arrows in mean-std space (S1 -> S3)
    # --------------------------
    fig, ax = plt.subplots(figsize=(9.5, 7))

    # subjects present in both sessions
    subj_s1 = {subj for (subj, sess) in ms.keys() if sess == 1}
    subj_s3 = {subj for (subj, sess) in ms.keys() if sess == 3}
    common = sorted(subj_s1 & subj_s3)

    if common:
        s1_means = np.asarray([ms[(subj, 1)][0] for subj in common], dtype=float)
        s1_stds = np.asarray([ms[(subj, 1)][1] for subj in common], dtype=float)
        s3_means = np.asarray([ms[(subj, 3)][0] for subj in common], dtype=float)
        s3_stds = np.asarray([ms[(subj, 3)][1] for subj in common], dtype=float)

        # Draw arrows (thin, gray) plus start/end markers in session colors
        for x0, y0, x1, y1 in zip(s1_means, s1_stds, s3_means, s3_stds):
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.25),
            )

        ax.scatter(s1_means, s1_stds, s=35, alpha=0.75, color=session_colors[1], label="Session 1")
        ax.scatter(s3_means, s3_stds, s=35, alpha=0.75, color=session_colors[3], label="Session 3")

        ax.set_title(f"Per subject transition in mean-std space (n={len(common)})")
    else:
        ax.text(
            0.5,
            0.5,
            "No subjects found with both Session 1 and Session 3",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Mean reaction time (seconds)")
    ax.set_ylabel("Std reaction time (seconds)")
    ax.set_xlim(*rt_xlim)
    ax.set_ylim(*rt_ylim)
    ax.grid(True)
    if common:
        ax.legend(fontsize=13)

    fig.tight_layout()
    fig.savefig(output_dir / "arrows_session1_to_3_mean_std.png", dpi=dpi)
    plt.close(fig)

    # --------------------------
    # Plot 2: Change in distribution shape across sessions (pooled KDE)
    # --------------------------
    fig, ax = plt.subplots(figsize=(9.5, 7))

    pooled = pooled_rts_by_session(subj_sess_points)
    y0, y1 = float(kde_grid[0]), float(kde_grid[1])
    grid = np.linspace(y0, y1, int(kde_points))

    max_d = 0.0
    for sess in [1, 2, 3]:
        if sess not in pooled or pooled[sess].size == 0:
            continue
        dens = _compute_kde_density(pooled[sess], grid, bandwidth_scale=bandwidth_scale)
        max_d = max(max_d, float(np.max(dens)) if dens.size else 0.0)

        ax.plot(grid, dens, color=session_colors[sess], lw=2.6, label=f"Session {sess}")
        ax.fill_between(grid, 0.0, dens, color=session_colors[sess], alpha=0.18, linewidth=0.0)

    ax.set_xlabel("Reaction time (seconds)")
    ax.set_ylabel("Density")
    ax.set_xlim(y0, y1)
    if max_d > 0:
        ax.set_ylim(0.0, max_d * 1.05)
    ax.grid(True)
    ax.legend(fontsize=13)
    ax.set_title("Pooled distribution shape across sessions")

    fig.tight_layout()
    fig.savefig(output_dir / "distribution_shape_across_sessions.png", dpi=dpi)
    plt.close(fig)

    # --------------------------
    # Plot 3: Paired differences with confidence intervals (S3 - S1)
    # --------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    if common:
        delta_mean = s3_means - s1_means
        delta_std = s3_stds - s1_stds

        # x positions for two metrics
        x_mean = 0.0
        x_std = 1.0

        # jittered subject points
        rng = np.random.default_rng(0)
        j1 = rng.normal(0.0, 0.06, size=delta_mean.size)
        j2 = rng.normal(0.0, 0.06, size=delta_std.size)

        ax.scatter(np.full(delta_mean.size, x_mean) + j1, delta_mean, s=28, alpha=0.35, color=session_colors[3])
        ax.scatter(np.full(delta_std.size, x_std) + j2, delta_std, s=28, alpha=0.35, color=session_colors[3])

        m_mean, lo_mean, hi_mean = bootstrap_ci_mean(delta_mean, n_boot=n_boot, level=0.95, seed=1)
        m_std, lo_std, hi_std = bootstrap_ci_mean(delta_std, n_boot=n_boot, level=0.95, seed=2)

        # CI bars
        ax.plot([x_mean, x_mean], [lo_mean, hi_mean], lw=3.2, alpha=0.9)
        ax.plot([x_std, x_std], [lo_std, hi_std], lw=3.2, alpha=0.9)

        # point for mean
        ax.scatter([x_mean], [m_mean], s=80, alpha=0.9)
        ax.scatter([x_std], [m_std], s=80, alpha=0.9)

        ax.axhline(0.0, lw=2.0, alpha=0.8)
        ax.set_xticks([x_mean, x_std])
        ax.set_xticklabels(
            [
                f"Delta mean\n(mean={m_mean:.3e})",
                f"Delta std\n(mean={m_std:.3e})",
            ]
        )
        ax.set_ylabel("Session 3 minus Session 1 (seconds)")
        ax.set_title(f"Paired differences with 95% bootstrap CI (n={len(common)})")
        ax.grid(True, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "No paired subjects available for Session 1 vs Session 3",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "paired_differences_with_ci.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    output_dir = Path("/home/mojtaba.kananisarche/Zh/plots/rt_between_session_transitions")

    subj_sess_points = load_subject_session_points(data_folder)

    save_between_session_transition_plots(
        subj_sess_points=subj_sess_points,
        output_dir=output_dir,
        dpi=300,
        rt_xlim=(0.1, 1.0),
        rt_ylim=(0.0, 0.35),
        kde_grid=(0.1, 1.0),
        kde_points=500,
        bandwidth_scale=1.0,
        n_boot=5000,
    )


if __name__ == "__main__":
    main()