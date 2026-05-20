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


def fit_slope_intercept(points: List[Tuple[float, float]]) -> Tuple[float, float] | None:
    """
    Fit rt = slope * t + intercept using least squares.
    Returns (slope, intercept) or None if insufficient data.
    """
    if len(points) < 2:
        return None

    xs = np.asarray([t for t, _ in points], dtype=float)
    ys = np.asarray([rt for _, rt in points], dtype=float)

    ok = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[ok]
    ys = ys[ok]
    if xs.size < 2:
        return None

    if float(np.var(xs)) == 0.0:
        return None

    slope, intercept = np.polyfit(xs, ys, deg=1)
    return float(slope), float(intercept)


def compute_slopes_by_session(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]]
) -> Dict[int, Dict[int, float]]:
    """
    Returns:
        session_id -> {subject_id -> slope}
    """
    by_session: Dict[int, Dict[int, float]] = {}

    for (subject_id, session_id), points in subj_sess_points.items():
        fit = fit_slope_intercept(points)
        if fit is None:
            continue
        slope, _ = fit
        by_session.setdefault(session_id, {})[subject_id] = slope

    return by_session


def save_slope_summary_plots(
    slopes_by_session: Dict[int, Dict[int, float]],
    output_dir: Path,
    dpi: int = 300,
    bins: int = 30,
) -> None:
    """
    Produces:
      1) Slope distribution per session (hist)
      2) Percent positive vs negative slopes per session (bar chart)
      3) Change in slope from session 1 to 3 per subject (paired plot)
    """
    session_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Plot 1: Slope distributions
    # ----------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6))

    legend_items: List[Tuple[str, str]] = []
    all_slopes = []

    for session_id in sorted(slopes_by_session.keys()):
        if session_id not in session_colors:
            continue
        subj_to_slope = slopes_by_session[session_id]
        if not subj_to_slope:
            continue

        slopes = np.asarray(list(subj_to_slope.values()), dtype=float)
        slopes = slopes[np.isfinite(slopes)]
        if slopes.size == 0:
            continue

        all_slopes.append(slopes)
        color = session_colors[session_id]

        ax.hist(
            slopes,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
        )
        legend_items.append((color, f"Session {session_id} (n={slopes.size})"))

    ax.axvline(0.0, linewidth=2.0, alpha=0.8)
    ax.set_xlabel("Slope of RT over time (seconds RT per second)")
    ax.set_ylabel("Density")
    ax.grid(True)

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
        ax.legend(handles=legend_handles, fontsize=13, handlelength=1.0)

    fig.tight_layout()
    fig.savefig(output_dir / "rt_slope_distributions.png", dpi=dpi)
    plt.close(fig)

    # --------------------------------------------
    # Plot 2: Percent positive vs negative slopes
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))

    sessions = [sid for sid in sorted(slopes_by_session.keys()) if sid in session_colors]
    pos_perc = []
    neg_perc = []
    ns = []

    for sid in sessions:
        slopes = np.asarray(list(slopes_by_session.get(sid, {}).values()), dtype=float)
        slopes = slopes[np.isfinite(slopes)]
        n = int(slopes.size)
        ns.append(n)
        if n == 0:
            pos_perc.append(0.0)
            neg_perc.append(0.0)
            continue
        pos = float(np.mean(slopes > 0.0)) * 100.0
        neg = float(np.mean(slopes < 0.0)) * 100.0
        pos_perc.append(pos)
        neg_perc.append(neg)

    x = np.arange(len(sessions), dtype=float)
    width = 0.38

    ax.bar(x - width / 2.0, pos_perc, width=width, alpha=0.85, label="Positive slope")
    ax.bar(x + width / 2.0, neg_perc, width=width, alpha=0.85, label="Negative slope")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Session {sid}\n(n={n})" for sid, n in zip(sessions, ns)])
    ax.set_ylabel("Percent of subjects")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, axis="y")
    ax.legend(fontsize=13)

    fig.tight_layout()
    fig.savefig(output_dir / "rt_slope_sign_percentages.png", dpi=dpi)
    plt.close(fig)

    # -----------------------------------------------------
    # Plot 3: Change in slope from session 1 to 3 per subject
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6))

    s1 = slopes_by_session.get(1, {})
    s3 = slopes_by_session.get(3, {})

    common_subjects = sorted(set(s1.keys()) & set(s3.keys()))
    if common_subjects:
        x_vals = np.asarray([s1[subj] for subj in common_subjects], dtype=float)
        y_vals = np.asarray([s3[subj] for subj in common_subjects], dtype=float)

        ax.scatter(
            x_vals,
            y_vals,
            s=28,
            alpha=0.45,
            color="tab:blue",
        )

        lo = float(min(x_vals.min(), y_vals.min()))
        hi = float(max(x_vals.max(), y_vals.max()))
        pad = 0.02 * (hi - lo) if hi > lo else 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linewidth=2.0, alpha=0.9)

        deltas = y_vals - x_vals
        ax.set_title(f"Session 1 vs Session 3 slope per subject (n={len(common_subjects)})")
        ax.set_xlabel("Slope in Session 1 (seconds RT per second)")
        ax.set_ylabel("Slope in Session 3 (seconds RT per second)")
        ax.grid(True)

        # Simple text summary on plot
        med_delta = float(np.median(deltas))
        mean_delta = float(np.mean(deltas))
        ax.text(
            0.02,
            0.98,
            f"Median (S3 - S1) = {med_delta:.3e}\nMean (S3 - S1) = {mean_delta:.3e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No subjects found with both Session 1 and Session 3 slopes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "rt_slope_change_session1_vs_session3.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    out_dir = Path("/home/mojtaba.kananisarche/Zh/plots/rt_drift_summaries")

    subj_sess_points = load_subject_session_points(data_folder)
    slopes_by_session = compute_slopes_by_session(subj_sess_points)

    save_slope_summary_plots(
        slopes_by_session=slopes_by_session,
        output_dir=out_dir,
        dpi=300,
        bins=30,
    )


if __name__ == "__main__":
    main()