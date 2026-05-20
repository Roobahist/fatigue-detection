from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")

LAPSE_THRESHOLD_SEC = 0.5


# =========================
# Parsing utilities
# =========================

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, int, List[float]]:
    """
    Returns:
        subject_id
        session_id
        list of reaction times (seconds)
    """
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename: {path.name}")

    subject_str, session_str = path.stem.split("-")
    subject_id = int(subject_str)
    session_id = int(session_str)

    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    session_start = parse_timestamp(lines[0])

    rts: List[float] = []
    for line in lines[1:]:
        stim_ts, react_ts = line.split(";")
        stim_time = parse_timestamp(stim_ts)
        react_time = parse_timestamp(react_ts)
        rts.append((react_time - stim_time).total_seconds())

    return subject_id, session_id, rts


def load_all_subject_sessions(folder: Path) -> Dict[Tuple[int, int], List[float]]:
    """
    Returns:
        (subject_id, session_id) -> list of RTs
    """
    out: Dict[Tuple[int, int], List[float]] = {}
    for f in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(f.name):
            continue
        subj, sess, rts = load_session_file(f)
        if rts:
            out[(subj, sess)] = rts
    return out


def load_kss_matrix(txt_path: Path) -> np.ndarray:
    """
    Rows: subjects (1 indexed)
    Columns: sessions (1..3)
    """
    kss = np.loadtxt(str(txt_path), dtype=float)
    if kss.ndim != 2 or kss.shape[1] != 3:
        raise ValueError("KSS file must have shape (n_subjects, 3)")
    return kss


# =========================
# Aggregation
# =========================

def summarize_subject_session(
    data: Dict[Tuple[int, int], List[float]]
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Returns:
        (subject, session) -> {
            "mean_rt": float,
            "median_rt": float,
            "lapse_rate": float
        }
    """
    out: Dict[Tuple[int, int], Dict[str, float]] = {}

    for key, rts in data.items():
        arr = np.asarray(rts, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        out[key] = {
            "mean_rt": float(arr.mean()),
            "median_rt": float(np.median(arr)),
            "lapse_rate": float(np.mean(arr > LAPSE_THRESHOLD_SEC)),
        }

    return out


# =========================
# Plotting
# =========================

def _plot_within_subject_lines(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    subjects: np.ndarray,
    color: str,
    alpha: float = 0.35,
) -> None:
    for subj in np.unique(subjects):
        mask = subjects == subj
        if np.sum(mask) >= 2:
            ax.plot(xs[mask], ys[mask], color=color, alpha=alpha, linewidth=1.2)


def _add_theil_sen(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return

    model = TheilSenRegressor(random_state=0)
    model.fit(x[mask].reshape(-1, 1), y[mask])

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color="black", linewidth=2.5, alpha=0.9)


def save_rt_and_lapse_vs_kss_plots(
    summaries: Dict[Tuple[int, int], Dict[str, float]],
    kss: np.ndarray,
    output_dir: Path,
    use_median: bool = True,
    dpi: int = 300,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    session_markers = {1: "o", 2: "s", 3: "D"}
    session_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}

    # Build flat arrays
    xs, ys_rt, ys_lapse, subjects, sessions = [], [], [], [], []

    for (subj, sess), vals in summaries.items():
        if sess not in (1, 2, 3):
            continue
        xs.append(kss[subj - 1, sess - 1])
        ys_rt.append(vals["median_rt"] if use_median else vals["mean_rt"])
        ys_lapse.append(vals["lapse_rate"])
        subjects.append(subj)
        sessions.append(sess)

    xs = np.asarray(xs)
    ys_rt = np.asarray(ys_rt)
    ys_lapse = np.asarray(ys_lapse)
    subjects = np.asarray(subjects)
    sessions = np.asarray(sessions)

    # -------- Plot 1.1: RT vs KSS --------
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for sess in (1, 2, 3):
        mask = sessions == sess
        ax.scatter(
            xs[mask],
            ys_rt[mask],
            s=55,
            marker=session_markers[sess],
            color=session_colors[sess],
            label=f"Session {sess}",
        )
        _plot_within_subject_lines(
            ax,
            xs[mask],
            ys_rt[mask],
            subjects[mask],
            color=session_colors[sess],
        )

    _add_theil_sen(ax, xs, ys_rt)

    ax.set_xlabel("KSS level")
    ylabel = "Median RT (seconds)" if use_median else "Mean RT (seconds)"
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(output_dir / "rt_vs_kss_within_subject.png", dpi=dpi)
    plt.close(fig)

    # -------- Plot 1.2: Lapse rate vs KSS --------
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for sess in (1, 2, 3):
        mask = sessions == sess
        ax.scatter(
            xs[mask],
            ys_lapse[mask],
            s=55,
            marker=session_markers[sess],
            color=session_colors[sess],
            label=f"Session {sess}",
        )
        _plot_within_subject_lines(
            ax,
            xs[mask],
            ys_lapse[mask],
            subjects[mask],
            color=session_colors[sess],
        )

    _add_theil_sen(ax, xs, ys_lapse)

    ax.set_xlabel("KSS level")
    ax.set_ylabel("Lapse rate (RT > 500 ms)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(output_dir / "lapse_rate_vs_kss_within_subject.png", dpi=dpi)
    plt.close(fig)


# =========================
# Main
# =========================

def main() -> None:
    rt_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    kss_file = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/KSS.txt")
    output_dir = Path("/home/mojtaba.kananisarche/Zh/plots/kss_vs_behavior")

    raw_data = load_all_subject_sessions(rt_folder)
    summaries = summarize_subject_session(raw_data)
    kss = load_kss_matrix(kss_file)

    save_rt_and_lapse_vs_kss_plots(
        summaries,
        kss,
        output_dir,
        use_median=True,
        dpi=300,
    )


if __name__ == "__main__":
    main()