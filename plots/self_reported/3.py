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

GLOBAL_LAPSE_THRESHOLD_SEC = 0.5


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

    rts: List[float] = []
    for line in lines[1:]:
        stim_ts, react_ts = line.split(";")
        stim_time = parse_timestamp(stim_ts)
        react_time = parse_timestamp(react_ts)
        rts.append((react_time - stim_time).total_seconds())

    return subject_id, session_id, rts


def load_all_subject_sessions(folder: Path) -> Dict[Tuple[int, int], List[float]]:
    """
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
# Quantiles and tail mass
# =========================

def compute_quantiles(
    rts: np.ndarray,
    qs: Tuple[float, ...],
) -> Dict[float, float]:
    rts = rts[np.isfinite(rts)]
    if rts.size == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.quantile(rts, q)) for q in qs}


def compute_tail_mass(
    rts: np.ndarray,
    threshold: float,
) -> float:
    rts = rts[np.isfinite(rts)]
    if rts.size == 0:
        return float("nan")
    return float(np.mean(rts > threshold))


# =========================
# Plot helpers
# =========================

def theil_sen_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str) -> None:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return
    model = TheilSenRegressor(random_state=0)
    model.fit(x[m].reshape(-1, 1), y[m])
    xline = np.linspace(float(x[m].min()), float(x[m].max()), 200)
    yline = model.predict(xline.reshape(-1, 1))
    ax.plot(xline, yline, linewidth=2.5, alpha=0.9, label=label)


# =========================
# Main plotting functions
# =========================

def save_rt_quantiles_vs_kss(
    summaries: Dict[Tuple[int, int], Dict[float, float]],
    kss: np.ndarray,
    output_path: Path,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    quantile_styles = {
        0.50: ("Median (50%)", "tab:blue"),
        0.75: ("75th percentile", "tab:orange"),
        0.90: ("90th percentile", "tab:red"),
    }

    for q, (label, color) in quantile_styles.items():
        xs, ys = [], []
        for (subj, sess), qvals in summaries.items():
            x = kss[subj - 1, sess - 1]
            y = qvals[q]
            xs.append(x)
            ys.append(y)

        xs = np.asarray(xs)
        ys = np.asarray(ys)

        ax.scatter(xs, ys, s=45, alpha=0.6, color=color)
        theil_sen_line(ax, xs, ys, label=label)

    ax.set_xlabel("KSS level")
    ax.set_ylabel("Reaction time (seconds)")
    ax.grid(True)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_tail_mass_vs_kss(
    tail_mass: Dict[Tuple[int, int], float],
    kss: np.ndarray,
    output_path: Path,
    dpi: int = 300,
) -> None:
    xs, ys = [], []

    for (subj, sess), val in tail_mass.items():
        xs.append(kss[subj - 1, sess - 1])
        ys.append(val)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.scatter(xs, ys, s=55, color="tab:red", alpha=0.6)
    theil_sen_line(ax, xs, ys, label="Tail mass trend")

    ax.set_xlabel("KSS level")
    ax.set_ylabel("Fraction of RTs > 500 ms")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# =========================
# Orchestration
# =========================

def main() -> None:
    rt_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    kss_file = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/KSS.txt")
    output_dir = Path("/home/mojtaba.kananisarche/Zh/plots/kss_tail_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = load_all_subject_sessions(rt_folder)
    kss = load_kss_matrix(kss_file)

    quantiles = (0.50, 0.75, 0.90)

    quantile_summaries: Dict[Tuple[int, int], Dict[float, float]] = {}
    tail_mass_summaries: Dict[Tuple[int, int], float] = {}

    for key, rts in trials.items():
        arr = np.asarray(rts, dtype=float)
        quantile_summaries[key] = compute_quantiles(arr, quantiles)
        tail_mass_summaries[key] = compute_tail_mass(arr, GLOBAL_LAPSE_THRESHOLD_SEC)

    save_rt_quantiles_vs_kss(
        quantile_summaries,
        kss,
        output_dir / "rt_quantiles_vs_kss.png",
    )

    save_tail_mass_vs_kss(
        tail_mass_summaries,
        kss,
        output_dir / "tail_mass_vs_kss.png",
    )


if __name__ == "__main__":
    main()