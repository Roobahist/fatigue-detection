from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Constants
# =========================

TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")

KSS_PATH = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/KSS.txt")
RT_FOLDER = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
OUTPUT_PATH = Path("/home/mojtaba.kananisarche/Zh/plots/kss_objective_quadrant.png")

LAPSE_THRESHOLD_SEC = 0.5
DPI = 300


# =========================
# Parsing utilities
# =========================

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, int, List[float]]:
    """
    Returns:
        subject_id, session_id, list of RTs
    """
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename: {path.name}")

    subj_str, sess_str = path.stem.split("-")
    subject_id = int(subj_str)
    session_id = int(sess_str)

    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    rts: List[float] = []
    for line in lines[1:]:
        stim_ts, react_ts = line.split(";")
        stim_time = parse_timestamp(stim_ts)
        react_time = parse_timestamp(react_ts)
        rts.append((react_time - stim_time).total_seconds())

    return subject_id, session_id, rts


def load_all_sessions(folder: Path) -> Dict[Tuple[int, int], List[float]]:
    """
    (subject, session) -> RT list
    """
    out: Dict[Tuple[int, int], List[float]] = {}
    for f in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(f.name):
            continue
        subj, sess, rts = load_session_file(f)
        if rts:
            out[(subj, sess)] = rts
    return out


def load_kss_matrix(path: Path) -> np.ndarray:
    """
    Rows: subjects (1-indexed)
    Columns: sessions (1..3)
    """
    kss = np.loadtxt(str(path), dtype=float)
    if kss.ndim != 2 or kss.shape[1] != 3:
        raise ValueError("KSS must be (n_subjects, 3)")
    return kss


# =========================
# Metrics
# =========================

def lapse_rate(rts: List[float]) -> float:
    arr = np.asarray(rts, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > LAPSE_THRESHOLD_SEC))


def compute_subject_deltas(
    rt_data: Dict[Tuple[int, int], List[float]],
    kss: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        delta_kss, delta_lapse, subject_ids
    """
    subjects = sorted({subj for subj, _ in rt_data.keys()})

    dk, dl, sids = [], [], []

    for subj in subjects:
        if (subj, 1) not in rt_data or (subj, 3) not in rt_data:
            continue

        kss1 = kss[subj - 1, 0]
        kss3 = kss[subj - 1, 2]
        delta_kss = kss3 - kss1

        lapse1 = lapse_rate(rt_data[(subj, 1)])
        lapse3 = lapse_rate(rt_data[(subj, 3)])
        delta_lapse = lapse3 - lapse1

        if np.isfinite(delta_kss) and np.isfinite(delta_lapse):
            dk.append(delta_kss)
            dl.append(delta_lapse)
            sids.append(subj)

    return np.asarray(dk), np.asarray(dl), np.asarray(sids)


# =========================
# Plotting
# =========================

def plot_quadrant(
    delta_kss: np.ndarray,
    delta_lapse: np.ndarray,
    subject_ids: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(delta_kss, delta_lapse, s=70, color="tab:red", alpha=0.75)

    # Zero lines define quadrants
    ax.axhline(0.0, color="black", linewidth=1.5)
    ax.axvline(0.0, color="black", linewidth=1.5)

    # Annotate quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(xlim[1]*0.6, ylim[1]*0.6, "High subjective\nHigh objective", ha="center")
    ax.text(xlim[1]*0.6, ylim[0]*0.6, "High subjective\nLow objective", ha="center")
    ax.text(xlim[0]*0.6, ylim[1]*0.6, "Low subjective\nHigh objective", ha="center")
    ax.text(xlim[0]*0.6, ylim[0]*0.6, "Low subjective\nLow objective", ha="center")

    # Optional subject labels
    for x, y, sid in zip(delta_kss, delta_lapse, subject_ids):
        ax.text(x, y, f"{sid}", fontsize=8, alpha=0.7)

    ax.set_xlabel("Δ KSS (Session 3 − Session 1)")
    ax.set_ylabel("Δ Lapse rate (Session 3 − Session 1)")
    ax.set_title("Subjective vs Objective Fatigue Agreement")

    ax.grid(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


# =========================
# Main
# =========================

def main() -> None:
    rt_data = load_all_sessions(RT_FOLDER)
    kss = load_kss_matrix(KSS_PATH)

    delta_kss, delta_lapse, subject_ids = compute_subject_deltas(rt_data, kss)

    plot_quadrant(
        delta_kss,
        delta_lapse,
        subject_ids,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()