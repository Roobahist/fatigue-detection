from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")

LAPSE_THRESHOLD_SEC = 0.5


# =========================
# Config
# =========================

@dataclass(frozen=True)
class PathsConfig:
    rt_folder: Path
    kss_file: Path
    output_dir: Path


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 300
    figsize: Tuple[float, float] = (8.5, 6.0)

    # Robust regression on the residual-vs-KSS plot
    add_theil_sen: bool = True

    # If you want to cap extreme RTs before fitting drift, set e.g. (0.1, 2.0).
    rt_clip: Tuple[float, float] | None = None

    # How to summarize residuals per subject-session for the final scatter
    # "mean" or "median"
    residual_summary: str = "mean"

    # Lapse drift model:
    # Use a smoothed lapse probability over time (kernel) then detrend linearly
    lapse_kernel_bandwidth_sec: float = 60.0
    lapse_grid_points: int = 200


# =========================
# Data loading
# =========================

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Returns:
        subject_id, session_id, times_sec (stimulus time since start), rts_sec
    """
    if not FILENAME_REGEX.match(path.name):
        raise ValueError(f"Invalid filename: {path.name}")

    subject_str, session_str = path.stem.split("-")
    subject_id = int(subject_str)
    session_id = int(session_str)

    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    session_start = parse_timestamp(lines[0])

    ts: List[float] = []
    rts: List[float] = []

    for line in lines[1:]:
        stim_ts, react_ts = line.split(";")
        stim_time = parse_timestamp(stim_ts)
        react_time = parse_timestamp(react_ts)

        t = (stim_time - session_start).total_seconds()
        rt = (react_time - stim_time).total_seconds()

        ts.append(float(t))
        rts.append(float(rt))

    t_arr = np.asarray(ts, dtype=float)
    rt_arr = np.asarray(rts, dtype=float)

    order = np.argsort(t_arr)
    return subject_id, session_id, t_arr[order], rt_arr[order]


def load_all_trials(folder: Path) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    (subject, session) -> (t_sec, rt_sec)
    """
    out: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for f in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(f.name):
            continue
        subj, sess, t, rt = load_session_file(f)
        if t.size == 0:
            continue
        out[(subj, sess)] = (t, rt)
    return out


def load_kss_matrix(txt_path: Path) -> np.ndarray:
    kss = np.loadtxt(str(txt_path), dtype=float)
    if kss.ndim != 2 or kss.shape[1] != 3:
        raise ValueError("KSS file must have shape (n_subjects, 3)")
    return kss


# =========================
# Helpers
# =========================

def theil_sen_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return
    model = TheilSenRegressor(random_state=0)
    model.fit(x[m].reshape(-1, 1), y[m])
    xline = np.linspace(float(np.min(x[m])), float(np.max(x[m])), 200)
    yline = model.predict(xline.reshape(-1, 1))
    ax.plot(xline, yline, color="black", linewidth=2.5, alpha=0.9)


def summarize(arr: np.ndarray, mode: str) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if mode == "median":
        return float(np.median(arr))
    if mode == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unknown residual_summary: {mode}")


def session_style():
    return (
        {1: "tab:blue", 2: "tab:orange", 3: "tab:green"},
        {1: "o", 2: "s", 3: "D"},
    )


# =========================
# 2.1 RT residuals vs KSS
# =========================

def fit_linear_drift_and_residuals(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Simple linear drift model: y = a + b t
    Residuals: y - (a + b t)
    """
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 3:
        return np.full_like(y, np.nan, dtype=float)

    X = np.column_stack([np.ones_like(t), t])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return y - yhat


def compute_subject_session_rt_residual_summary(
    trials: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    plot_cfg: PlotConfig,
) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}

    for key, (t, rt) in trials.items():
        rt_use = rt.copy()
        if plot_cfg.rt_clip is not None:
            lo, hi = plot_cfg.rt_clip
            rt_use = np.clip(rt_use, lo, hi)

        res = fit_linear_drift_and_residuals(t, rt_use)
        out[key] = summarize(res, plot_cfg.residual_summary)

    return out


# =========================
# 2.2 Lapse residuals vs KSS
# =========================

def gaussian_kernel(x: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-0.5 * (x / bandwidth) ** 2)


def smooth_lapse_prob_over_time(
    t: np.ndarray,
    lapse: np.ndarray,
    grid: np.ndarray,
    bandwidth_sec: float,
) -> np.ndarray:
    """
    Kernel-smoothed lapse probability p(t) = E[lapse | time=t]
    """
    out = np.empty_like(grid, dtype=float)
    for i, tg in enumerate(grid):
        w = gaussian_kernel(t - tg, bandwidth_sec)
        wsum = float(np.sum(w))
        out[i] = float(np.sum(w * lapse) / wsum) if wsum > 0 else np.nan
    return out


def linear_residuals_on_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit y = a + b x on finite points and return residuals on those points.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return np.full_like(y, np.nan, dtype=float)

    X = np.column_stack([np.ones_like(x[m]), x[m]])
    beta, _, _, _ = np.linalg.lstsq(X, y[m], rcond=None)
    yhat = beta[0] + beta[1] * x
    return y - yhat


def compute_subject_session_lapse_residual_summary(
    trials: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    plot_cfg: PlotConfig,
) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}

    for key, (t, rt) in trials.items():
        t = t[np.isfinite(t) & np.isfinite(rt)]
        rt = rt[np.isfinite(t) & np.isfinite(rt)]
        if t.size < 5:
            out[key] = float("nan")
            continue

        lapse = (rt > LAPSE_THRESHOLD_SEC).astype(float)

        t_min, t_max = float(np.min(t)), float(np.max(t))
        grid = np.linspace(t_min, t_max, int(plot_cfg.lapse_grid_points))

        p = smooth_lapse_prob_over_time(t, lapse, grid, plot_cfg.lapse_kernel_bandwidth_sec)
        res = linear_residuals_on_grid(grid, p)

        # Summarize residual lapse probability over time into one number per subject-session
        out[key] = summarize(res, plot_cfg.residual_summary)

    return out


# =========================
# Plotting
# =========================

def plot_residuals_vs_kss(
    residual_by_subj_sess: Dict[Tuple[int, int], float],
    kss: np.ndarray,
    out_path: Path,
    plot_cfg: PlotConfig,
    ylabel: str,
) -> None:
    colors, markers = session_style()

    xs, ys, subs, sess = [], [], [], []
    for (subj, session), val in residual_by_subj_sess.items():
        if session not in (1, 2, 3):
            continue
        if subj < 1 or subj > kss.shape[0]:
            continue
        x = float(kss[subj - 1, session - 1])
        xs.append(x)
        ys.append(val)
        subs.append(subj)
        sess.append(session)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    subs = np.asarray(subs, dtype=int)
    sess = np.asarray(sess, dtype=int)

    fig, ax = plt.subplots(figsize=plot_cfg.figsize)

    # Scatter per session + within-subject lines (connecting sessions)
    for s in (1, 2, 3):
        m = sess == s
        ax.scatter(
            xs[m],
            ys[m],
            s=55,
            marker=markers[s],
            color=colors[s],
            label=f"Session {s}",
        )

    # Connect within subject across sessions
    for subj in np.unique(subs):
        m = subs == subj
        if np.sum(m) < 2:
            continue
        order = np.argsort(xs[m])
        ax.plot(xs[m][order], ys[m][order], color="gray", alpha=0.35, linewidth=1.2)

    if plot_cfg.add_theil_sen:
        theil_sen_line(ax, xs, ys)

    ax.set_xlabel("KSS level")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(fontsize=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_cfg.dpi)
    plt.close(fig)


# =========================
# Orchestration
# =========================

def run(paths: PathsConfig, plot_cfg: PlotConfig) -> None:
    trials = load_all_trials(paths.rt_folder)
    kss = load_kss_matrix(paths.kss_file)

    # 2.1 RT residuals vs KSS
    rt_res_summary = compute_subject_session_rt_residual_summary(trials, plot_cfg)
    plot_residuals_vs_kss(
        rt_res_summary,
        kss,
        paths.output_dir / "rt_residuals_vs_kss.png",
        plot_cfg,
        ylabel=f"RT residual ({plot_cfg.residual_summary}) after removing linear time drift (s)",
    )

    # 2.2 Lapse residuals vs KSS
    lapse_res_summary = compute_subject_session_lapse_residual_summary(trials, plot_cfg)
    plot_residuals_vs_kss(
        lapse_res_summary,
        kss,
        paths.output_dir / "lapse_residuals_vs_kss.png",
        plot_cfg,
        ylabel=f"Lapse-prob residual ({plot_cfg.residual_summary}) after removing linear time drift",
    )


def main() -> None:
    paths = PathsConfig(
        rt_folder=Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt"),
        kss_file=Path("/work/long_lab/Mojtaba/data/DROZY/data/original/KSS.txt"),
        output_dir=Path("/home/mojtaba.kananisarche/Zh/plots/kss_time_detrended"),
    )

    plot_cfg = PlotConfig(
        dpi=300,
        figsize=(8.5, 6.0),
        add_theil_sen=True,
        rt_clip=None,  # e.g. (0.1, 2.0) if you want to limit extreme RTs
        residual_summary="mean",  # "mean" or "median"
        lapse_kernel_bandwidth_sec=60.0,
        lapse_grid_points=200,
    )

    run(paths, plot_cfg)


if __name__ == "__main__":
    main()