from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")


# =========================
# Config
# =========================

@dataclass(frozen=True)
class PathsConfig:
    data_folder: Path
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    rt_ylim: Tuple[float, float] = (0.1, 1.0)
    max_time_sec: Optional[float] = None  # set to something like 600 if you want a fixed x-range
    dpi: int = 300


@dataclass(frozen=True)
class QuantileRegConfig:
    quantiles: Tuple[float, ...] = (0.5, 0.75, 0.9)


@dataclass(frozen=True)
class ChangePointConfig:
    min_segment_points: int = 8
    min_improvement_frac: float = 0.05  # require improvement vs no-change SSE to accept a changepoint
    max_subject_examples: int = 0        # set >0 to also save example per-subject plots


# =========================
# Parsing and loading
# =========================

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, int, List[Tuple[float, float]]]:
    """
    Returns:
        subject_id, session_id, points (t_sec_since_start, rt_sec)
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

    points.sort(key=lambda x: x[0])
    return subject_id, session_id, points


def load_all_points_long(folder: Path) -> "pd.DataFrame":
    """
    Loads all valid CSVs into a long dataframe:
      columns: subject, session, time, rt
    """
    import pandas as pd

    rows = []
    for csv_file in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(csv_file.name):
            continue
        subject_id, session_id, points = load_session_file(csv_file)
        for t, rt in points:
            rows.append((subject_id, session_id, t, rt))

    df = pd.DataFrame(rows, columns=["subject", "session", "time", "rt"])
    return df


# =========================
# Styling
# =========================

def session_colors() -> Dict[int, str]:
    return {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}


def square_legend(ax, items: List[Tuple[str, str]], fontsize: int = 14) -> None:
    handles = [
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
        for color, label in items
    ]
    if handles:
        ax.legend(handles=handles, fontsize=fontsize, handlelength=1.0)


# =========================
# Mixed effects model
# =========================

def fit_mixed_effects(df: "pd.DataFrame"):
    """
    Mixed effects:
      rt ~ time + C(session)
      random intercept per subject: (1 | subject)

    Returns:
      result object, or None if statsmodels is unavailable or fit fails.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return None

    dff = df.copy()
    dff = dff[np.isfinite(dff["time"]) & np.isfinite(dff["rt"])]
    if dff.shape[0] < 10:
        return None

    try:
        model = smf.mixedlm("rt ~ time + C(session)", data=dff, groups=dff["subject"])
        res = model.fit(reml=False, method="lbfgs")
        return res
    except Exception:
        try:
            model = smf.mixedlm("rt ~ time + C(session)", data=dff, groups=dff["subject"])
            res = model.fit(reml=True)
            return res
        except Exception:
            return None


def predict_fixed_effects_mixed(res, sessions: List[int], t_grid: np.ndarray):
    """
    Build population-level predicted lines using fixed effects only.
    """
    import pandas as pd

    params = res.fe_params
    out: Dict[int, np.ndarray] = {}

    for sess in sessions:
        X = pd.DataFrame({"time": t_grid, "session": [sess] * t_grid.size})
        try:
            yhat = res.model.predict(params, exog=X)
        except Exception:
            base = float(params.get("Intercept", 0.0))
            time_coef = float(params.get("time", 0.0))
            add = 0.0
            if sess != 1:
                add = float(params.get(f"C(session)[T.{sess}]", 0.0))
            yhat = base + add + time_coef * t_grid
        out[sess] = np.asarray(yhat, dtype=float)

    return out


def save_mixed_effects_plot(
    df: "pd.DataFrame",
    out_path: Path,
    model_cfg: ModelConfig,
) -> None:
    colors = session_colors()

    res = fit_mixed_effects(df)

    fig, ax = plt.subplots(figsize=(11.5, 6))

    # Light scatter per session
    for sess in sorted(df["session"].unique()):
        if int(sess) not in colors:
            continue
        d = df[df["session"] == sess]
        ax.scatter(
            d["time"].to_numpy(),
            d["rt"].to_numpy(),
            s=18,
            alpha=0.08,
            color=colors[int(sess)],
        )

    # Fitted lines
    legend_items: List[Tuple[str, str]] = []
    sessions = [s for s in [1, 2, 3] if s in colors]

    t_max = float(df["time"].max()) if model_cfg.max_time_sec is None else float(model_cfg.max_time_sec)
    t_grid = np.linspace(0.0, t_max, 250)

    if res is not None:
        yhat = predict_fixed_effects_mixed(res, sessions=sessions, t_grid=t_grid)

        fe = res.fe_params
        time_coef = float(fe.get("time", np.nan))
        for sess in sessions:
            ax.plot(t_grid, yhat[sess], color=colors[sess], linewidth=2.8)
            base = float(fe.get("Intercept", 0.0))
            add = 0.0 if sess == 1 else float(fe.get(f"C(session)[T.{sess}]", 0.0))
            intercept = base + add
            legend_items.append((colors[sess], f"Session {sess} fixed: y = {time_coef:.3e} x + {intercept:.3e}"))
    else:
        ax.text(
            0.5,
            0.95,
            "Mixed effects fit unavailable (statsmodels missing or fit failed).",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Reaction time (seconds)")
    ax.set_ylim(model_cfg.rt_ylim[0], model_cfg.rt_ylim[1])
    ax.grid(True)

    if legend_items:
        square_legend(ax, legend_items, fontsize=13)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=model_cfg.dpi)
    plt.close(fig)


# =========================
# Quantile regression
# =========================

def fit_quantile_regressions(df: "pd.DataFrame", quantiles: Tuple[float, ...]):
    """
    Quantile regression:
      rt ~ time + C(session)
    Returns:
      dict q -> fitted result, skipping failures.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return {}

    dff = df.copy()
    dff = dff[np.isfinite(dff["time"]) & np.isfinite(dff["rt"])]
    if dff.shape[0] < 10:
        return {}

    res_by_q = {}
    for q in quantiles:
        try:
            mod = smf.quantreg("rt ~ time + C(session)", dff)
            res = mod.fit(q=float(q))
            res_by_q[float(q)] = res
        except Exception:
            continue
    return res_by_q


def predict_quantile_lines(res, sessions: List[int], t_grid: np.ndarray) -> Dict[int, np.ndarray]:
    import pandas as pd

    out: Dict[int, np.ndarray] = {}
    for sess in sessions:
        X = pd.DataFrame({"time": t_grid, "session": [sess] * t_grid.size})
        out[sess] = np.asarray(res.predict(X), dtype=float)
    return out


def save_quantile_regression_plot(
    df: "pd.DataFrame",
    out_path: Path,
    model_cfg: ModelConfig,
    qcfg: QuantileRegConfig,
) -> None:
    colors = session_colors()
    sessions = [s for s in [1, 2, 3] if s in colors]

    t_max = float(df["time"].max()) if model_cfg.max_time_sec is None else float(model_cfg.max_time_sec)
    t_grid = np.linspace(0.0, t_max, 250)

    res_by_q = fit_quantile_regressions(df, qcfg.quantiles)

    fig, ax = plt.subplots(figsize=(11.5, 6))

    # Light scatter per session
    for sess in sessions:
        d = df[df["session"] == sess]
        ax.scatter(
            d["time"].to_numpy(),
            d["rt"].to_numpy(),
            s=18,
            alpha=0.06,
            color=colors[sess],
        )

    if not res_by_q:
        ax.text(
            0.5,
            0.95,
            "Quantile regression fit unavailable (statsmodels missing or fit failed).",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )
    else:
        # For each session, plot multiple quantiles with different alphas
        q_sorted = sorted(res_by_q.keys())
        alpha_map = {q_sorted[0]: 0.95}
        for q in q_sorted[1:]:
            alpha_map[q] = 0.65

        legend_items: List[Tuple[str, str]] = []
        for q in q_sorted:
            res = res_by_q[q]
            lines = predict_quantile_lines(res, sessions=sessions, t_grid=t_grid)

            # Keep same color per session, but different line style per quantile
            linestyle = "-" if abs(q - 0.5) < 1e-9 else ":"
            lw = 2.8 if abs(q - 0.5) < 1e-9 else 2.0

            for sess in sessions:
                ax.plot(
                    t_grid,
                    lines[sess],
                    color=colors[sess],
                    linewidth=lw,
                    alpha=alpha_map[q],
                    linestyle=linestyle,
                )

        # Legend: show per session color plus quantile line styles
        for sess in sessions:
            legend_items.append((colors[sess], f"Session {sess}"))

        square_legend(ax, legend_items, fontsize=13)

        # Add a small line-style legend for quantiles
        style_handles = []
        style_labels = []
        style_handles.append(Line2D([0], [0], color="black", linewidth=2.8, linestyle="-"))
        style_labels.append("q=0.50")
        if any(abs(q - 0.75) < 1e-9 for q in q_sorted) or any(abs(q - 0.9) < 1e-9 for q in q_sorted):
            style_handles.append(Line2D([0], [0], color="black", linewidth=2.0, linestyle=":"))
            qs = [f"{q:.2f}" for q in q_sorted if abs(q - 0.5) > 1e-9]
            style_labels.append("q=" + ",".join(qs))
        ax.legend(style_handles, style_labels, fontsize=12, loc="upper left")

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Reaction time (seconds)")
    ax.set_ylim(model_cfg.rt_ylim[0], model_cfg.rt_ylim[1])
    ax.grid(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=model_cfg.dpi)
    plt.close(fig)


# =========================
# Change point detection (per subject-session)
# =========================

@dataclass(frozen=True)
class ChangePointResult:
    subject: int
    session: int
    change_time: float
    improvement_frac: float


def best_mean_changepoint(
    t: np.ndarray,
    y: np.ndarray,
    min_seg: int,
) -> Tuple[Optional[int], float, float]:
    """
    Find best single changepoint for piecewise constant mean:
      y ~ mean1 for i <= k, mean2 for i > k

    Returns:
      best_k, sse_best, sse_nochange
    """
    n = y.size
    if n < 2 * min_seg + 1:
        return None, np.inf, np.inf

    y = y.astype(float)
    sse0 = float(np.sum((y - y.mean()) ** 2))

    # Precompute cumulative sums for O(1) segment SSE
    csum = np.cumsum(y)
    csum2 = np.cumsum(y * y)

    def seg_sse(i0: int, i1: int) -> float:
        # inclusive i0..i1
        m = i1 - i0 + 1
        s = float(csum[i1] - (csum[i0 - 1] if i0 > 0 else 0.0))
        s2 = float(csum2[i1] - (csum2[i0 - 1] if i0 > 0 else 0.0))
        mean = s / m
        return s2 - 2.0 * mean * s + m * mean * mean

    best_k = None
    best_sse = np.inf

    for k in range(min_seg - 1, n - min_seg - 1):
        sse_left = seg_sse(0, k)
        sse_right = seg_sse(k + 1, n - 1)
        sse = sse_left + sse_right
        if sse < best_sse:
            best_sse = sse
            best_k = k

    return best_k, float(best_sse), float(sse0)


def detect_changepoints(
    df: "pd.DataFrame",
    cpcfg: ChangePointConfig,
) -> List[ChangePointResult]:
    """
    Detect one changepoint per subject-session using piecewise constant mean.
    """
    results: List[ChangePointResult] = []

    # Group by subject-session
    grouped = df.groupby(["subject", "session"], sort=False)
    for (subj, sess), g in grouped:
        t = g["time"].to_numpy(dtype=float)
        y = g["rt"].to_numpy(dtype=float)

        ok = np.isfinite(t) & np.isfinite(y)
        t = t[ok]
        y = y[ok]
        if t.size < 2 * cpcfg.min_segment_points + 1:
            continue

        order = np.argsort(t)
        t = t[order]
        y = y[order]

        k, sse_best, sse0 = best_mean_changepoint(t, y, min_seg=cpcfg.min_segment_points)
        if k is None or not np.isfinite(sse0) or sse0 <= 0:
            continue

        improvement_frac = (sse0 - sse_best) / sse0
        if improvement_frac < cpcfg.min_improvement_frac:
            continue

        change_time = float(t[k])
        results.append(ChangePointResult(int(subj), int(sess), change_time, float(improvement_frac)))

    return results


def save_changepoint_summary_plot(
    df: "pd.DataFrame",
    out_path: Path,
    model_cfg: ModelConfig,
    cpcfg: ChangePointConfig,
) -> None:
    colors = session_colors()
    cps = detect_changepoints(df, cpcfg=cpcfg)

    fig, ax = plt.subplots(figsize=(10.5, 6))

    legend_items: List[Tuple[str, str]] = []

    if not cps:
        ax.text(
            0.5,
            0.5,
            "No changepoints detected under current settings.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        # Histogram of change times per session
        for sess in [1, 2, 3]:
            times = np.asarray([cp.change_time for cp in cps if cp.session == sess], dtype=float)
            times = times[np.isfinite(times)]
            if times.size == 0 or sess not in colors:
                continue

            ax.hist(
                times,
                bins=30,
                density=True,
                alpha=0.35,
                color=colors[sess],
            )
            legend_items.append((colors[sess], f"Session {sess} (n={times.size})"))

    ax.set_xlabel("Estimated changepoint time (seconds since session start)")
    ax.set_ylabel("Density")
    ax.grid(True)
    if legend_items:
        square_legend(ax, legend_items, fontsize=13)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=model_cfg.dpi)
    plt.close(fig)


# =========================
# Main orchestration
# =========================

def run_model_based_summaries(
    paths: PathsConfig,
    model_cfg: ModelConfig,
    qcfg: QuantileRegConfig,
    cpcfg: ChangePointConfig,
) -> None:
    import pandas as pd  # noqa: F401

    df = load_all_points_long(paths.data_folder)
    df = df[np.isfinite(df["time"]) & np.isfinite(df["rt"])]

    # Ensure session is int
    df["session"] = df["session"].astype(int)
    df["subject"] = df["subject"].astype(int)

    # Save plots
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    save_mixed_effects_plot(
        df=df,
        out_path=paths.output_dir / "mixed_effects_fixed_lines.png",
        model_cfg=model_cfg,
    )

    save_quantile_regression_plot(
        df=df,
        out_path=paths.output_dir / "quantile_regression_lines.png",
        model_cfg=model_cfg,
        qcfg=qcfg,
    )

    save_changepoint_summary_plot(
        df=df,
        out_path=paths.output_dir / "changepoint_time_distribution.png",
        model_cfg=model_cfg,
        cpcfg=cpcfg,
    )


def main() -> None:
    paths = PathsConfig(
        data_folder=Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt"),
        output_dir=Path("/home/mojtaba.kananisarche/Zh/plots/rt_model_based_summaries"),
    )

    model_cfg = ModelConfig(
        rt_ylim=(0.1, 1.0),
        max_time_sec=None,
        dpi=300,
    )

    qcfg = QuantileRegConfig(
        quantiles=(0.5, 0.75, 0.9),
    )

    cpcfg = ChangePointConfig(
        min_segment_points=8,
        min_improvement_frac=0.05,
        max_subject_examples=0,
    )

    run_model_based_summaries(paths, model_cfg, qcfg, cpcfg)


if __name__ == "__main__":
    main()