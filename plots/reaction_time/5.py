from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Tuple
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
        if points:
            out[(subject_id, session_id)] = points

    return out


def build_pooled_session_points(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]]
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Pools all subject-session points into session_id -> [(t, rt), ...]
    Time is seconds since that subject's session start.
    """
    pooled: Dict[int, List[Tuple[float, float]]] = {}
    for (_, session_id), points in subj_sess_points.items():
        pooled.setdefault(session_id, []).extend(points)
    return pooled


def compute_subject_session_lapse_thresholds(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]],
    mode: str,
    lapse_threshold_sec: float,
    lapse_percentile: float,
) -> Dict[Tuple[int, int], float]:
    """
    Returns:
        (subject_id, session_id) -> lapse threshold (seconds)
    """
    thresholds: Dict[Tuple[int, int], float] = {}

    if mode == "fixed":
        for key in subj_sess_points.keys():
            thresholds[key] = float(lapse_threshold_sec)
        return thresholds

    if mode != "percentile":
        raise ValueError("mode must be 'fixed' or 'percentile'")

    q = float(lapse_percentile)
    if not (0.0 < q < 1.0):
        raise ValueError("lapse_percentile must be in (0, 1)")

    for key, points in subj_sess_points.items():
        rts = np.asarray([rt for _, rt in points], dtype=float)
        rts = rts[np.isfinite(rts)]
        if rts.size == 0:
            thresholds[key] = float(lapse_threshold_sec)
            continue
        thresholds[key] = float(np.quantile(rts, q))

    return thresholds


def rolling_rate(
    points: List[Tuple[float, float]],
    predicate: Callable[[np.ndarray], np.ndarray],
    window_sec: float,
    step_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling fraction of points satisfying predicate(rt) within each time window.

    predicate takes rt array and returns boolean array.
    """
    times = np.asarray([t for t, _ in points], dtype=float)
    rts = np.asarray([rt for _, rt in points], dtype=float)

    ok = np.isfinite(times) & np.isfinite(rts)
    times = times[ok]
    rts = rts[ok]

    if times.size == 0:
        return np.asarray([]), np.asarray([])

    order = np.argsort(times)
    times = times[order]
    rts = rts[order]

    left = float(times.min())
    right_max = float(times.max())

    centers: List[float] = []
    rates: List[float] = []

    while left + window_sec <= right_max:
        right = left + window_sec
        mask = (times >= left) & (times < right)
        if mask.any():
            w = rts[mask]
            centers.append(left + window_sec / 2.0)
            rates.append(float(np.mean(predicate(w))))
        left += step_sec

    return np.asarray(centers), np.asarray(rates)


def rolling_mean(
    points: List[Tuple[float, float]],
    value_fn: Callable[[np.ndarray], np.ndarray],
    window_sec: float,
    step_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling mean of transformed rt values within each time window.

    value_fn can mask values by returning np.nan for excluded elements.
    """
    times = np.asarray([t for t, _ in points], dtype=float)
    rts = np.asarray([rt for _, rt in points], dtype=float)

    ok = np.isfinite(times) & np.isfinite(rts)
    times = times[ok]
    rts = rts[ok]

    if times.size == 0:
        return np.asarray([]), np.asarray([])

    order = np.argsort(times)
    times = times[order]
    rts = rts[order]

    left = float(times.min())
    right_max = float(times.max())

    centers: List[float] = []
    means: List[float] = []

    while left + window_sec <= right_max:
        right = left + window_sec
        mask = (times >= left) & (times < right)
        if mask.any():
            w = value_fn(rts[mask])
            w = w[np.isfinite(w)]
            centers.append(left + window_sec / 2.0)
            means.append(float(w.mean()) if w.size else np.nan)
        left += step_sec

    return np.asarray(centers), np.asarray(means)


def cumulative_count_over_time(
    points: List[Tuple[float, float]],
    predicate: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative count of predicate(rt) vs time (sorted by time).
    """
    times = np.asarray([t for t, _ in points], dtype=float)
    rts = np.asarray([rt for _, rt in points], dtype=float)

    ok = np.isfinite(times) & np.isfinite(rts)
    times = times[ok]
    rts = rts[ok]

    if times.size == 0:
        return np.asarray([]), np.asarray([])

    order = np.argsort(times)
    times = times[order]
    rts = rts[order]

    hits = predicate(rts).astype(int)
    cum = np.cumsum(hits)
    return times, cum.astype(float)


def save_lapse_microfailure_plots(
    subj_sess_points: Dict[Tuple[int, int], List[Tuple[float, float]]],
    output_dir: Path,
    mode: str = "fixed",                 # "fixed" or "percentile"
    lapse_threshold_sec: float = 0.5,    # classic PVT lapse threshold
    lapse_percentile: float = 0.9,       # used if mode="percentile"
    false_start_threshold_sec: float = 0.1,
    window_sec: float = 60.0,
    step_sec: float = 15.0,
    rt_ylim: Tuple[float, float] = (0.1, 1.0),
    dpi: int = 300,
    also_save_false_start_rate: bool = True,
) -> None:
    """
    Outputs (PNG):
      1) lapse_rate_over_time.png
      2) cumulative_lapses_per_session.png
      3) mean_rt_included_vs_excluded.png
      4) false_start_rate_over_time.png (optional)

    Notes:
      - Lapse definition is configurable:
          mode="fixed": lapse if RT > lapse_threshold_sec
          mode="percentile": lapse if RT > subject-session quantile (lapse_percentile)
      - False start: RT < false_start_threshold_sec
    """
    session_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = compute_subject_session_lapse_thresholds(
        subj_sess_points=subj_sess_points,
        mode=mode,
        lapse_threshold_sec=lapse_threshold_sec,
        lapse_percentile=lapse_percentile,
    )

    # Build per-session pooled points, plus per-point lapse thresholds if percentile mode
    pooled_by_session: Dict[int, List[Tuple[float, float]]] = {}
    pooled_by_session_lapse_thr: Dict[int, List[Tuple[float, float, float]]] = {}

    for (subject_id, session_id), points in subj_sess_points.items():
        pooled_by_session.setdefault(session_id, []).extend(points)
        thr = float(thresholds[(subject_id, session_id)])
        for t, rt in points:
            pooled_by_session_lapse_thr.setdefault(session_id, []).append((t, rt, thr))

    def lapse_pred_for_session(session_id: int) -> Callable[[np.ndarray], np.ndarray]:
        if mode == "fixed":
            thr = float(lapse_threshold_sec)

            def pred(rt_arr: np.ndarray) -> np.ndarray:
                return rt_arr > thr

            return pred

        triples = pooled_by_session_lapse_thr.get(session_id, [])

        times = np.asarray([t for t, _, _ in triples], dtype=float)
        rts = np.asarray([rt for _, rt, _ in triples], dtype=float)
        thrs = np.asarray([thr for _, _, thr in triples], dtype=float)

        ok = np.isfinite(times) & np.isfinite(rts) & np.isfinite(thrs)
        times = times[ok]
        rts = rts[ok]
        thrs = thrs[ok]

        if times.size == 0:
            def pred_empty(rt_arr: np.ndarray) -> np.ndarray:
                return np.zeros_like(rt_arr, dtype=bool)
            return pred_empty

        order = np.argsort(times)
        times = times[order]
        rts = rts[order]
        thrs = thrs[order]

        def pred(rt_arr: np.ndarray) -> np.ndarray:
            # This predicate is only used in rolling functions that slice by time.
            # We will not call it here directly on pooled rt arrays in percentile mode.
            raise RuntimeError("Internal misuse: percentile lapse predicate is windowed")

        return pred  # unused directly

    def rolling_lapse_rate_percentile(session_id: int) -> Tuple[np.ndarray, np.ndarray]:
        triples = pooled_by_session_lapse_thr.get(session_id, [])
        if not triples:
            return np.asarray([]), np.asarray([])

        times = np.asarray([t for t, _, _ in triples], dtype=float)
        rts = np.asarray([rt for _, rt, _ in triples], dtype=float)
        thrs = np.asarray([thr for _, _, thr in triples], dtype=float)

        ok = np.isfinite(times) & np.isfinite(rts) & np.isfinite(thrs)
        times = times[ok]
        rts = rts[ok]
        thrs = thrs[ok]

        if times.size == 0:
            return np.asarray([]), np.asarray([])

        order = np.argsort(times)
        times = times[order]
        rts = rts[order]
        thrs = thrs[order]

        left = float(times.min())
        right_max = float(times.max())

        centers: List[float] = []
        rates: List[float] = []

        while left + window_sec <= right_max:
            right = left + window_sec
            mask = (times >= left) & (times < right)
            if mask.any():
                centers.append(left + window_sec / 2.0)
                rates.append(float(np.mean(rts[mask] > thrs[mask])))
            left += step_sec

        return np.asarray(centers), np.asarray(rates)

    def rolling_mean_excluding_lapses_percentile(session_id: int) -> Tuple[np.ndarray, np.ndarray]:
        triples = pooled_by_session_lapse_thr.get(session_id, [])
        if not triples:
            return np.asarray([]), np.asarray([])

        times = np.asarray([t for t, _, _ in triples], dtype=float)
        rts = np.asarray([rt for _, rt, _ in triples], dtype=float)
        thrs = np.asarray([thr for _, _, thr in triples], dtype=float)

        ok = np.isfinite(times) & np.isfinite(rts) & np.isfinite(thrs)
        times = times[ok]
        rts = rts[ok]
        thrs = thrs[ok]

        if times.size == 0:
            return np.asarray([]), np.asarray([])

        order = np.argsort(times)
        times = times[order]
        rts = rts[order]
        thrs = thrs[order]

        left = float(times.min())
        right_max = float(times.max())

        centers: List[float] = []
        means: List[float] = []

        while left + window_sec <= right_max:
            right = left + window_sec
            mask = (times >= left) & (times < right)
            if mask.any():
                w = rts[mask]
                wthr = thrs[mask]
                keep = w <= wthr
                ww = w[keep]
                ww = ww[np.isfinite(ww)]
                centers.append(left + window_sec / 2.0)
                means.append(float(ww.mean()) if ww.size else np.nan)
            left += step_sec

        return np.asarray(centers), np.asarray(means)

    # --------------------------
    # Plot 1: Lapse rate over time
    # --------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    for session_id in sorted(pooled_by_session.keys()):
        if session_id not in session_colors:
            continue
        points = pooled_by_session[session_id]
        if not points:
            continue

        color = session_colors[session_id]

        if mode == "fixed":
            pred = lapse_pred_for_session(session_id)
            centers, rates = rolling_rate(points, pred, window_sec, step_sec)
        else:
            centers, rates = rolling_lapse_rate_percentile(session_id)

        if centers.size == 0:
            continue

        ax.plot(centers, rates, color=color, linewidth=2.6, label=f"Session {session_id}")

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Lapse rate (fraction per window)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "lapse_rate_over_time.png", dpi=dpi)
    plt.close(fig)

    # --------------------------
    # Plot 2: Cumulative lapses per session
    # --------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    for session_id in sorted(pooled_by_session.keys()):
        if session_id not in session_colors:
            continue
        points = pooled_by_session[session_id]
        if not points:
            continue

        color = session_colors[session_id]

        if mode == "fixed":
            pred = lapse_pred_for_session(session_id)
            t, cum = cumulative_count_over_time(points, pred)
        else:
            triples = pooled_by_session_lapse_thr.get(session_id, [])
            if not triples:
                continue
            times = np.asarray([tt for tt, _, _ in triples], dtype=float)
            rts = np.asarray([rr for _, rr, _ in triples], dtype=float)
            thrs = np.asarray([thr for _, _, thr in triples], dtype=float)

            ok = np.isfinite(times) & np.isfinite(rts) & np.isfinite(thrs)
            times = times[ok]
            rts = rts[ok]
            thrs = thrs[ok]
            if times.size == 0:
                continue
            order = np.argsort(times)
            times = times[order]
            rts = rts[order]
            thrs = thrs[order]
            hits = (rts > thrs).astype(int)
            t = times
            cum = np.cumsum(hits).astype(float)

        ax.plot(t, cum, color=color, linewidth=2.6, label=f"Session {session_id}")

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Cumulative lapse count")
    ax.grid(True)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_lapses_per_session.png", dpi=dpi)
    plt.close(fig)

    # --------------------------
    # Plot 3: Mean RT included vs excluded (rolling)
    # --------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    for session_id in sorted(pooled_by_session.keys()):
        if session_id not in session_colors:
            continue
        points = pooled_by_session[session_id]
        if not points:
            continue

        color = session_colors[session_id]

        # Included
        centers_all, mean_all = rolling_mean(
            points,
            value_fn=lambda w: w,
            window_sec=window_sec,
            step_sec=step_sec,
        )

        # Excluded lapses
        if mode == "fixed":
            thr = float(lapse_threshold_sec)

            def value_excl(w: np.ndarray) -> np.ndarray:
                out = w.copy()
                out[out > thr] = np.nan
                return out

            centers_ex, mean_ex = rolling_mean(points, value_fn=value_excl, window_sec=window_sec, step_sec=step_sec)
        else:
            centers_ex, mean_ex = rolling_mean_excluding_lapses_percentile(session_id)

        if centers_all.size:
            ax.plot(
                centers_all,
                mean_all,
                color=color,
                linewidth=2.4,
                alpha=0.9,
                label=f"Session {session_id} mean (all)",
            )
        if centers_ex.size:
            ax.plot(
                centers_ex,
                mean_ex,
                color=color,
                linewidth=2.4,
                alpha=0.9,
                linestyle=":",
                label=f"Session {session_id} mean (no lapses)",
            )

    ax.set_xlabel("Time since session start (seconds)")
    ax.set_ylabel("Rolling mean reaction time (seconds)")
    ax.set_ylim(rt_ylim[0], rt_ylim[1])
    ax.grid(True)
    ax.legend(fontsize=11, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_rt_included_vs_excluded.png", dpi=dpi)
    plt.close(fig)

    # --------------------------
    # Optional: False start rate over time
    # --------------------------
    if also_save_false_start_rate:
        fig, ax = plt.subplots(figsize=(11, 6))

        def fs_pred(rt_arr: np.ndarray) -> np.ndarray:
            return rt_arr < float(false_start_threshold_sec)

        for session_id in sorted(pooled_by_session.keys()):
            if session_id not in session_colors:
                continue
            points = pooled_by_session[session_id]
            if not points:
                continue

            centers, rates = rolling_rate(points, fs_pred, window_sec, step_sec)
            if centers.size == 0:
                continue

            ax.plot(
                centers,
                rates,
                color=session_colors[session_id],
                linewidth=2.6,
                label=f"Session {session_id}",
            )

        ax.set_xlabel("Time since session start (seconds)")
        ax.set_ylabel("False start rate (fraction per window)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        ax.legend(fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "false_start_rate_over_time.png", dpi=dpi)
        plt.close(fig)


def main() -> None:
    data_folder = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
    output_dir = Path("/home/mojtaba.kananisarche/Zh/plots/rt_lapses_microfailures")

    subj_sess_points = load_subject_session_points(data_folder)

    # Choose one:
    #   mode="fixed" uses lapse_threshold_sec (default 0.5s)
    #   mode="percentile" uses per subject-session lapse_percentile (default 0.90)
    save_lapse_microfailure_plots(
        subj_sess_points=subj_sess_points,
        output_dir=output_dir,
        mode="fixed",
        lapse_threshold_sec=0.5,
        lapse_percentile=0.9,
        false_start_threshold_sec=0.1,
        window_sec=60.0,
        step_sec=15.0,
        rt_ylim=(0.1, 1.0),
        also_save_false_start_rate=True,
    )


if __name__ == "__main__":
    main()