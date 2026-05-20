from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# =========================
# Constants
# =========================

TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"
FILENAME_REGEX = re.compile(r"^\d+-\d+\.csv$")

DEFAULT_RT_FOLDER = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/pvt-rt")
DEFAULT_KSS_FILE = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/KSS.txt")
DEFAULT_OUTPUT_DIR = Path("/home/mojtaba.kananisarche/Zh/plots/reaction_time/kss_rt_boxplots_horizontal")

SortMode = Literal[
    "subject_then_session",
    "subject_then_kss",
    "subject_then_median_rt",
    "session_then_kss",
    "median_rt_only",
    "kss_then_median_rt",
]


# =========================
# Layout and style with inch control
# =========================

@dataclass(frozen=True)
class BoxplotLayout:
    # Main plot axis size in inches (exact)
    ax_width_in: float = 14.0
    ax_height_in: float = 12.0

    # Figure margins in inches
    margin_left_in: float = 4.0
    margin_right_in: float = 0.6
    margin_top_in: float = 0.8
    margin_bottom_in: float = 0.8

    # Group spacing
    # Number of blank slots inserted after each change in the primary sort key (subject, session, or KSS)
    primary_group_gap_slots: int = 1

    # Which modes should receive primary-group gaps
    # median_rt_only is excluded by default
    gap_enabled_modes: Tuple[SortMode, ...] = (
        "subject_then_session",
        "subject_then_kss",
        "subject_then_median_rt",
        "session_then_kss",
        "kss_then_median_rt",
    )

    # Vertical legend (9x1 mini-heatmap)
    show_legend: bool = True
    legend_cell_width_in: float = 0.35
    legend_cell_height_in: float = 0.35
    legend_pad_in: float = 0.45
    legend_title_pad_in: float = 0.18

    dpi: int = 300


@dataclass(frozen=True)
class BoxplotStyle:
    title_prefix: str = "RT boxplots colored by KSS"
    xlabel: str = "Reaction time (seconds)"
    ylabel: str = "Subject-Session"

    # Tick controls
    tick_fontsize: int = 8
    x_tick_fontsize: int = 10

    # Grid
    show_grid: bool = True
    grid_alpha: float = 0.35

    # Boxplot cosmetics
    box_linewidth: float = 1.1
    median_linewidth: float = 1.8
    whisker_linewidth: float = 1.1
    cap_linewidth: float = 1.1

    show_fliers: bool = True
    flier_markersize: float = 3.0
    flier_alpha: float = 0.35

    # Axis limits
    xlim: Optional[Tuple[float, float]] = (0.1, 1.2)

    # Legend, exactly like your heatmap style
    legend_label: str = "KSS"
    legend_label_fontsize: int = 10
    legend_tick_fontsize: int = 9
    legend_ticks: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    legend_show_grid: bool = True
    legend_grid_color: str = "black"
    legend_grid_linewidth: float = 0.6
    legend_grid_alpha: float = 0.35


# =========================
# Loading
# =========================

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT)


def load_session_file(path: Path) -> Tuple[int, int, np.ndarray]:
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

    arr = np.asarray(rts, dtype=float)
    arr = arr[np.isfinite(arr)]
    return subject_id, session_id, arr


def load_all_subject_sessions(folder: Path) -> Dict[Tuple[int, int], np.ndarray]:
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for f in folder.glob("*.csv"):
        if not FILENAME_REGEX.match(f.name):
            continue
        subj, sess, rts = load_session_file(f)
        if rts.size > 0:
            out[(subj, sess)] = rts
    return out


def load_kss_matrix(path: Path) -> np.ndarray:
    kss = np.loadtxt(str(path), dtype=float)
    if kss.ndim != 2 or kss.shape[1] != 3:
        raise ValueError("KSS must have shape (n_subjects, 3)")
    return kss


# =========================
# Colors (discrete bwr 9)
# =========================

def make_discrete_bwr_9() -> Tuple[ListedColormap, BoundaryNorm]:
    base = plt.get_cmap("bwr")
    colors = base(np.linspace(0.0, 1.0, 9))
    cmap = ListedColormap(colors, name="bwr_9")
    boundaries = np.arange(0.5, 10.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def kss_to_rgba(
    kss_value: float,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    missing: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Tuple[float, float, float, float]:
    if not np.isfinite(kss_value):
        return missing
    v = float(kss_value)
    if v < 1.0 or v > 9.0:
        return missing
    return tuple(cmap(norm(v)))


# =========================
# Axes placement in inches
# =========================

def _add_axes_in_inches(fig: plt.Figure, rect_in: Tuple[float, float, float, float]) -> plt.Axes:
    fig_w, fig_h = fig.get_size_inches()
    left_in, bottom_in, width_in, height_in = rect_in
    return fig.add_axes(
        [
            left_in / fig_w,
            bottom_in / fig_h,
            width_in / fig_w,
            height_in / fig_h,
        ]
    )


# =========================
# Group records and sorting
# =========================

@dataclass(frozen=True)
class GroupRecord:
    subject: int
    session: int
    kss: float
    median_rt: float
    rts: np.ndarray
    delta_median_rt_s3_minus_s1: float


def _median_or_nan(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def build_group_records(
    rt_data: Dict[Tuple[int, int], np.ndarray],
    kss_mat: np.ndarray,
) -> List[GroupRecord]:
    subj_to_med_s1: Dict[int, float] = {}
    subj_to_med_s3: Dict[int, float] = {}
    for (subj, sess), rts in rt_data.items():
        if sess == 1:
            subj_to_med_s1[subj] = _median_or_nan(rts)
        elif sess == 3:
            subj_to_med_s3[subj] = _median_or_nan(rts)

    subj_delta: Dict[int, float] = {}
    for subj in set(list(subj_to_med_s1.keys()) + list(subj_to_med_s3.keys())):
        m1 = subj_to_med_s1.get(subj, float("nan"))
        m3 = subj_to_med_s3.get(subj, float("nan"))
        subj_delta[subj] = (m3 - m1) if (np.isfinite(m1) and np.isfinite(m3)) else float("nan")

    records: List[GroupRecord] = []
    for (subj, sess), rts in rt_data.items():
        kss_val = float("nan")
        if 1 <= subj <= kss_mat.shape[0] and 1 <= sess <= kss_mat.shape[1]:
            kss_val = float(kss_mat[subj - 1, sess - 1])
        med = _median_or_nan(rts)
        dmed = subj_delta.get(subj, float("nan"))
        records.append(GroupRecord(subj, sess, kss_val, med, rts, dmed))
    return records


def sort_records(records: List[GroupRecord], mode: SortMode) -> List[GroupRecord]:
    def kss_key(v: float) -> float:
        return v if np.isfinite(v) else 1e9

    def med_key(v: float) -> float:
        return v if np.isfinite(v) else 1e9

    if mode == "subject_then_session":
        key = lambda r: (r.subject, r.session)
    elif mode == "subject_then_kss":
        key = lambda r: (r.subject, kss_key(r.kss), r.session)
    elif mode == "subject_then_median_rt":
        key = lambda r: (r.subject, med_key(r.median_rt), r.session)
    elif mode == "session_then_kss":
        key = lambda r: (r.session, kss_key(r.kss), r.subject)
    elif mode == "median_rt_only":
        key = lambda r: (med_key(r.median_rt), r.subject, r.session)
    elif mode == "kss_then_median_rt":
        key = lambda r: (kss_key(r.kss), med_key(r.median_rt), r.subject, r.session)
    else:
        raise ValueError(f"Unknown sort mode: {mode}")

    return sorted(records, key=key)


def primary_group_key(rec: GroupRecord, mode: SortMode) -> Optional[float]:
    """
    Return the primary sort dimension that should receive a gap.
    If None, no gaps should be added for this mode.
    """
    if mode == "subject_then_session":
        return float(rec.subject)
    if mode == "subject_then_kss":
        return float(rec.subject)
    if mode == "subject_then_median_rt":
        return float(rec.subject)
    if mode == "session_then_kss":
        return float(rec.session)
    if mode == "kss_then_median_rt":
        # Primary is KSS class
        return float(rec.kss) if np.isfinite(rec.kss) else 1e9
    if mode == "median_rt_only":
        return None
    return None


# =========================
# Legend exactly like your code (vertical 9x1, ticks on right)
# =========================

def add_vertical_kss_legend_exact(
    fig: plt.Figure,
    layout: BoxplotLayout,
    style: BoxplotStyle,
    plot_left_in: float,
    plot_bottom_in: float,
    plot_width_in: float,
    plot_height_in: float,
) -> None:
    if not layout.show_legend:
        return

    cmap, norm = make_discrete_bwr_9()

    legend_w_in = layout.legend_cell_width_in
    legend_h_in = 9.0 * layout.legend_cell_height_in

    leg_left_in = plot_left_in + plot_width_in + layout.legend_pad_in
    leg_bottom_in = plot_bottom_in + (plot_height_in - legend_h_in) / 2.0

    ax_leg = _add_axes_in_inches(fig, (leg_left_in, leg_bottom_in, legend_w_in, legend_h_in))

    legend_vals = np.arange(9, 0, -1, dtype=float).reshape(9, 1)
    x_e = np.arange(0, 2, 1, dtype=float)
    y_e = np.arange(0, 10, 1, dtype=float)

    ax_leg.pcolormesh(x_e, y_e, legend_vals, cmap=cmap, norm=norm, shading="flat")
    ax_leg.invert_yaxis()
    ax_leg.set_xticks([])

    tick_vals = list(style.legend_ticks)
    tick_pos = [(9 - v) + 0.5 for v in tick_vals]
    ax_leg.set_yticks(tick_pos)
    ax_leg.set_yticklabels([str(v) for v in tick_vals], fontsize=style.legend_tick_fontsize)

    ax_leg.yaxis.tick_right()
    ax_leg.yaxis.set_label_position("right")

    if style.legend_show_grid:
        ax_leg.set_xticks(np.arange(0, 2, 1), minor=True)
        ax_leg.set_yticks(np.arange(0, 10, 1), minor=True)
        ax_leg.grid(
            which="minor",
            color=style.legend_grid_color,
            linewidth=style.legend_grid_linewidth,
            alpha=style.legend_grid_alpha,
        )
        ax_leg.tick_params(which="minor", bottom=False, left=False)

    ax_leg.set_title(style.legend_label, fontsize=style.legend_label_fontsize, pad=6)

    for spine in ax_leg.spines.values():
        spine.set_visible(False)


# =========================
# Plotting: horizontal boxplots with optional gaps after primary groups
# =========================

def inject_primary_group_gaps(
    ordered: List[GroupRecord],
    mode: SortMode,
    gap_slots: int,
    enabled: bool,
) -> Tuple[List[Optional[GroupRecord]], List[bool]]:
    """
    Returns:
        expanded list with None entries for blank slots
        is_gap list (True for gaps)
    """
    if not enabled or gap_slots <= 0:
        return list(ordered), [False] * len(ordered)

    expanded: List[Optional[GroupRecord]] = []
    is_gap: List[bool] = []

    prev_key: Optional[float] = None
    for rec in ordered:
        k = primary_group_key(rec, mode)
        # If mode has no primary gap logic, do nothing
        if k is None:
            expanded.append(rec)
            is_gap.append(False)
            continue

        if prev_key is None:
            expanded.append(rec)
            is_gap.append(False)
            prev_key = k
            continue

        if k != prev_key:
            for _ in range(gap_slots):
                expanded.append(None)
                is_gap.append(True)
            prev_key = k

        expanded.append(rec)
        is_gap.append(False)

    return expanded, is_gap


def save_horizontal_boxplot_colored_by_kss(
    records: List[GroupRecord],
    out_path: Path,
    sort_mode: SortMode,
    layout: BoxplotLayout,
    style: BoxplotStyle,
) -> None:
    ordered = sort_records(records, sort_mode)

    use_gaps = sort_mode in layout.gap_enabled_modes
    expanded, is_gap = inject_primary_group_gaps(
        ordered=ordered,
        mode=sort_mode,
        gap_slots=layout.primary_group_gap_slots,
        enabled=use_gaps,
    )

    cmap, norm = make_discrete_bwr_9()
    missing = (1.0, 1.0, 1.0, 1.0)

    # Build data/labels/colors only for real records, but reserve positions for gaps
    data: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[Tuple[float, float, float, float]] = []
    positions: List[int] = []

    pos = 1
    for rec, gap in zip(expanded, is_gap):
        if gap or rec is None:
            pos += 1
            continue
        data.append(rec.rts)
        labels.append(f"{rec.subject}-{rec.session}  ({int(rec.kss) if np.isfinite(rec.kss) else 'NA'})")
        colors.append(kss_to_rgba(rec.kss, cmap, norm, missing))
        positions.append(pos)
        pos += 1

    n_slots = len(expanded)

    plot_left_in = layout.margin_left_in
    plot_bottom_in = layout.margin_bottom_in
    plot_width_in = layout.ax_width_in
    plot_height_in = layout.ax_height_in

    legend_block_w_in = 0.0
    if layout.show_legend:
        legend_block_w_in = layout.legend_pad_in + layout.legend_cell_width_in

    fig_w_in = layout.margin_left_in + layout.ax_width_in + legend_block_w_in + layout.margin_right_in
    fig_h_in = layout.margin_bottom_in + layout.ax_height_in + layout.margin_top_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=layout.dpi)
    ax = _add_axes_in_inches(fig, (plot_left_in, plot_bottom_in, plot_width_in, plot_height_in))

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        positions=positions,
        showfliers=style.show_fliers,
        widths=0.7,
    )

    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_edgecolor("black")
        box.set_linewidth(style.box_linewidth)

    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(style.median_linewidth)

    for w in bp["whiskers"]:
        w.set_color("black")
        w.set_linewidth(style.whisker_linewidth)

    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(style.cap_linewidth)

    if style.show_fliers and "fliers" in bp:
        for fl in bp["fliers"]:
            fl.set_markersize(style.flier_markersize)
            fl.set_alpha(style.flier_alpha)
            fl.set_markeredgecolor("black")

    # All y ticks, including empty slots for gaps
    ax.set_yticks(np.arange(1, n_slots + 1))

    # Map labels onto their positions, blank for gaps
    slot_labels = [""] * n_slots
    for p, lab in zip(positions, labels):
        slot_labels[p - 1] = lab
    ax.set_yticklabels(slot_labels, fontsize=style.tick_fontsize)

    ax.set_title(f"{style.title_prefix} | order: {sort_mode}")
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)

    ax.tick_params(axis="x", labelsize=style.x_tick_fontsize)

    if style.show_grid:
        ax.grid(True, axis="x", alpha=style.grid_alpha)

    if style.xlim is not None:
        ax.set_xlim(style.xlim[0], style.xlim[1])

    add_vertical_kss_legend_exact(
        fig=fig,
        layout=layout,
        style=style,
        plot_left_in=plot_left_in,
        plot_bottom_in=plot_bottom_in,
        plot_width_in=plot_width_in,
        plot_height_in=plot_height_in,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=layout.dpi)
    plt.close(fig)


# =========================
# Orchestration
# =========================

def generate_all_sort_orders(
    rt_folder: Path,
    kss_file: Path,
    output_dir: Path,
    layout: BoxplotLayout,
    style: BoxplotStyle,
    sort_modes: Optional[Iterable[SortMode]] = None,
) -> None:
    rt_data = load_all_subject_sessions(rt_folder)
    kss = load_kss_matrix(kss_file)
    records = build_group_records(rt_data, kss)

    if sort_modes is None:
        sort_modes = [
            "subject_then_session",
            "subject_then_kss",
            "subject_then_median_rt",
            "session_then_kss",
            "median_rt_only",
            "kss_then_median_rt",
        ]

    for mode in sort_modes:
        out_path = output_dir / f"rt_boxplots_kss_horizontal_order_{mode}.png"
        save_horizontal_boxplot_colored_by_kss(records, out_path, mode, layout, style)


def main() -> None:
    layout = BoxplotLayout(
        ax_width_in=7,
        ax_height_in=7,
        margin_left_in=1.4,
        margin_right_in=0.7,
        margin_top_in=0.7,
        margin_bottom_in=0.7,
        primary_group_gap_slots=1,  # change this
        show_legend=True,
        legend_cell_width_in=0.42,
        legend_cell_height_in=0.42,
        legend_pad_in=0.6,
        dpi=300,
    )

    style = BoxplotStyle(
        title_prefix="RT boxplots colored by KSS",
        xlabel="Reaction time (seconds)",
        ylabel="Subject Index - Session Index (KSS)",
        tick_fontsize=8,
        x_tick_fontsize=10,
        show_grid=True,
        xlim=(0.1, 1.2),
        legend_label="KSS",
        legend_ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    )

    generate_all_sort_orders(
        rt_folder=DEFAULT_RT_FOLDER,
        kss_file=DEFAULT_KSS_FILE,
        output_dir=DEFAULT_OUTPUT_DIR,
        layout=layout,
        style=style,
    )


if __name__ == "__main__":
    main()