from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


@dataclass(frozen=True)
class HeatmapLayout:
    # Exact heatmap cell size in inches
    cell_width_in: float = 1.2
    cell_height_in: float = 0.55

    # Figure margins in inches (do not affect heatmap cell size)
    margin_left_in: float = 1.1
    margin_right_in: float = 0.3
    margin_top_in: float = 0.8
    margin_bottom_in: float = 0.9

    # Vertical legend (a 9x1 mini-heatmap) sizing in inches
    show_legend: bool = True
    legend_cell_width_in: float = 0.32
    legend_cell_height_in: float = 0.32
    legend_pad_in: float = 0.35
    legend_title_pad_in: float = 0.18  # extra top space above legend cells for label

    dpi: int = 300


@dataclass(frozen=True)
class HeatmapStyle:
    title: str = "Self-reported tiredness (KSS)"
    xlabel: str = "Session"
    ylabel: str = "Subject"

    session_tick_labels: Optional[Tuple[str, ...]] = None
    subject_tick_labels: Optional[Tuple[str, ...]] = None
    tick_fontsize: int = 10

    show_grid: bool = True
    grid_color: str = "black"
    grid_linewidth: float = 0.6
    grid_alpha: float = 0.35

    show_values: bool = False
    value_fmt: str = "{:.0f}"
    value_fontsize: int = 9
    value_color: str = "black"

    legend_label: str = "KSS"
    legend_label_fontsize: int = 10
    legend_tick_fontsize: int = 9
    legend_ticks: Tuple[int, ...] = (1, 3, 5, 7, 9)
    legend_show_grid: bool = True
    legend_grid_color: str = "black"
    legend_grid_linewidth: float = 0.6
    legend_grid_alpha: float = 0.35


def read_kss_matrix(
    txt_path: Path,
    *,
    expected_cols: Optional[int] = 3,
    dtype: Callable = float,
) -> np.ndarray:
    if not txt_path.exists():
        raise FileNotFoundError(f"KSS file not found: {txt_path}")
    data = np.loadtxt(str(txt_path), dtype=dtype)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if expected_cols is not None and data.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns but got {data.shape[1]}")
    return data


def make_discrete_bwr_9() -> Tuple[ListedColormap, BoundaryNorm]:
    base = plt.get_cmap("bwr")
    colors = base(np.linspace(0.0, 1.0, 9))
    cmap = ListedColormap(colors, name="bwr_9")
    boundaries = np.arange(0.5, 10.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


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


def plot_kss_heatmap_exact_cells_with_vertical_legend(
    kss: np.ndarray,
    output_path: Path,
    layout: HeatmapLayout,
    style: HeatmapStyle,
) -> None:
    if kss.ndim != 2:
        raise ValueError(f"kss must be 2D, got shape {kss.shape}")

    n_subjects, n_sessions = kss.shape

    cmap, norm = make_discrete_bwr_9()

    # Heatmap axes size in inches, so cell sizes are exact
    ax_w_in = n_sessions * layout.cell_width_in
    ax_h_in = n_subjects * layout.cell_height_in

    # Legend axes size in inches (vertical: 9 rows x 1 col)
    legend_w_in = layout.legend_cell_width_in
    legend_h_in = 9.0 * layout.legend_cell_height_in
    legend_block_w_in = 0.0
    if layout.show_legend:
        legend_block_w_in = layout.legend_pad_in + legend_w_in

    fig_w_in = layout.margin_left_in + ax_w_in + legend_block_w_in + layout.margin_right_in
    fig_h_in = layout.margin_bottom_in + ax_h_in + layout.margin_top_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=layout.dpi)

    # Main axes
    ax = _add_axes_in_inches(
        fig,
        rect_in=(layout.margin_left_in, layout.margin_bottom_in, ax_w_in, ax_h_in),
    )

    x_edges = np.arange(0, n_sessions + 1, 1, dtype=float)
    y_edges = np.arange(0, n_subjects + 1, 1, dtype=float)

    ax.pcolormesh(
        x_edges,
        y_edges,
        kss,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    ax.invert_yaxis()

    ax.set_xticks(np.arange(n_sessions) + 0.5)
    ax.set_yticks(np.arange(n_subjects) + 0.5)

    if style.session_tick_labels is None:
        ax.set_xticklabels([str(i + 1) for i in range(n_sessions)], fontsize=style.tick_fontsize)
    else:
        if len(style.session_tick_labels) != n_sessions:
            raise ValueError(
                f"session_tick_labels must have length {n_sessions}, got {len(style.session_tick_labels)}"
            )
        ax.set_xticklabels(list(style.session_tick_labels), fontsize=style.tick_fontsize)

    if style.subject_tick_labels is None:
        ax.set_yticklabels([str(i + 1) for i in range(n_subjects)], fontsize=style.tick_fontsize)
    else:
        if len(style.subject_tick_labels) != n_subjects:
            raise ValueError(
                f"subject_tick_labels must have length {n_subjects}, got {len(style.subject_tick_labels)}"
            )
        ax.set_yticklabels(list(style.subject_tick_labels), fontsize=style.tick_fontsize)

    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    ax.set_title(style.title)

    if style.show_grid:
        ax.set_xticks(np.arange(0, n_sessions + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, n_subjects + 1, 1), minor=True)
        ax.grid(
            which="minor",
            color=style.grid_color,
            linewidth=style.grid_linewidth,
            alpha=style.grid_alpha,
        )
        ax.tick_params(which="minor", bottom=False, left=False)

    if style.show_values:
        for i in range(n_subjects):
            for j in range(n_sessions):
                v = kss[i, j]
                if np.isfinite(v):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        style.value_fmt.format(v),
                        ha="center",
                        va="center",
                        fontsize=style.value_fontsize,
                        color=style.value_color,
                    )

    # Vertical legend: 9x1 cells, top is KSS=9 (red), bottom is KSS=1 (blue)
    if layout.show_legend:
        leg_left_in = layout.margin_left_in + ax_w_in + layout.legend_pad_in
        # Center legend vertically to the heatmap
        leg_bottom_in = layout.margin_bottom_in + (ax_h_in - legend_h_in) / 2.0

        ax_leg = _add_axes_in_inches(
            fig,
            rect_in=(leg_left_in, leg_bottom_in, legend_w_in, legend_h_in),
        )

        legend_vals = np.arange(9, 0, -1, dtype=float).reshape(9, 1)  # 9 at top, 1 at bottom
        x_e = np.arange(0, 2, 1, dtype=float)
        y_e = np.arange(0, 10, 1, dtype=float)

        ax_leg.pcolormesh(
            x_e,
            y_e,
            legend_vals,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )

        # Put the first row at the top
        ax_leg.invert_yaxis()

        # No x ticks
        ax_leg.set_xticks([])

        # Y ticks at cell centers for selected KSS values
        tick_vals = list(style.legend_ticks)
        # legend row index for value v in [9..1] is (9 - v)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=layout.dpi)
    plt.close(fig)


def main() -> None:
    kss_txt = Path("/work/long_lab/Mojtaba/data/DROZY/data/original/KSS.txt")
    output_png = Path("/home/mojtaba.kananisarche/Zh/plots/kss_heatmap.png")

    layout = HeatmapLayout(
        # Heatmap cell size
        cell_width_in=1.2,
        cell_height_in=0.5,
        # Margins
        margin_left_in=1.1,
        margin_right_in=0.3,
        margin_top_in=0.85,
        margin_bottom_in=0.9,
        # Vertical legend cell size (editable)
        show_legend=True,
        legend_cell_width_in=0.4,
        legend_cell_height_in=0.4,
        legend_pad_in=0.35,
        dpi=300,
    )

    style = HeatmapStyle(
        title="Karolinska Sleepiness Scale (KSS) Self-Reports",
        xlabel="Session",
        ylabel="Subject",
        session_tick_labels=("1", "2", "3"),
        show_grid=True,
        show_values=False,
        legend_label="KSS",
        legend_ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    )

    kss = read_kss_matrix(kss_txt, expected_cols=3)
    plot_kss_heatmap_exact_cells_with_vertical_legend(kss, output_png, layout, style)


if __name__ == "__main__":
    main()