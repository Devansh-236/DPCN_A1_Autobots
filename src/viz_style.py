"""Shared matplotlib styling: one visual system for every figure in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import SERIES, SURFACE, INK, INK_2, INK_MUTED, GRID, DIVERGING, SEQ_BLUE

DIVERGING_CMAP = LinearSegmentedColormap.from_list("bl_gy_rd", list(DIVERGING))
SEQ_CMAP = LinearSegmentedColormap.from_list("blues", SEQ_BLUE)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.labelsize": 8.5,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 22,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 1.0,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 7.5, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    })


def strip(ax, x=True, y=True, grid_axis="y"):
    """Recessive frame: no box, grid on one axis only."""
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    return ax


def subtitle(ax, text):
    """Deck line, placed under the title. Call AFTER ax.set_title()."""
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=7.6,
            color=INK_MUTED, va="bottom", ha="left")
