"""Shared figure style and deterministic saving (figures/<name>.pdf and .png).

Series colours come in a fixed order (never cycled); at most three series share one panel (more series go to
separate panels). Every figure with two or more series carries a legend; error bars are drawn for every measured
point; one y-axis per panel.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import FIGURES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (validated all-pairs for colour-vision deficiency)
NEUTRAL = "#52514e"  # free-theory references, guide lines
MARKERS = ["o", "s", "^"]

STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "errorbar.capsize": 2,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "svg.hashsalt": "masspairing",
}


def figure(ncols=1, nrows=1, width=3.4, height=2.6):
    """A figure with the shared style; width/height per panel in inches."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    return fig, ax


def save(fig, name):
    """Write figures/<name>.pdf and figures/<name>.png without timestamps (byte-reproducible)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=200, metadata={"Software": None})
    plt.close(fig)
    return FIGURES / f"{name}.pdf"
