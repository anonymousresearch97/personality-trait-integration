"""Draw minimal manuscript skeleton figures from the validated saved networks.

Only skeleton_edges are plotted. This script does not estimate networks,
sample participants, or use any inferred orientation information.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

from plot_revision_cpdags import (
    COLORS, METHODS, NODES, POSITIONS, PRIMARY_GROUPS, SAPA_GROUPS, edges,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/main_paper_cpdags/network_cpdag_results.csv"
OUT = ROOT / "outputs/publication_network_figures/main"


def draw_skeleton(ax, row, color):
    pairs = edges(row["skeleton_edges"])
    assert len(pairs) == int(row["edge_count"])
    assert len({frozenset(pair) for pair in pairs}) == len(pairs)
    for first, second in pairs:
        x1, y1 = POSITIONS[first]
        x2, y2 = POSITIONS[second]
        ax.plot((x1, x2), (y1, y2), color=color, linewidth=1.6,
                linestyle="-", solid_capstyle="round", zorder=1)
    for node in NODES:
        x, y = POSITIONS[node]
        ax.add_patch(Circle((x, y), .15, facecolor="white", edgecolor=color,
                            linewidth=1.7, zorder=2))
        ax.text(x, y, node, fontsize=12.5, fontweight="bold", color="#172b3a",
                ha="center", va="center", zorder=3)
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
    ax.axis("off")


def draw_figure(frame, dataset, groups, labels, stem):
    ncols = len(groups)
    assert ncols in (2, 4)
    selected = frame.loc[frame.dataset.eq(dataset) & frame.group.isin(groups)]
    assert len(selected) == 2 * ncols
    fig = plt.figure(figsize=(12 if ncols == 4 else 7, 6.8))
    margin, width = (.022, .235) if ncols == 4 else (.025, .45)
    step = .245 if ncols == 4 else .49
    for method, row_label, bottom, heading_y in zip(
            METHODS, ("Gaussian PC", "Gaussian BIC"), (.530, .045), (.980, .490)):
        color = COLORS[method]
        fig.text(.025, heading_y, row_label, fontsize=14, fontweight="bold",
                 color=color, ha="left", va="top")
        for column, (group, label) in enumerate(zip(groups, labels)):
            fit = selected.loc[selected.group.eq(group) & selected.method.eq(method)]
            assert len(fit) == 1
            ax = fig.add_axes([margin + step * column, bottom, width, .37])
            draw_skeleton(ax, fit.iloc[0], color)
            if method == METHODS[0]:
                ax.set_title(label, fontsize=12.2, color="#172b3a", pad=11)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    frame = pd.read_csv(SOURCE)
    assert len(frame) == 20
    draw_figure(frame, "Primary", PRIMARY_GROUPS,
                ("Ages 13–19", "Ages 20–39", "Ages 40–64", "Ages 65–100"),
                "primary_gaussian_skeletons")
    draw_figure(frame, "SAPA", SAPA_GROUPS,
                ("18 and under", "19–39", "40–59", "60 and over"),
                "sapa_gaussian_skeletons")
    draw_figure(frame, "Thielmann community", ("Wave 1 (2008)", "Wave 2 (2018)"),
                ("2008", "2018"), "thielmann_community_gaussian_skeletons")
    print("Wrote 20 skeleton panels in three PNG/SVG figure pairs; no analyses rerun.")


if __name__ == "__main__":
    main()
