"""Draw audited equivalence-class networks without inventing orientations.

Read network_cpdag_results.csv produced by revision_cpdag_reconstruction.py.
No model fitting or resampling takes place here. An incompatible PC result is
shown only as an explicitly labelled skeleton, never as a CPDAG.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "revision_cpdag"
SOURCE = OUT / "network_cpdag_results.csv"
NODES = ("H", "E", "X", "A", "C", "O")
POSITIONS = {node: (math.cos(math.pi / 2 - i * math.pi / 3),
                    math.sin(math.pi / 2 - i * math.pi / 3))
             for i, node in enumerate(NODES)}
COLORS = {"Gaussian PC-stable": "#245c82", "Gaussian BIC": "#8b4d28"}
METHODS = tuple(COLORS)
PRIMARY_GROUPS = ["Adolescence 13-19", "Early adulthood 20-39",
                  "Middle adulthood 40-64", "Older adulthood 65-100"]
PRIMARY_TITLES = ["Adolescence: 13–19", "Early adulthood: 20–39",
                  "Middle adulthood: 40–64", "Older adulthood: 65–100"]
SAPA_GROUPS = ["18 and under", "19-39", "40-59", "60 and over"]
SAPA_TITLES = ["18 and under", "19–39", "40–59", "60 and over"]


def valid(row):
    return str(row["cpdag_valid"]).lower() in ("true", "1")


def edges(value, arrow=False):
    if pd.isna(value) or not str(value).strip():
        return []
    delimiter = "->" if arrow else "-"
    result = [tuple(edge.strip().split(delimiter)) for edge in str(value).split(",")]
    assert all(len(edge) == 2 and len(set(edge)) == 2 and set(edge) <= set(NODES)
               for edge in result)
    return result


def validate(row):
    skeleton = {frozenset(edge) for edge in edges(row["skeleton_edges"])}
    assert len(skeleton) == int(row["edge_count"])
    if valid(row):
        directed = edges(row["directed_edges"], arrow=True)
        reversible = edges(row["undirected_edges"])
        assert len(directed) == int(row["directed_count"])
        assert len(reversible) == int(row["undirected_count"])
        assert len(directed) + len(reversible) == len(skeleton)
        assert {frozenset(edge) for edge in directed + reversible} == skeleton
        assert int(row["equivalence_class_size"]) >= 1
    else:
        assert not edges(row["directed_edges"], arrow=True)


def draw_panel(ax, row, color, title=None, sample=True, fontsize=10):
    """Draw arrows only for edges verified compelled across every DAG."""
    validate(row)
    ok = valid(row)
    edge_color = color if ok else "#929ca4"
    reversible = edges(row["undirected_edges"]) if ok else edges(row["skeleton_edges"])
    for first, second in reversible:
        x1, y1 = POSITIONS[first]
        x2, y2 = POSITIONS[second]
        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=1.5,
                alpha=.63, solid_capstyle="round", linestyle="-" if ok else "--",
                zorder=1)
    if ok:
        for first, second in edges(row["directed_edges"], arrow=True):
            start, end = POSITIONS[first], POSITIONS[second]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            trim = .165
            p1 = (start[0] + trim * dx / length, start[1] + trim * dy / length)
            p2 = (end[0] - trim * dx / length, end[1] - trim * dy / length)
            ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
                                        mutation_scale=15, linewidth=1.65,
                                        color=color, shrinkA=0, shrinkB=0,
                                        zorder=1.5))
    for node, (x, y) in POSITIONS.items():
        ax.add_patch(Circle((x, y), .15, facecolor="white", edgecolor=edge_color,
                            linewidth=1.7, zorder=2))
        ax.text(x, y, node, ha="center", va="center", fontsize=12,
                fontweight="bold", color="#172b3a", zorder=3)
    ax.set(xlim=(-1.27, 1.27), ylim=(-1.27, 1.27), aspect="equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11.5, color="#152e40", pad=28)
    if sample:
        ax.text(.5, 1.03, f"N = {int(row['n']):,}   ·   {int(row['edge_count'])} links",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=fontsize,
                color="#505c66")
    if ok:
        detail = f"{int(row['directed_count'])} compelled  ·  {int(row['undirected_count'])} reversible"
        detail_color = color
    else:
        detail = "No consistent CPDAG\nDashed skeleton shown only"
        detail_color = "#9b3b35"
    ax.text(.5, -.045, detail, transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize, color=detail_color, fontweight="bold", linespacing=1.35)


def save(fig, stem):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def draw_cross_sectional(frame, dataset, groups, titles, stem, note):
    selected = frame.loc[frame.dataset.eq(dataset) & frame.group.isin(groups)]
    assert len(selected) == 8
    fig = plt.figure(figsize=(15, 9.5))
    fig.text(.035, .966, f"{dataset}: Gaussian CPDAG reconstruction", fontsize=20,
             fontweight="bold", color="#142e40", va="top")
    fig.text(.035, .918,
             "Continuous domain scores · full samples · identical node positions in every panel",
             fontsize=11, color="#53616c", va="top")
    for method, bottom, heading_y in zip(METHODS, (.565, .215), (.873, .523)):
        heading = ("A   Gaussian PC-stable (α = .05)" if method == METHODS[0]
                   else "B   Exact Gaussian BIC")
        color = COLORS[method]
        fig.text(.035, heading_y, heading, fontsize=12.5, fontweight="bold", color=color)
        for column, (group, title) in enumerate(zip(groups, titles)):
            fit = selected.loc[selected.group.eq(group) & selected.method.eq(method)]
            assert len(fit) == 1
            ax = fig.add_axes([.025 + .244 * column, bottom, .236, .24])
            draw_panel(ax, fit.iloc[0], color, title=title)
    fig.text(.035, .145,
             "Arrows: compelled directions within the fitted equivalence class. Plain lines: reversible directions.",
             fontsize=10.1, color="#414f5b")
    fig.text(.035, .121,
             "These orientations do not establish causal effects. Line widths do not encode association magnitude.",
             fontsize=10.1, color="#414f5b")
    fig.text(.035, .096, note, fontsize=9.8, color="#414f5b")
    fig.text(.035, .058,
             "H: Honesty–Humility    E: Emotionality    X: Extraversion    A: Agreeableness",
             fontsize=10.2, color="#142e40")
    fig.text(.035, .032,
             "C: Conscientiousness    O: Openness to Experience    ·    Maximum 15 connections per network",
             fontsize=10.2, color="#142e40")
    save(fig, stem)


def draw_community(frame):
    fits = frame.loc[frame.dataset.eq("Thielmann community")]
    assert len(fits) == 4
    changes = pd.read_csv(ROOT / "outputs/longitudinal_bn_followup/paired_change_summary.csv")
    changes = changes.loc[changes.dataset.eq("thielmann_community") &
                          changes.representation.eq("continuous")]
    assert len(changes) == 2
    fig = plt.figure(figsize=(12.7, 8.5))
    grid = fig.add_gridspec(2, 3, left=.045, right=.975, bottom=.29, top=.755,
                           width_ratios=[1, 1, 1.35], wspace=.29, hspace=.62)
    fig.text(.045, .956, "Personality networks across ten years", fontsize=20,
             fontweight="bold", color="#1F2933", va="top")
    fig.text(.045, .910,
             "Thielmann community panel · Same 429 participants at both assessments",
             fontsize=11.5, color="#4D5964", va="top")
    fig.text(.045, .850, "A  Gaussian CPDAG reconstruction", fontsize=12, fontweight="bold")
    for row_index, method in enumerate(METHODS):
        for col_index, wave in enumerate((1, 2)):
            year = {1: 2008, 2: 2018}[wave]
            group = f"Wave {wave} ({year})"
            selected = fits.loc[fits.group.eq(group) & fits.method.eq(method)]
            assert len(selected) == 1
            row = selected.iloc[0]
            ax = fig.add_subplot(grid[row_index, col_index])
            draw_panel(ax, row, COLORS[method], sample=False, fontsize=8.6)
            short_method = "PC" if row_index == 0 else "BIC"
            ax.text(.5, 1.01, f"{short_method}: {int(row.edge_count)} of 15 links",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=COLORS[method])
            if row_index == 0:
                mean_age = {1: 51.10, 2: 60.97}[wave]
                ax.text(.5, 1.20, f"{year} · Mean age {mean_age:.2f}",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=10.5, color="#1F2933")
    change_ax = fig.add_subplot(grid[:, 2])
    box = change_ax.get_position()
    change_ax.set_position([box.x0, box.y0 + .085, box.width, box.height - .15])
    fig.text(box.x0, .850, "B  Change in link count", fontsize=12, fontweight="bold")
    fig.text(box.x0, .812, "2018 minus 2008", fontsize=10.5, color="#4D5964")
    change_ax.axvline(0, color="#7A838B", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    for method, short_method, y, expected, low in (
            (METHODS[0], "PC", 1, 2, -2), (METHODS[1], "BIC", 0, 3, -1)):
        row = changes.loc[changes.method.eq(short_method)].iloc[0]
        observed = float(row.observed_change)
        lower, upper = float(row.bootstrap_p2_5), float(row.bootstrap_p97_5)
        assert observed == expected and lower == low and upper == 4
        assert int(row.replicates) == 1000
        change_ax.errorbar(observed, y, xerr=[[observed - lower], [upper - observed]],
                           fmt="o", color=COLORS[method], markersize=7, capsize=5,
                           elinewidth=2, capthick=1.5, zorder=3)
        change_ax.text(1.0, y + .27,
                       f"{observed:+.0f} links   [{lower:.0f}, {upper:.0f}]",
                       color=COLORS[method], ha="center", fontsize=11, fontweight="bold")
    change_ax.set(yticks=[0, 1], yticklabels=["BIC", "PC"], ylim=(-.45, 1.7),
                  xlim=(-2.65, 4.65), xticks=list(range(-2, 5)),
                  xlabel="Change in selected links")
    change_ax.spines[["left", "top", "right"]].set_visible(False)
    change_ax.spines["bottom"].set_color("#A7AFB6")
    change_ax.tick_params(axis="y", length=0, pad=6)
    change_ax.tick_params(axis="x", colors="#4D5964")
    fig.text(box.x0, .300,
             "Dots: observed changes\nBars: 2.5–97.5 percentile ranges\nfrom 1,000 paired bootstrap draws",
             fontsize=9.4, color="#4D5964", va="top", linespacing=1.5)
    fig.text(.045, .218,
             "Arrows: compelled directions within the fitted equivalence class. Plain lines: reversible directions.\n"
             "Orientations do not establish causal effects. Dashed panels show skeletons with inconsistent PC orientations.",
             fontsize=9.5, color="#394752", va="top", linespacing=1.55)
    fig.text(.045, .151,
             "H  Honesty–Humility      E  Emotionality      X  Extraversion\n"
             "A  Agreeableness           C  Conscientiousness      O  Openness",
             fontsize=9.5, color="#394752", va="top", linespacing=1.55)
    fig.text(.045, .080,
             "Gaussian PC (α = .05) and exact Gaussian BIC. Wave networks describe associations between people.\n"
             "Two waves cannot locate a lifespan peak. Bootstrap ranges describe resampling stability; both include zero.",
             fontsize=9.1, color="#56616B", va="top", linespacing=1.5)
    save(fig, "thielmann_community_gaussian_cpdags")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    frame = pd.read_csv(SOURCE)
    for _, row in frame.iterrows():
        validate(row)
    draw_cross_sectional(frame, "Primary", PRIMARY_GROUPS, PRIMARY_TITLES,
                         "primary_gaussian_cpdags",
                         "Unequal full-sample group sizes. Childhood is excluded from this main figure.")
    draw_cross_sectional(frame, "SAPA", SAPA_GROUPS, SAPA_TITLES,
                         "sapa_gaussian_cpdags",
                         "Actual pooled SAPA bands differ from primary age ranges. No separate childhood group is available.")
    draw_community(frame)
    print("Wrote three PNG/SVG figures from audited CPDAG reconstruction results.")


if __name__ == "__main__":
    main()
