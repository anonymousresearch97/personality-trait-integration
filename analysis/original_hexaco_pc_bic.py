from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyreadstat
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2

from hexaco_bic_exhaustive import exact_bic_dag, skeleton


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Lee & Ashton 2018  online sample data.sav"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DOMAINS = ["H", "E", "X", "A", "C", "O"]
AGE_LABELS = [
    "Childhood 3-12",
    "Adolescence 13-19",
    "Early adulthood 20-39",
    "Middle adulthood 40-64",
    "Older adulthood 65-100",
]
AGE_BINS = [2, 12, 19, 39, 64, 100]


def load_original() -> pd.DataFrame:
    cols = ["user_age", "hones16", "emoti16", "extra16", "agree16", "consc16", "openn16"]
    raw, _ = pyreadstat.read_sav(DATA_FILE, usecols=cols, apply_value_formats=False)
    df = raw.rename(
        columns={
            "user_age": "age",
            "hones16": "H",
            "emoti16": "E",
            "extra16": "X",
            "agree16": "A",
            "consc16": "C",
            "openn16": "O",
        }
    )
    df = df.dropna(subset=["age", *DOMAINS]).copy()
    df = df[(df["age"] >= 3) & (df["age"] <= 100)].copy()
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def discretize_equal_width(df: pd.DataFrame) -> pd.DataFrame:
    bins = [1.0, 1.8, 2.6, 3.4, 4.2, 5.0000001]
    out = pd.DataFrame(index=df.index)
    for col in DOMAINS:
        out[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True) + 1
    return out.astype(int)


def conditional_mi_pvalue_full_df(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning: tuple[str, ...],
    levels: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> float:
    d = data[[x, y, *conditioning]].dropna().astype(int)
    g_stat = 0.0
    if conditioning:
        groups = d.groupby(list(conditioning), observed=True, sort=False)
    else:
        groups = [((), d)]

    for _, group in groups:
        table = (
            pd.crosstab(group[x], group[y])
            .reindex(index=levels, columns=levels, fill_value=0)
            .to_numpy(dtype=float)
        )
        n = table.sum()
        if n <= 0:
            continue
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
        valid = (table > 0) & (expected > 0)
        g_stat += 2 * np.sum(table[valid] * np.log(table[valid] / expected[valid]))

    dfree = (len(levels) - 1) * (len(levels) - 1) * (len(levels) ** len(conditioning))
    return float(chi2.sf(g_stat, dfree))


def pc_stable_edges(data: pd.DataFrame, alpha: float = 0.05) -> tuple[tuple[str, str], ...]:
    nodes = list(data.columns)
    adjacency = {node: set(nodes) - {node} for node in nodes}
    level = 0
    while True:
        adjacency_at_level = {node: set(neighbors) for node, neighbors in adjacency.items()}
        edges = [
            tuple(sorted((a, b)))
            for a in nodes
            for b in adjacency_at_level[a]
            if a < b
        ]
        for x, y in edges:
            if y not in adjacency[x]:
                continue
            candidates: list[tuple[str, ...]] = []
            for base in (adjacency_at_level[x] - {y}, adjacency_at_level[y] - {x}):
                if len(base) >= level:
                    candidates.extend(itertools.combinations(sorted(base), level))
            seen: set[tuple[str, ...]] = set()
            for conditioning in candidates:
                if conditioning in seen:
                    continue
                seen.add(conditioning)
                if conditional_mi_pvalue_full_df(data, x, y, conditioning) > alpha:
                    adjacency[x].discard(y)
                    adjacency[y].discard(x)
                    break
        level += 1
        if not any(len(adjacency[node]) >= level + 1 for node in nodes):
            break
        if level > len(nodes) - 2:
            break
    return tuple(sorted(tuple(sorted((a, b))) for a in nodes for b in adjacency[a] if a < b))


def run_pc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for age_group in AGE_LABELS:
        group = df[df["age_group"] == age_group]
        discrete = discretize_equal_width(group[DOMAINS])
        edges = pc_stable_edges(discrete, alpha=0.05)
        rows.append(
            {
                "dataset": "Original HEXACO-100",
                "age_group": age_group,
                "n": len(group),
                "alpha": 0.05,
                "discretization": "equal_width_1_to_5",
                "edge_count": len(edges),
                "edges": ", ".join(f"{a}-{b}" for a, b in edges),
            }
        )
    return pd.DataFrame(rows)


def run_exact_bic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("All ages 3-100", df)]
    groups.extend((age_group, df[df["age_group"] == age_group]) for age_group in AGE_LABELS)
    for label, group in groups:
        discrete = discretize_equal_width(group[DOMAINS])
        bic, dag = exact_bic_dag(discrete)
        skel = skeleton(dag)
        rows.append(
            {
                "dataset": "Original HEXACO-100",
                "group": label,
                "n": len(group),
                "bic": bic,
                "dag_edge_count": len(dag),
                "skeleton_edge_count": len(skel),
                "dag_edges": ", ".join(f"{a}->{b}" for a, b in dag),
                "skeleton_edges": ", ".join(f"{a}-{b}" for a, b in skel),
            }
        )
    return pd.DataFrame(rows)


def positions() -> dict[str, tuple[float, float]]:
    ordered = ["H", "A", "C", "O", "X", "E"]
    angles = np.linspace(0, 2 * np.pi, len(ordered), endpoint=False)
    return {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(ordered, angles)}


def draw_network_grid(pdf: PdfPages, table: pd.DataFrame, edge_col: str, title: str, directed: bool) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.2))
    pos = positions()
    for ax, age_group in zip(axes, AGE_LABELS):
        row = table[(table.iloc[:, 1] == age_group)]
        if row.empty:
            row = table[(table["group"] == age_group)]
        row = row.iloc[0]
        n = int(row["n"])
        ax.set_title(f"{age_group}\nn={n}", fontsize=9)
        ax.axis("off")
        graph = nx.DiGraph() if directed else nx.Graph()
        graph.add_nodes_from(DOMAINS)
        edge_text = row[edge_col] if isinstance(row[edge_col], str) else ""
        if edge_text:
            for edge in edge_text.split(", "):
                if directed:
                    parent, child = edge.split("->")
                    graph.add_edge(parent, child)
                else:
                    a, b = edge.split("-")
                    graph.add_edge(a, b)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#f8fafc", edgecolors="#334155", node_size=850)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10)
        if graph.edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                arrows=directed,
                arrowsize=13,
                arrowstyle="-|>",
                width=2.0,
                edge_color="#2563eb" if not directed else "#7c3aed",
                connectionstyle="arc3,rad=0.05",
            )
        else:
            ax.text(0, -1.35, "No retained edges", ha="center", fontsize=8, color="#64748b")
    fig.suptitle(title, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(df: pd.DataFrame, pc: pd.DataFrame, bic: pd.DataFrame) -> Path:
    path = OUT_DIR / "original_hexaco_pc_bic_results.pdf"
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        fig.text(0.06, 0.93, "Original HEXACO-100 PC-Stable and Exact BIC Rerun", fontsize=17, weight="bold")
        summary = (
            f"Analytic N = {len(df):,}, after dropping missing ages and retaining ages 3-100. "
            "Domain scores used the six supplied HEXACO variables: H, E, X, A, C, and O. "
            "All models use five equal-width 1-5 bins."
        )
        fig.text(0.06, 0.84, summary, fontsize=10.5, va="top")
        fig.text(0.06, 0.75, "PC-stable edge counts: " + "; ".join(f"{r.age_group}: {int(r.edge_count)}" for r in pc.itertuples()), fontsize=10.5, va="top")
        global_row = bic[bic["group"] == "All ages 3-100"].iloc[0]
        fig.text(
            0.06,
            0.67,
            f"Exact BIC global all-ages skeleton: {int(global_row.skeleton_edge_count)} edges: {global_row.skeleton_edges}.",
            fontsize=10.5,
            va="top",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        draw_network_grid(pdf, pc, "edges", "PC-stable skeletons by age group", directed=False)
        draw_network_grid(pdf, bic[bic["group"].isin(AGE_LABELS)], "dag_edges", "Exact BIC-optimal DAGs by age group", directed=True)
    return path


def main() -> None:
    df = load_original()
    pc = run_pc(df)
    bic = run_exact_bic(df)
    scored_path = OUT_DIR / "original_hexaco_scored_domains.csv"
    pc_path = OUT_DIR / "original_pc_stable_edge_counts.csv"
    bic_path = OUT_DIR / "original_bic_exact_edge_counts.csv"
    df.to_csv(scored_path, index=False)
    pc.to_csv(pc_path, index=False)
    bic.to_csv(bic_path, index=False)
    pdf_path = build_pdf(df, pc, bic)
    print(f"Wrote {scored_path}")
    print(f"Wrote {pc_path}")
    print(f"Wrote {bic_path}")
    print(f"Wrote {pdf_path}")
    print("\nPC-stable")
    print(pc.to_string(index=False))
    print("\nExact BIC")
    print(bic.to_string(index=False))


if __name__ == "__main__":
    main()
