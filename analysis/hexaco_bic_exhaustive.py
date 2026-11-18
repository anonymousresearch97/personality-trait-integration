from __future__ import annotations

import itertools
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SCORED_DATA = OUT_DIR / "scored_domain_data.csv"

DOMAINS = ["H", "E", "X", "A", "C", "O"]
DOMAIN_NAMES = {
    "H": "Honesty-Humility",
    "E": "Emotionality",
    "X": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "O": "Openness",
}
AGE_LABELS = [
    "Childhood 3-12",
    "Adolescence 13-19",
    "Early adulthood 20-39",
    "Middle adulthood 40-64",
    "Older adulthood 65-100",
]
MIN_GROUP_N = 30


@dataclass(frozen=True)
class BicResult:
    dataset: str
    age_group: str
    n: int
    bic: float | None
    dag_edges: tuple[tuple[str, str], ...]
    skeleton_edges: tuple[tuple[str, str], ...]
    status: str


def discretize_equal_width(df: pd.DataFrame) -> pd.DataFrame:
    bins = [1.0, 1.8, 2.6, 3.4, 4.2, 5.0000001]
    out = pd.DataFrame(index=df.index)
    for col in DOMAINS:
        out[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True) + 1
    return out.astype(int)


def local_discrete_bic(data: np.ndarray, child: int, parents: tuple[int, ...]) -> float:
    """Discrete BN BIC score for one node with 5-level variables.

    This matches the manuscript's discretized-variable setting: log likelihood is
    computed from empirical conditional tables, and the BIC penalty uses the full
    5 ** number_of_parents parent-state space.
    """
    n = data.shape[0]
    child_values = data[:, child] - 1
    if parents:
        parent_values = data[:, parents] - 1
        parent_index = np.zeros(n, dtype=int)
        multiplier = 1
        for j in range(len(parents) - 1, -1, -1):
            parent_index += parent_values[:, j] * multiplier
            multiplier *= 5
        q = 5 ** len(parents)
    else:
        parent_index = np.zeros(n, dtype=int)
        q = 1

    counts = np.zeros((q, 5), dtype=float)
    np.add.at(counts, (parent_index, child_values), 1.0)
    row_totals = counts.sum(axis=1, keepdims=True)
    nonzero = counts > 0
    probs = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
    loglik = float(np.sum(counts[nonzero] * np.log(probs[nonzero])))
    n_parameters = q * (5 - 1)
    return loglik - 0.5 * n_parameters * math.log(n)


def exact_bic_dag(discrete: pd.DataFrame) -> tuple[float, tuple[tuple[str, str], ...]]:
    nodes = list(discrete.columns)
    p = len(nodes)
    arr = discrete[nodes].to_numpy(dtype=int)

    local_scores: list[dict[int, float]] = []
    best_parent_score: list[dict[int, tuple[float, int]]] = []
    for child in range(p):
        child_scores: dict[int, float] = {}
        allowed_bits = [i for i in range(p) if i != child]
        for r in range(len(allowed_bits) + 1):
            for parents in itertools.combinations(allowed_bits, r):
                mask = sum(1 << parent for parent in parents)
                child_scores[mask] = local_discrete_bic(arr, child, parents)
        local_scores.append(child_scores)

        best_for_allowed: dict[int, tuple[float, int]] = {}
        for allowed in range(1 << p):
            if allowed & (1 << child):
                continue
            best_score = -np.inf
            best_mask = 0
            parent_mask = allowed
            while True:
                score = child_scores[parent_mask]
                if score > best_score:
                    best_score = score
                    best_mask = parent_mask
                if parent_mask == 0:
                    break
                parent_mask = (parent_mask - 1) & allowed
            best_for_allowed[allowed] = (best_score, best_mask)
        best_parent_score.append(best_for_allowed)

    full = (1 << p) - 1
    dp = {0: 0.0}
    choice: dict[int, tuple[int, int]] = {}
    for size in range(1, p + 1):
        for subset_nodes in itertools.combinations(range(p), size):
            subset = sum(1 << node for node in subset_nodes)
            best = -np.inf
            best_child = -1
            best_parents = 0
            for sink in subset_nodes:
                remaining = subset & ~(1 << sink)
                family_score, parent_mask = best_parent_score[sink][remaining]
                score = dp[remaining] + family_score
                if score > best:
                    best = score
                    best_child = sink
                    best_parents = parent_mask
            dp[subset] = best
            choice[subset] = (best_child, best_parents)

    edges: list[tuple[str, str]] = []
    subset = full
    while subset:
        child, parents = choice[subset]
        for parent in range(p):
            if parents & (1 << parent):
                edges.append((nodes[parent], nodes[child]))
        subset &= ~(1 << child)
    return dp[full], tuple(sorted(edges))


def skeleton(edges: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(tuple(sorted(edge)) for edge in edges))


def load_scores() -> pd.DataFrame:
    if not SCORED_DATA.exists():
        raise FileNotFoundError(
            f"Missing {SCORED_DATA}. Run analysis/hexaco_pc_replication.py first."
        )
    scores = pd.read_csv(SCORED_DATA)
    scores[DOMAINS] = scores[DOMAINS].apply(pd.to_numeric, errors="coerce")
    return scores


def run_bic(scores: pd.DataFrame, datasets: list[str]) -> list[BicResult]:
    results: list[BicResult] = []
    for dataset in datasets:
        base = scores[(scores["dataset"] == dataset) & ~scores["excluded"]].copy()
        for age_group in AGE_LABELS:
            group = base[base["age_group"] == age_group].dropna(subset=DOMAINS)
            n = len(group)
            if n < MIN_GROUP_N:
                results.append(
                    BicResult(dataset, age_group, n, None, tuple(), tuple(), f"skipped n<{MIN_GROUP_N}")
                )
                continue
            discrete = discretize_equal_width(group[DOMAINS])
            bic, dag = exact_bic_dag(discrete)
            results.append(
                BicResult(dataset, age_group, n, bic, dag, skeleton(dag), "estimated")
            )
    return results


def results_to_frames(results: list[BicResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = []
    edges = []
    for result in results:
        counts.append(
            {
                "dataset": result.dataset,
                "age_group": result.age_group,
                "n": result.n,
                "status": result.status,
                "bic": result.bic,
                "dag_edge_count": len(result.dag_edges) if result.bic is not None else np.nan,
                "skeleton_edge_count": len(result.skeleton_edges) if result.bic is not None else np.nan,
                "dag_edges": ", ".join(f"{a}->{b}" for a, b in result.dag_edges),
                "skeleton_edges": ", ".join(f"{a}-{b}" for a, b in result.skeleton_edges),
            }
        )
        for parent, child in result.dag_edges:
            edges.append(
                {
                    "dataset": result.dataset,
                    "age_group": result.age_group,
                    "n": result.n,
                    "bic": result.bic,
                    "edge": f"{parent}->{child}",
                    "skeleton_edge": "-".join(sorted((parent, child))),
                    "parent_domain": DOMAIN_NAMES[parent],
                    "child_domain": DOMAIN_NAMES[child],
                }
            )
    return pd.DataFrame(counts), pd.DataFrame(edges)


def network_positions() -> dict[str, tuple[float, float]]:
    ordered = ["H", "A", "C", "O", "X", "E"]
    angles = np.linspace(0, 2 * np.pi, len(ordered), endpoint=False)
    return {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(ordered, angles)}


def draw_network_grid(pdf: PdfPages, counts: pd.DataFrame, dataset: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.3))
    pos = network_positions()
    for ax, age_group in zip(axes, AGE_LABELS):
        row = counts[(counts["dataset"] == dataset) & (counts["age_group"] == age_group)].iloc[0]
        ax.set_title(f"{age_group}\nn={int(row['n'])}", fontsize=9)
        ax.axis("off")
        graph = nx.DiGraph()
        graph.add_nodes_from(DOMAINS)
        if row["status"] == "estimated" and isinstance(row["dag_edges"], str) and row["dag_edges"]:
            for edge in row["dag_edges"].split(", "):
                parent, child = edge.split("->")
                graph.add_edge(parent, child)
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_size=850, node_color="#f6f8fa", edgecolors="#334155"
        )
        nx.draw_networkx_labels(graph, pos, ax=ax, labels={d: d for d in DOMAINS}, font_size=10)
        if graph.edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                arrows=True,
                arrowsize=14,
                arrowstyle="-|>",
                width=2.0,
                edge_color="#7c3aed",
                connectionstyle="arc3,rad=0.05",
            )
        elif row["status"] != "estimated":
            ax.text(0, -1.35, "Skipped", ha="center", fontsize=8, color="#64748b")
        else:
            ax.text(0, -1.35, "Empty optimal DAG", ha="center", fontsize=8, color="#64748b")
    fig.suptitle(f"{dataset}: exact BIC-optimal DAGs by age group", x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_edge_count_plot(pdf: PdfPages, counts: pd.DataFrame, datasets: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(AGE_LABELS))
    for dataset in datasets:
        sub = counts[counts["dataset"] == dataset].set_index("age_group").reindex(AGE_LABELS)
        y = sub["skeleton_edge_count"].astype(float).to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=dataset)
        for xi, yi, n in zip(x, y, sub["n"].fillna(0).astype(int)):
            if np.isfinite(yi):
                ax.text(xi, yi + 0.15, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS, rotation=20, ha="right")
    ax.set_ylim(-0.2, 10.2)
    ax.set_ylabel("BIC-optimal skeleton edge count")
    ax.set_title("Exact exhaustive BIC search over manuscript age groups", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_table(pdf: PdfPages, title: str, df: pd.DataFrame, columns: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=14)
    shown = df[columns].copy()
    if "bic" in shown:
        shown["bic"] = shown["bic"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    for col in shown.columns:
        shown[col] = shown[col].fillna("").astype(str)
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.28)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#eee7ff")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#d7d3e8")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(counts: pd.DataFrame, datasets: list[str]) -> Path:
    pdf_path = OUT_DIR / "hexaco_exhaustive_bic_results.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        fig.text(0.06, 0.93, "Exact Exhaustive BIC Search", fontsize=18, weight="bold", ha="left")
        text = (
            "I ran exact dynamic-programming exhaustive search for the BIC-optimal discrete Bayesian "
            "network over the six HEXACO domains. Scores used the same five equal-width 1-5 bins as "
            "the manuscript PC-stable analysis. Arrows identify one BIC-optimal DAG; interpretation "
            "should focus mainly on the skeleton and edge counts because cross-sectional orientation "
            "is not causal."
        )
        fig.text(0.06, 0.84, "\n".join(textwrap.wrap(text, 96)), fontsize=10.5, va="top")
        text2 = (
            "The original Lee & Ashton HEXACO-100 respondent-level file was not available locally, and "
            "public sources indicate that raw access requires contacting the authors. I therefore ran "
            "the exhaustive BIC search on the available external datasets and included a missing-source "
            "note for the original dataset."
        )
        fig.text(0.06, 0.68, "\n".join(textwrap.wrap(text2, 96)), fontsize=10.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        draw_table(
            pdf,
            "BIC-optimal DAG/skeleton table",
            counts,
            ["dataset", "age_group", "n", "status", "bic", "skeleton_edge_count", "skeleton_edges"],
        )
        draw_edge_count_plot(pdf, counts[counts["dataset"].isin(datasets)], datasets)
        for dataset in datasets:
            draw_network_grid(pdf, counts, dataset)
    return pdf_path


def main() -> None:
    scores = load_scores()
    datasets = [
        "MTurk HEXACO-208",
        "MTurk HEXACO-96",
        "MTurk pooled 96+208",
        "Conflict-management HEXACO-60",
    ]
    results = run_bic(scores, datasets)
    counts, edges = results_to_frames(results)

    original_note = pd.DataFrame(
        [
            {
                "dataset": "Original manuscript HEXACO-100",
                "age_group": "all manuscript groups",
                "n": 0,
                "status": "not run: respondent-level source file not available",
                "bic": np.nan,
                "dag_edge_count": np.nan,
                "skeleton_edge_count": np.nan,
                "dag_edges": "",
                "skeleton_edges": "",
            }
        ]
    )
    counts_with_note = pd.concat([counts, original_note], ignore_index=True)
    counts_with_note.to_csv(OUT_DIR / "bic_exhaustive_edge_counts.csv", index=False)
    edges.to_csv(OUT_DIR / "bic_exhaustive_edges.csv", index=False)
    pdf = build_pdf(counts_with_note, datasets[:3])
    print(f"Wrote {pdf}")
    print(f"Wrote {OUT_DIR / 'bic_exhaustive_edge_counts.csv'}")
    print(f"Wrote {OUT_DIR / 'bic_exhaustive_edges.csv'}")


if __name__ == "__main__":
    main()
