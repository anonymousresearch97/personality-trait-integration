from __future__ import annotations

import itertools
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2

from hexaco_bic_exhaustive import exact_bic_dag, skeleton


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_external" / "sapa"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "sapa_data.tab"
KEY_FILE = DATA_DIR / "superKey696.csv"

DOMAINS = ["H", "E", "X", "A", "C", "O"]
DOMAIN_KEY_COLUMNS = {
    "H": "HEXACO_H",
    "E": "HEXACO_E",
    "X": "HEXACO_X",
    "A": "HEXACO_A",
    "C": "HEXACO_C",
    "O": "HEXACO_O",
}
DOMAIN_NAMES = {
    "H": "Honesty-Humility",
    "E": "Emotionality",
    "X": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "O": "Openness",
}

AGE_VALUE_TO_LABEL = {
    "18andUnder": "18 and under",
    "19to24": "19-24",
    "25to29": "25-29",
    "30to34": "30-34",
    "35to39": "35-39",
    "40to49": "40-49",
    "50to59": "50-59",
    "60andOver": "60 and over",
}
AGE_LABELS = list(AGE_VALUE_TO_LABEL.values())
AGE_MIDPOINTS = {
    "18 and under": 16.0,
    "19-24": 21.5,
    "25-29": 27.0,
    "30-34": 32.0,
    "35-39": 37.0,
    "40-49": 44.5,
    "50-59": 54.5,
    "60 and over": 65.0,
}

MAIN_MIN_ITEMS_PER_DOMAIN = 3
SENSITIVITY_THRESHOLDS = [2, 3, 4]
MIN_GROUP_N = 30


def load_key() -> dict[str, pd.Series]:
    key = pd.read_csv(KEY_FILE)
    item_col = key.columns[0]
    item_signs: dict[str, pd.Series] = {}
    for domain, key_col in DOMAIN_KEY_COLUMNS.items():
        signs = pd.to_numeric(key[key_col], errors="coerce").fillna(0).astype(int)
        keyed = key.loc[signs != 0, [item_col, key_col]].copy()
        keyed[key_col] = pd.to_numeric(keyed[key_col], errors="coerce").astype(int)
        item_signs[domain] = pd.Series(keyed[key_col].values, index=keyed[item_col].values)
    return item_signs


def load_sapa_raw(item_signs: dict[str, pd.Series]) -> pd.DataFrame:
    item_columns = sorted({item for signs in item_signs.values() for item in signs.index})
    usecols = ["RID", "gender", "age", *item_columns]
    return pd.read_csv(DATA_FILE, sep="\t", usecols=usecols)


def score_domains(raw: pd.DataFrame, item_signs: dict[str, pd.Series]) -> pd.DataFrame:
    scored = raw[["RID", "gender", "age"]].copy()
    scored["age_band"] = scored["age"].map(AGE_VALUE_TO_LABEL)

    for domain, signs in item_signs.items():
        items = list(signs.index)
        responses = raw[items].apply(pd.to_numeric, errors="coerce")
        keyed = responses.copy()
        for item, sign in signs.items():
            keyed[item] = responses[item] if sign > 0 else 7 - responses[item]
        scored[f"{domain}_item_count"] = keyed.notna().sum(axis=1)
        raw_1_to_6 = keyed.mean(axis=1, skipna=True)
        scored[domain] = 1 + (raw_1_to_6 - 1) * 4 / 5

    return scored


def analytic_sample(scored: pd.DataFrame, min_items: int) -> pd.DataFrame:
    count_cols = [f"{domain}_item_count" for domain in DOMAINS]
    mask = scored["age_band"].notna() & scored[count_cols].ge(min_items).all(axis=1)
    return scored.loc[mask].dropna(subset=DOMAINS).copy()


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


def group_iterator(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(label, df[df["age_band"] == label]) for label in AGE_LABELS]


def run_pc(df: pd.DataFrame, min_items: int) -> pd.DataFrame:
    rows = []
    for age_band, group in group_iterator(df):
        n = len(group)
        if n < MIN_GROUP_N:
            rows.append(
                {
                    "dataset": "SAPA IPIP-HEXACO",
                    "age_band": age_band,
                    "n": n,
                    "status": f"skipped n<{MIN_GROUP_N}",
                    "min_items_per_domain": min_items,
                    "alpha": 0.05,
                    "discretization": "score_1_to_5_equal_width_5_bins",
                    "edge_count": np.nan,
                    "edges": "",
                }
            )
            continue
        edges = pc_stable_edges(discretize_equal_width(group[DOMAINS]), alpha=0.05)
        rows.append(
            {
                "dataset": "SAPA IPIP-HEXACO",
                "age_band": age_band,
                "n": n,
                "status": "estimated",
                "min_items_per_domain": min_items,
                "alpha": 0.05,
                "discretization": "score_1_to_5_equal_width_5_bins",
                "edge_count": len(edges),
                "edges": ", ".join(f"{a}-{b}" for a, b in edges),
            }
        )
    return pd.DataFrame(rows)


def run_exact_bic(df: pd.DataFrame, min_items: int) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("All SAPA age bands", df)]
    groups.extend(group_iterator(df))
    for label, group in groups:
        n = len(group)
        if n < MIN_GROUP_N:
            rows.append(
                {
                    "dataset": "SAPA IPIP-HEXACO",
                    "group": label,
                    "n": n,
                    "status": f"skipped n<{MIN_GROUP_N}",
                    "min_items_per_domain": min_items,
                    "bic": np.nan,
                    "dag_edge_count": np.nan,
                    "skeleton_edge_count": np.nan,
                    "dag_edges": "",
                    "skeleton_edges": "",
                }
            )
            continue
        bic, dag = exact_bic_dag(discretize_equal_width(group[DOMAINS]))
        skel = skeleton(dag)
        rows.append(
            {
                "dataset": "SAPA IPIP-HEXACO",
                "group": label,
                "n": n,
                "status": "estimated",
                "min_items_per_domain": min_items,
                "bic": bic,
                "dag_edge_count": len(dag),
                "skeleton_edge_count": len(skel),
                "dag_edges": ", ".join(f"{a}->{b}" for a, b in dag),
                "skeleton_edges": ", ".join(f"{a}-{b}" for a, b in skel),
            }
        )
    return pd.DataFrame(rows)


def threshold_sample_sizes(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in range(1, 9):
        sample = analytic_sample(scored, threshold)
        row = {"min_items_per_domain": threshold, "n": len(sample)}
        for age_band in AGE_LABELS:
            row[age_band] = int((sample["age_band"] == age_band).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run_threshold_sensitivity(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        sample = analytic_sample(scored, threshold)
        pc = run_pc(sample, threshold)
        bic = run_exact_bic(sample, threshold)
        for row in pc.itertuples(index=False):
            rows.append(
                {
                    "method": "PC-stable",
                    "min_items_per_domain": threshold,
                    "group": row.age_band,
                    "n": row.n,
                    "status": row.status,
                    "edge_count": row.edge_count,
                    "edges": row.edges,
                }
            )
        for row in bic.itertuples(index=False):
            if row.group == "All SAPA age bands":
                continue
            rows.append(
                {
                    "method": "Exact BIC skeleton",
                    "min_items_per_domain": threshold,
                    "group": row.group,
                    "n": row.n,
                    "status": row.status,
                    "edge_count": row.skeleton_edge_count,
                    "edges": row.skeleton_edges,
                }
            )
    return pd.DataFrame(rows)


def network_positions() -> dict[str, tuple[float, float]]:
    ordered = ["H", "A", "C", "O", "X", "E"]
    angles = np.linspace(0, 2 * np.pi, len(ordered), endpoint=False)
    return {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(ordered, angles)}


def parse_edges(edge_text: object, directed: bool) -> list[tuple[str, str]]:
    if not isinstance(edge_text, str) or not edge_text:
        return []
    edges = []
    for edge in edge_text.split(", "):
        if directed:
            parent, child = edge.split("->")
            edges.append((parent, child))
        else:
            a, b = edge.split("-")
            edges.append((a, b))
    return edges


def draw_network_grid(pdf: PdfPages, table: pd.DataFrame, edge_col: str, title: str, directed: bool) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    pos = network_positions()
    for ax, age_band in zip(axes.ravel(), AGE_LABELS):
        row = table[table.iloc[:, 1] == age_band]
        if row.empty:
            row = table[table["group"] == age_band]
        row = row.iloc[0]
        n = int(row["n"])
        ax.set_title(f"{age_band}\nn={n:,}", fontsize=9)
        ax.axis("off")
        graph = nx.DiGraph() if directed else nx.Graph()
        graph.add_nodes_from(DOMAINS)
        graph.add_edges_from(parse_edges(row[edge_col], directed=directed))
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#f8fafc", edgecolors="#334155", node_size=800)
        nx.draw_networkx_labels(graph, pos, ax=ax, labels={d: d for d in DOMAINS}, font_size=10)
        if graph.edges:
            edge_kwargs = {
                "G": graph,
                "pos": pos,
                "ax": ax,
                "width": 2.0,
                "edge_color": "#2563eb" if not directed else "#7c3aed",
            }
            if directed:
                edge_kwargs.update(
                    {
                        "arrows": True,
                        "arrowsize": 13,
                        "arrowstyle": "-|>",
                        "connectionstyle": "arc3,rad=0.05",
                    }
                )
            nx.draw_networkx_edges(**edge_kwargs)
        else:
            ax.text(0, -1.35, "No retained edges", ha="center", fontsize=8, color="#64748b")
    fig.suptitle(title, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_edge_count_plot(pdf: PdfPages, pc: pd.DataFrame, bic: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(AGE_LABELS))
    pc_sub = pc.set_index("age_band").reindex(AGE_LABELS)
    bic_sub = bic[bic["group"].isin(AGE_LABELS)].set_index("group").reindex(AGE_LABELS)
    ax.plot(x, pc_sub["edge_count"].astype(float), marker="o", linewidth=2.2, label="PC-stable skeleton")
    ax.plot(x, bic_sub["skeleton_edge_count"].astype(float), marker="o", linewidth=2.2, label="Exact BIC skeleton")
    for xi, n in zip(x, pc_sub["n"].fillna(0).astype(int)):
        ax.text(xi, -0.55, f"n={n:,}", ha="center", va="top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS, rotation=20, ha="right")
    ax.set_ylim(-1.0, 15.5)
    ax.set_ylabel("Skeleton edge count")
    ax.set_title("SAPA IPIP-HEXACO edge counts by recorded age band", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_age_domain_means(pdf: PdfPages, df: pd.DataFrame) -> None:
    means = df.groupby("age_band", observed=True)[DOMAINS].mean().reindex(AGE_LABELS)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(AGE_LABELS))
    for domain in DOMAINS:
        ax.plot(x, means[domain], marker="o", linewidth=1.8, label=f"{domain}: {DOMAIN_NAMES[domain]}")
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS, rotation=20, ha="right")
    ax.set_ylim(1.0, 5.0)
    ax.set_ylabel("Scored domain mean (1-5)")
    ax.set_title("SAPA scored HEXACO domain means by recorded age band", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8.5)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_sensitivity_plot(pdf: PdfPages, sensitivity: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    x = np.arange(len(AGE_LABELS))
    for ax, method in zip(axes, ["PC-stable", "Exact BIC skeleton"]):
        sub = sensitivity[sensitivity["method"] == method]
        for threshold in SENSITIVITY_THRESHOLDS:
            rows = sub[sub["min_items_per_domain"] == threshold].set_index("group").reindex(AGE_LABELS)
            ax.plot(x, rows["edge_count"].astype(float), marker="o", linewidth=1.8, label=f"min {threshold}")
        ax.set_title(method, loc="left", weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, title="Items/domain")
    axes[0].set_ylabel("Skeleton edge count")
    fig.suptitle("SAPA threshold sensitivity", x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_table(pdf: PdfPages, title: str, df: pd.DataFrame, columns: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=14)
    shown = df[columns].copy()
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
    table.set_fontsize(7.4)
    table.scale(1, 1.24)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8f0fe")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#cbd5e1")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(
    scored: pd.DataFrame,
    sample: pd.DataFrame,
    item_signs: dict[str, pd.Series],
    sample_sizes: pd.DataFrame,
    pc: pd.DataFrame,
    bic: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> Path:
    path = OUT_DIR / "sapa_hexaco_pc_bic_results.pdf"
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        fig.text(0.06, 0.93, "SAPA IPIP-HEXACO PC-Stable and Exact BIC Replication", fontsize=17, weight="bold")
        keyed_summary = "; ".join(
            f"{domain}: {int((signs > 0).sum())} direct, {int((signs < 0).sum())} reversed"
            for domain, signs in item_signs.items()
        )
        summary = (
            f"Raw SAPA rows = {len(scored):,}. Main analytic N = {len(sample):,} after requiring a recorded "
            f"age band and at least {MAIN_MIN_ITEMS_PER_DOMAIN} keyed items for every HEXACO domain. "
            "The SAPA public file top-codes age as categories, so this is an age-band replication rather "
            "than exact-year or manuscript-age-bin replication."
        )
        fig.text(0.06, 0.84, "\n".join(textwrap.wrap(summary, 98)), fontsize=10.5, va="top")
        scoring = (
            "Domain scores use the SAPA superKey696 IPIP-HEXACO keys. Each domain has 40 keyed 1-6 response "
            f"items; reverse-keyed items were transformed as 7 - response, then means were rescaled to 1-5. "
            f"Keying counts: {keyed_summary}."
        )
        fig.text(0.06, 0.72, "\n".join(textwrap.wrap(scoring, 98)), fontsize=10.5, va="top")
        modeling = (
            "PC-stable used alpha .05 on five equal-width bins over the rescaled 1-5 score range. Exact BIC "
            "used exhaustive dynamic-programming DAG search over the six discretized domains; arrows are one "
            "BIC-optimal orientation and should not be read causally."
        )
        fig.text(0.06, 0.57, "\n".join(textwrap.wrap(modeling, 98)), fontsize=10.5, va="top")
        pc_counts = "; ".join(f"{r.age_band}: {int(r.edge_count)}" for r in pc.itertuples())
        global_bic = bic[bic["group"] == "All SAPA age bands"].iloc[0]
        bic_text = (
            f"Main PC-stable edge counts: {pc_counts}. Exact BIC all-age skeleton has "
            f"{int(global_bic.skeleton_edge_count)} edges: {global_bic.skeleton_edges}."
        )
        fig.text(0.06, 0.42, "\n".join(textwrap.wrap(bic_text, 98)), fontsize=10.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        draw_table(
            pdf,
            "Analytic N by minimum keyed items per domain",
            sample_sizes,
            ["min_items_per_domain", "n", *AGE_LABELS],
        )
        draw_age_domain_means(pdf, sample)
        draw_edge_count_plot(pdf, pc, bic)
        draw_network_grid(pdf, pc, "edges", "SAPA PC-stable skeletons by recorded age band", directed=False)
        draw_network_grid(
            pdf,
            bic[bic["group"].isin(AGE_LABELS)],
            "dag_edges",
            "SAPA exact BIC-optimal DAGs by recorded age band",
            directed=True,
        )
        draw_table(
            pdf,
            "SAPA PC-stable age-band table",
            pc,
            ["age_band", "n", "min_items_per_domain", "edge_count", "edges"],
        )
        draw_table(
            pdf,
            "SAPA exact BIC age-band table",
            bic[bic["group"].isin(["All SAPA age bands", *AGE_LABELS])],
            ["group", "n", "min_items_per_domain", "skeleton_edge_count", "skeleton_edges"],
        )
        draw_sensitivity_plot(pdf, sensitivity)
    return path


def main() -> None:
    item_signs = load_key()
    raw = load_sapa_raw(item_signs)
    scored = score_domains(raw, item_signs)
    sample_sizes = threshold_sample_sizes(scored)
    sample = analytic_sample(scored, MAIN_MIN_ITEMS_PER_DOMAIN)
    pc = run_pc(sample, MAIN_MIN_ITEMS_PER_DOMAIN)
    bic = run_exact_bic(sample, MAIN_MIN_ITEMS_PER_DOMAIN)
    sensitivity = run_threshold_sensitivity(scored)

    scored.to_csv(OUT_DIR / "sapa_hexaco_scored_domains_all_rows.csv", index=False)
    sample.to_csv(OUT_DIR / "sapa_hexaco_scored_domains.csv", index=False)
    sample_sizes.to_csv(OUT_DIR / "sapa_min_item_sample_sizes.csv", index=False)
    pc.to_csv(OUT_DIR / "sapa_pc_stable_edge_counts.csv", index=False)
    bic.to_csv(OUT_DIR / "sapa_bic_exact_edge_counts.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "sapa_min_item_sensitivity_edge_counts.csv", index=False)
    pdf = build_pdf(scored, sample, item_signs, sample_sizes, pc, bic, sensitivity)

    print(f"Wrote {pdf}")
    print(f"Wrote {OUT_DIR / 'sapa_hexaco_scored_domains.csv'}")
    print(f"Wrote {OUT_DIR / 'sapa_pc_stable_edge_counts.csv'}")
    print(f"Wrote {OUT_DIR / 'sapa_bic_exact_edge_counts.csv'}")


if __name__ == "__main__":
    main()
