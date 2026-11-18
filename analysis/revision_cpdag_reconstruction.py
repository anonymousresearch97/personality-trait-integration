"""Strict CPDAG reconstruction for the fixed Gaussian revision analyses.

Replays the original full-sample PC skeleton algorithm to recover separating
sets, retaining its exact ordering. BIC uses the saved optimum without fitting.
All 6! topological orders enumerate every possible acyclic orientation of a
fixed six-node skeleton; distinct DAGs are deduplicated and required to have
exactly the selected unshielded colliders. Zero extensions is an explicit
failure, never an all-undirected CPDAG fallback. No resampling is performed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

from original_continuous_bn_sensitivity import (
    DOMAINS, moments, fisher_pvalue, pc_stable_gaussian,
)
from longitudinal_gaussian_bn import pc_stable_details

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/revision_cpdag"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "analysis/vendor/continuous_ci"))


def parse(text, directed=False):
    if not isinstance(text, str) or not text or text == "none":
        return tuple()
    return tuple(sorted(tuple(e.strip().split("->" if directed else "-"))
                        for e in text.replace(";", ",").split(",")))


def edge_text(edges, directed=False):
    return ", ".join(("->" if directed else "-").join(e) for e in sorted(edges))


def skeleton(dag):
    return tuple(sorted(tuple(sorted(e)) for e in dag))


def colliders(dag):
    skel = set(skeleton(dag))
    parents = {v: set() for v in DOMAINS}
    for a, b in dag:
        parents[b].add(a)
    return frozenset((a, middle, b) for middle in DOMAINS
                     for a, b in itertools.combinations(sorted(parents[middle]), 2)
                     if tuple(sorted((a, b))) not in skel)


def pc_details_original(corr, n, alpha=.05):
    """Exact original_continuous_bn_sensitivity ordering with sepset logging."""
    nodes = list(DOMAINS)
    index = {name: i for i, name in enumerate(nodes)}
    adjacency = {node: set(nodes) - {node} for node in nodes}
    precision_cache, sepsets = {}, []
    level, n_tests = 0, 0
    while True:
        snapshot = {node: set(neighbors) for node, neighbors in adjacency.items()}
        edges = sorted((a, b) for a in nodes for b in snapshot[a] if a < b)
        for x, y in edges:
            candidates = []
            for base in (snapshot[x] - {y}, snapshot[y] - {x}):
                if len(base) >= level:
                    candidates.extend(itertools.combinations(sorted(base), level))
            seen = set()
            for conditioning in candidates:
                if conditioning in seen:
                    continue
                seen.add(conditioning)
                if not conditioning:
                    r = corr[index[x], index[y]]
                else:
                    family = tuple(sorted(index[node] for node in (x, y, *conditioning)))
                    if family not in precision_cache:
                        precision_cache[family] = np.linalg.inv(corr[np.ix_(family, family)])
                    precision = precision_cache[family]
                    i, j = family.index(index[x]), family.index(index[y])
                    r = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
                pvalue = fisher_pvalue(r, n, len(conditioning))
                n_tests += 1
                if pvalue > alpha:
                    adjacency[x].discard(y)
                    adjacency[y].discard(x)
                    sepsets.append(dict(pair=(x, y), conditioning=conditioning,
                                        partial_r=float(r), pvalue=pvalue))
                    break
        level += 1
        if not any(len(neighbors) >= level + 1 for neighbors in adjacency.values()) or level > len(nodes) - 2:
            break
    edges = tuple(sorted((a, b) for a in nodes for b in adjacency[a] if a < b))
    assert edges == pc_stable_gaussian(corr, n, alpha=alpha)
    return dict(skeleton=edges, separating_sets=sepsets, n_tests=n_tests)


def pc_colliders(edges, separators):
    adjacency = {v: set() for v in DOMAINS}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    sepsets = {tuple(row["pair"]): set(row["conditioning"]) for row in separators}
    result = set()
    for middle in DOMAINS:
        for a, b in itertools.combinations(sorted(adjacency[middle]), 2):
            if b in adjacency[a]:
                continue
            assert (a, b) in sepsets, (a, b, "missing separating set")
            if middle not in sepsets[(a, b)]:
                result.add((a, middle, b))
    return frozenset(result)


def enumerate_cpdag(edges, required_colliders):
    valid_dags = set()
    for order in itertools.permutations(DOMAINS):
        index = {v: i for i, v in enumerate(order)}
        dag = tuple(sorted((a, b) if index[a] < index[b] else (b, a) for a, b in edges))
        if colliders(dag) == required_colliders:
            valid_dags.add(dag)
    if not valid_dags:
        return dict(cpdag_valid=False, status="no_consistent_DAG_extension",
                    equivalence_class_size=0, directed_edges="", undirected_edges="",
                    directed_count=None, undirected_count=None), valid_dags
    common = set.intersection(*(set(dag) for dag in valid_dags))
    reversible = [e for e in edges if e not in common and e[::-1] not in common]
    return dict(cpdag_valid=True, status="verified_CPDAG",
                equivalence_class_size=len(valid_dags), directed_edges=edge_text(common, True),
                undirected_edges=edge_text(reversible), directed_count=len(common),
                undirected_count=len(reversible)), valid_dags


def independent_dag2cpdag(dag):
    # causal-learn imports pydot unconditionally for GraphUtils.to_pydot,
    # although its graph/matrix and CPDAG code never uses that optional renderer.
    # An explicit fail-on-access module allows that unused import only; no
    # graph operation is replaced or patched, and invoking rendering would fail.
    import importlib.util
    import types
    if "pydot" not in sys.modules and importlib.util.find_spec("pydot") is None:
        missing = types.ModuleType("pydot")
        def rendering_unavailable(name):
            raise ImportError(f"Optional pydot renderer unavailable: {name}")
        missing.__getattr__ = rendering_unavailable
        sys.modules["pydot"] = missing
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag
    nodes = [GraphNode(v) for v in DOMAINS]
    lookup = dict(zip(DOMAINS, nodes))
    graph = Dag(nodes)
    for a, b in dag:
        graph.add_directed_edge(lookup[a], lookup[b])
    cp = dag2cpdag(graph)
    directed, undirected = [], []
    for i, j in itertools.combinations(range(len(DOMAINS)), 2):
        endpoint_i, endpoint_j = cp.graph[i, j], cp.graph[j, i]
        if endpoint_i == -1 and endpoint_j == 1:
            directed.append((DOMAINS[i], DOMAINS[j]))
        elif endpoint_i == 1 and endpoint_j == -1:
            directed.append((DOMAINS[j], DOMAINS[i]))
        elif endpoint_i == endpoint_j == -1:
            undirected.append(tuple(sorted((DOMAINS[i], DOMAINS[j]))))
        else:
            assert endpoint_i == endpoint_j == 0
    return edge_text(directed, True), edge_text(undirected)


def process(dataset, group, values, pc_ref, bic_ref, source, longitudinal=False):
    _, corr = moments(values)
    details = pc_stable_details(corr, len(values)) if longitudinal else pc_details_original(corr, len(values))
    old_pc = parse(pc_ref["skeleton_edges"])
    assert details["skeleton"] == old_pc, (dataset, group, "PC skeleton mismatch")
    sepsets_log.append(dict(dataset=dataset, group=group, **details))
    dags = parse(bic_ref["one_optimal_dag"], directed=True)
    assert skeleton(dags) == parse(bic_ref["skeleton_edges"])
    for method, edges, required, selected_dag in [
        ("Gaussian PC-stable", old_pc, pc_colliders(old_pc, details["separating_sets"]), None),
        ("Gaussian BIC", skeleton(dags), colliders(dags), dags),
    ]:
        result, equivalents = enumerate_cpdag(edges, required)
        check = None
        if result["cpdag_valid"]:
            reference_dag = selected_dag or min(equivalents)
            indep_dir, indep_undir = independent_dag2cpdag(reference_dag)
            assert indep_dir == result["directed_edges"] and indep_undir == result["undirected_edges"]
            assert result["directed_count"] + result["undirected_count"] == len(edges)
            if selected_dag:
                assert selected_dag in equivalents
            check = True
        row = dict(dataset=dataset, method=method, group=group, n=len(values),
                   skeleton_edges=edge_text(edges), edge_count=len(edges), **result,
                   unshielded_colliders="; ".join(f"{a}->{m}<-{b}" for a, m, b in sorted(required)),
                   collider_count=len(required), independent_dag2cpdag_matches=check,
                   source_result=str(source.relative_to(ROOT)))
        rows.append(row)
        print(f"{dataset} | {group} | {method}: {len(edges)} edges; "
              f"{row['directed_count']} directed, {row['undirected_count']} reversible; "
              f"{row['equivalence_class_size']} DAGs; {row['status']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "network_cpdag_results.csv", index=False)


def main():
    # Fixed structural checks, independent of the scientific outcomes.
    toy, _ = enumerate_cpdag((("A", "E"), ("E", "X")), frozenset())
    assert toy["equivalence_class_size"] == 3 and toy["directed_count"] == 0
    toy, _ = enumerate_cpdag((("A", "E"), ("E", "X")), frozenset({("A", "E", "X")}))
    assert toy["equivalence_class_size"] == 1 and toy["directed_count"] == 2
    toy, _ = enumerate_cpdag(tuple(itertools.combinations(sorted(DOMAINS), 2)), frozenset())
    assert toy["equivalence_class_size"] == 720 and toy["directed_count"] == 0
    primary_source = ROOT / "outputs/original_continuous_bn_sensitivity/full_sample_results.csv"
    primary = pd.read_csv(ROOT / "outputs/original_hexaco_scored_domains.csv")
    results = pd.read_csv(primary_source)
    for group in results.age_group.unique():
        sub = results[results.age_group == group]
        pc = sub[sub.method.str.contains("PC")].iloc[0]
        bic = sub[sub.method.str.contains("BIC")].iloc[0]
        values = primary[primary.age_group == group][DOMAINS].to_numpy()
        assert len(values) == pc.n == bic.n
        process("Primary", group, values, pc, bic, primary_source)
    sapa_source = ROOT / "outputs/sapa_gaussian_wrapup/full_sample_results.csv"
    sapa = pd.read_csv(ROOT / "outputs/sapa_hexaco_scored_domains.csv")
    collapse = {"18 and under": "18 and under", "19-24": "19-39", "25-29": "19-39",
                "30-34": "19-39", "35-39": "19-39", "40-49": "40-59", "50-59": "40-59",
                "60 and over": "60 and over"}
    sapa["pooled"] = sapa.age_band.map(collapse)
    results = pd.read_csv(sapa_source).query("scheme == 'pooled_four'")
    for group in results.age_band.unique():
        sub = results[results.age_band == group]
        pc = sub[sub.method.str.contains("PC")].iloc[0]
        bic = sub[sub.method.str.contains("BIC")].iloc[0]
        values = sapa[sapa.pooled == group][DOMAINS].to_numpy()
        assert len(values) == pc.n == bic.n
        process("SAPA", group, values, pc, bic, sapa_source)
    community_source = ROOT / "outputs/longitudinal_bn_followup/all_wave_networks.csv"
    community = pd.read_csv(ROOT / "outputs/longitudinal_bn_followup/prepared/thielmann_community.csv")
    results = pd.read_csv(community_source).query("dataset == 'thielmann_community' and representation == 'continuous'")
    results = results.rename(columns={"edges": "skeleton_edges", "dag": "one_optimal_dag"})
    for wave in (1, 2):
        sub = results[results.wave == wave]
        pc = sub[sub.method == "PC"].iloc[0]
        bic = sub[sub.method == "BIC"].iloc[0]
        values = community[community.wave == wave].sort_values("id")[DOMAINS].to_numpy()
        assert len(values) == pc.n == bic.n == 429
        process("Thielmann community", f"Wave {wave} ({2008 if wave == 1 else 2018})",
                values, pc, bic, community_source, longitudinal=True)
    (OUT / "pc_separating_sets.json").write_text(json.dumps(sepsets_log, indent=2) + "\n")
    frame = pd.DataFrame(rows)
    # Independently establish every failed extension by enumerating all 2^m
    # edge-direction bit patterns and testing acyclicity with NetworkX.
    import networkx as nx
    invalid_checks = []
    for row in frame.loc[~frame.cpdag_valid].to_dict("records"):
        details = next(d for d in sepsets_log if d["dataset"] == row["dataset"] and d["group"] == row["group"])
        edges = parse(row["skeleton_edges"])
        required = pc_colliders(edges, details["separating_sets"])
        matches, acyclic = 0, 0
        for bits in itertools.product((0, 1), repeat=len(edges)):
            g = nx.DiGraph()
            g.add_nodes_from(DOMAINS)
            g.add_edges_from((a, b) if bit == 0 else (b, a) for (a, b), bit in zip(edges, bits))
            if not nx.is_directed_acyclic_graph(g):
                continue
            acyclic += 1
            found = set()
            for center in DOMAINS:
                for a, b in itertools.combinations(sorted(g.predecessors(center)), 2):
                    if not g.has_edge(a, b) and not g.has_edge(b, a):
                        found.add((a, center, b))
            matches += found == required
        assert matches == 0
        invalid_checks.append(dict(dataset=row["dataset"], group=row["group"],
                                   direction_patterns=2**len(edges), acyclic_orientations=acyclic,
                                   compatible_dags=matches))
    source_paths = [primary_source, sapa_source, community_source,
                    ROOT / "outputs/original_hexaco_scored_domains.csv",
                    ROOT / "outputs/sapa_hexaco_scored_domains.csv",
                    ROOT / "outputs/longitudinal_bn_followup/prepared/thielmann_community.csv"]
    validation = dict(
        rows=len(frame), all_original_skeletons_reproduced=True,
        valid_cpdag_rows=int(frame.cpdag_valid.sum()),
        invalid_rows=frame.loc[~frame.cpdag_valid, ["dataset", "method", "group", "status"]].to_dict("records"),
        all_valid_rows_independently_match_causal_learn=True,
        causal_learn_optional_renderer="pydot absent; fail-on-access placeholder satisfies unused import; no graph operations patched",
        enumeration="All 720 topological orders; unique DAGs with exactly the same skeleton and unshielded colliders",
        invalid_rows_independently_confirmed_by_networkx=invalid_checks,
        synthetic_chain_collider_and_complete_graph_checks_passed=True,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        resampling_performed=False,
    )
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    notes = """CPDAG reconstruction for revision network figures

Only full-sample PC skeleton discovery was repeated, to recover its separating
sets. All saved skeleton edges and sample sizes are reproduced exactly. The
primary/SAPA Gaussian PC test ordering is preserved from its original engine;
the community Gaussian PC uses its original longitudinal engine. Alpha=.05.
BIC uses each saved exact-Gaussian optimum, without refitting or resampling.

For each six-node skeleton, all 720 topological orders are examined. This covers
all acyclic orientations; duplicates are removed. Only DAGs with exactly the
same unshielded colliders are retained. An arrow is drawn only if its direction
is shared by every retained DAG. Reversible links are undirected. Every valid
reconstruction is independently checked using causal-learn dag2cpdag.

PC collider constraints can be inconsistent in finite samples. If no DAG has
exactly its skeleton and collider set, validity is false and arrow/reversible
counts are missing. Such a panel must not be called a CPDAG. No contradiction
is silently repaired and no all-undirected fallback is described as a CPDAG.

This changes graphical representation only: skeleton edge counts, equal-N
results, uncertainty distributions, and edge-selection frequencies remain
unchanged. Compelled arrows are statements about the selected observational
Markov-equivalence class, not confirmed causal influence, temporal ordering,
or within-person mechanisms. Ordinary Gaussian BIC is score equivalent.

PC skeleton adjacencies are not simply marginal covariance or full-order
partial correlations. PC searches for separating sets among subsets of other
variables; BIC optimizes a DAG factorization. An absent DAG-skeleton link does
not mean independence after conditioning on all other variables (conditioning
on a collider can create dependence). Gaussian CI tests use partial correlation
for the particular tested conditioning set, with distributional assumptions.
"""
    (OUT / "reconstruction_notes.txt").write_text(notes)


rows, sepsets_log = [], []
if __name__ == "__main__":
    main()
