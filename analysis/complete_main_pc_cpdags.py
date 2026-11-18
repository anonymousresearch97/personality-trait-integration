"""Finish main-paper CPDAGs from saved full-sample skeletons and separating sets.

No data fitting or resampling occurs. For the six previously unresolved PC
patterns, reproduce the collider-initialization and deterministic conflicting-
edge retry policy of pcalg::udag2pdagSpecial (u2pd='retry', n.max=100).
The six-node extension test and final completion are performed exhaustively,
then independently checked against causal-learn's DAG2CPDAG implementation.

Source: https://github.com/cran/pcalg/blob/master/R/pcalg.R
Source package pcalg: Markus Kalisch, Alain Hauser, Martin Maechler and contributors.
Source package license: GPL (>= 2); this port retains that attribution.
Metadata: https://raw.githubusercontent.com/cran/pcalg/master/DESCRIPTION
udag2pdagSpecial, pdag2dag; source reviewed 2026-11-11.
The installed R has bnlearn but not pcalg, so this is explicitly a Python port,
not a run of R pcalg. Reversing multiply visited edges may drop initial collider
suggestions. All changes are logged. Previously valid graphs remain unchanged.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from revision_cpdag_reconstruction import (
    DOMAINS, ROOT, parse, edge_text, skeleton, colliders, pc_colliders,
    enumerate_cpdag, independent_dag2cpdag,
)

OUT = ROOT / "outputs/main_paper_cpdags"
SOURCE = ROOT / "outputs/revision_cpdag"
INDEX = {v: i for i, v in enumerate(DOMAINS)}
SEED = 20261111


def colmajor_indices(matrix):
    """Match R which(matrix, arr.ind=TRUE), which traverses columns first."""
    return [(i, j) for j in range(len(DOMAINS)) for i in range(len(DOMAINS))
            if matrix[i, j]]


def partial_arrows(pdag):
    return frozenset((DOMAINS[i], DOMAINS[j])
                     for i, j in colmajor_indices((pdag == 1) & (pdag.T == 0)))


def partial_colliders(pdag, edges):
    arrows = partial_arrows(pdag)
    skeleton_set = set(edges)
    parents = {v: {a for a, b in arrows if b == v} for v in DOMAINS}
    return frozenset((a, b, c) for b in DOMAINS
                     for a, c in itertools.combinations(sorted(parents[b]), 2)
                     if tuple(sorted((a, c))) not in skeleton_set)


def all_dags(edges):
    results = set()
    for order in itertools.permutations(DOMAINS):
        rank = {v: i for i, v in enumerate(order)}
        results.add(tuple(sorted((a, b) if rank[a] < rank[b] else (b, a)
                                 for a, b in edges)))
    return [(dag, colliders(dag)) for dag in sorted(results)]


def extension(pdag, edges, candidates, keep_vstructures=True):
    """Exact existence test equivalent to pdag2dag's consistent extension test."""
    arrows = partial_arrows(pdag)
    required = partial_colliders(pdag, edges)
    for dag, cols in candidates:
        if arrows.issubset(dag) and (not keep_vstructures or cols == required):
            return dag
    return None


def fallback_sink_extension(pdag):
    """pcalg pdag2dag(keepVstruct=FALSE): first available sink by node order.

    With no adjacency-clique restriction this may create extra colliders.
    Unlike arbitrary exhaustive representative selection, this reproduces
    pcalg's deterministic sink choice and therefore its chosen fallback class.
    """
    graph = pdag.copy()
    remaining = list(range(len(DOMAINS)))
    while remaining:
        sink = next((x for x in remaining
                     if not any(graph[x, y] == 1 and graph[y, x] == 0
                                for y in remaining)), None)
        if sink is None:
            return None
        for y in remaining:
            if graph[sink, y] == graph[y, sink] == 1:
                graph[sink, y] = 0
                graph[y, sink] = 1
        remaining.remove(sink)
    return tuple(sorted(partial_arrows(graph)))


def retry_pc(edges, separators):
    """Port pcalg collider pass and conflicting-edge subset retry order."""
    p = len(DOMAINS)
    g = np.zeros((p, p), dtype=int)
    for a, b in edges:
        i, j = INDEX[a], INDEX[b]
        g[i, j] = g[j, i] = 1
    pdag = g.copy()
    evisit = np.zeros_like(g)
    sepsets = {frozenset(row['pair']): set(row['conditioning']) for row in separators}
    # pcalg visits both x-y-z and z-y-x, in R's column-major edge order.
    for x, y in colmajor_indices(g == 1):
        for z in range(p):
            if z == x or not g[y, z]:
                continue
            if not g[x, z] and DOMAINS[y] not in sepsets[frozenset((DOMAINS[x], DOMAINS[z]))]:
                for parent in (x, z):
                    if pdag[parent, y] == 0 and pdag[y, parent] == 1:
                        evisit[parent, y] += 1
                        evisit[y, parent] += 1
                pdag[x, y] = pdag[z, y] = 1
                pdag[y, x] = pdag[y, z] = 0
    initial = pdag.copy()
    candidates = all_dags(edges)
    dag = extension(pdag, edges, candidates)
    initial_extendable = dag is not None
    pairs = []
    for i, j in colmajor_indices(evisit > 0):
        pair = tuple(sorted((i, j)))
        if pair not in pairs:
            pairs.append(pair)
    tries = 0
    if dag is None and pairs:
        # digitsBase(counter) reversed: low bit corresponds to first pair.
        for tries in range(1, min(2**len(pairs) - 1, 100) + 1):
            trial = initial.copy()
            for k, (i, j) in enumerate(pairs):
                if tries & (1 << k):
                    trial[i, j], trial[j, i] = initial[j, i], initial[i, j]
            pdag = trial
            dag = extension(pdag, edges, candidates)
            if dag is not None:
                break
    fallback = dag is None
    fallback_kind = "none"
    if fallback:
        # pcalg next relaxes keepVstruct, preserving the directed pattern if
        # possible; its final random skeleton DAG fallback is explicitly named.
        dag = fallback_sink_extension(pdag)
        fallback_kind = "allow_additional_colliders"
        if dag is None:
            order = list(DOMAINS)
            random.Random(SEED).shuffle(order)
            rank = {v: i for i, v in enumerate(order)}
            dag = tuple(sorted((a, b) if rank[a] < rank[b] else (b, a) for a, b in edges))
            fallback_kind = "seeded_random_skeleton_DAG"
    return dag, {
        "orientation_policy": "pcalg_fallback_port" if fallback else "pcalg_retry_port",
        "retries": tries, "fallback": fallback, "fallback_kind": fallback_kind,
        "initial_pattern_extendable": initial_extendable,
        "conflicting_edges": edge_text((tuple(sorted((DOMAINS[i], DOMAINS[j]))) for i, j in pairs)),
        "initial_pattern_arrows": edge_text(partial_arrows(initial), True),
        "resolved_pattern_arrows": edge_text(partial_arrows(pdag), True),
        "resolution_seed": SEED if fallback_kind == "seeded_random_skeleton_DAG" else "not_used_deterministic",
    }


def collider_text(cols):
    return "; ".join(f"{a}->{b}<-{c}" for a, b, c in sorted(cols))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(SOURCE / "network_cpdag_results.csv").fillna("")
    rows = rows[~rows.group.str.contains("Childhood")].to_dict('records')
    sep_rows = json.loads((SOURCE / "pc_separating_sets.json").read_text())
    sep_map = {(r['dataset'], r['group']): r for r in sep_rows}
    audits = []
    for row in rows:
        edges = parse(row['skeleton_edges'])
        is_pc = row['method'] == "Gaussian PC-stable"
        already_valid = row['cpdag_valid'] in (True, "True")
        row.update(orientation_policy="existing_verified", retries=0, fallback=False,
                   fallback_kind="none", deleted_colliders="", added_colliders="",
                   resolution_seed="not_used_deterministic")
        if is_pc:
            raw_cols = pc_colliders(edges, sep_map[row['dataset'], row['group']]['separating_sets'])
        else:
            raw_cols = frozenset()
        row['original_collider_suggestions'] = collider_text(raw_cols) if is_pc else row['unshielded_colliders']
        if not already_valid:
            assert is_pc
            dag, metadata = retry_pc(edges, sep_map[row['dataset'], row['group']]['separating_sets'])
            row.update(metadata)
            final_cols = colliders(dag)
            result, equivalents = enumerate_cpdag(edges, final_cols)
            assert result['cpdag_valid'] and dag in equivalents
            row.update(result, status="verified_CPDAG_after_PC_conflict_resolution",
                       unshielded_colliders=collider_text(final_cols), collider_count=len(final_cols),
                       deleted_colliders=collider_text(raw_cols - final_cols),
                       added_colliders=collider_text(final_cols - raw_cols),
                       representative_dag=edge_text(dag, True))
        else:
            # Extract any valid representative from the already verified class.
            arrows = set(parse(row['directed_edges'], True))
            undir = set(parse(row['undirected_edges']))
            candidates = all_dags(edges)
            dag = next(dag for dag, cols in candidates
                       if arrows.issubset(dag)
                       and collider_text(cols) == row['unshielded_colliders'])
            result, equivalents = enumerate_cpdag(edges, colliders(dag))
            assert result['directed_edges'] == row['directed_edges']
            assert result['undirected_edges'] == row['undirected_edges']
            row['representative_dag'] = edge_text(dag, True)
        idir, iundir = independent_dag2cpdag(dag)
        assert idir == row['directed_edges'] and iundir == row['undirected_edges']
        assert skeleton(dag) == edges
        assert int(row['directed_count']) + int(row['undirected_count']) == int(row['edge_count'])
        row['independent_dag2cpdag_matches'] = True
        audits.append(dict(dataset=row['dataset'], group=row['group'], method=row['method'],
                           skeleton_unchanged=True, exhaustive_class_verified=True,
                           independent_conversion_matches=True, preserved_previous_cpdag=already_valid))
        print(f"{row['dataset']} | {row['group']} | {row['method']}: "
              f"{int(row['directed_count'])}/{int(row['edge_count'])} directed; "
              f"{row['orientation_policy']}; retries={row['retries']}; fallback={row['fallback']}", flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 20 and frame.cpdag_valid.all()
    frame.to_csv(OUT / "network_cpdag_results.csv", index=False)
    (OUT / "cpdag_validation.json").write_text(json.dumps({
        "networks": 20, "all_valid": True, "no_resampling": True,
        "pc_conflict_resolution_graphs": int((frame.orientation_policy != 'existing_verified').sum()),
        "fallback_graphs": int(frame.fallback.sum()), "checks": audits,
        "pcalg_source": "https://github.com/cran/pcalg/blob/master/R/pcalg.R",
        "policy": "Python port of pcalg::udag2pdagSpecial initialization and conflict-edge retries; exact extension and CPDAG completion; existing valid CPDAGs unchanged",
        "node_order": DOMAINS, "max_retry_subsets": 100,
    }, indent=2) + '\n')
    (OUT / "orientation_method_note.txt").write_text(
        "FULL-SAMPLE CPDAG COMPLETION (2026-11-11)\n\n"
        "All 20 networks are single fits to observed complete age-group or wave samples. "
        "No subsampling or bootstrap orientation analysis was performed. Bootstrap and "
        "subsampling outputs elsewhere concern skeleton edges/counts only.\n\n"
        "Ten Gaussian BIC CPDAGs and four previously valid Gaussian PC CPDAGs were "
        "preserved. Six PC skeletons required conflict resolution among finite-sample "
        "collider suggestions. We used a Python port of the documented pcalg "
        "udag2pdagSpecial (u2pd='retry') collider pass and binary-subset conflicting-edge "
        "retry policy, maximum 100 subsets. Node order H,E,X,A,C,O; R column-major "
        "traversal; low bit flips the first conflicting edge. This policy is deterministic "
        "unless its final random-skeleton fallback is required. Exact enumeration of "
        "all 6! topological orders replaces the consistent-extension test and final "
        "completion. For a successful consistent extension, representative choice "
        "does not change its equivalence class. The one fallback uses pcalg's "
        "first-sink ordering with keepVstruct=FALSE, permitting additional colliders "
        "and selecting a new class. No random-skeleton fallback was required. "
        "An independent causal-learn DAG-to-CPDAG conversion checks every "
        "result. The source is https://github.com/cran/pcalg/blob/master/R/pcalg.R .\n\n"
        f"Fallback graphs: {int(frame.fallback.sum())}. Every original skeleton and link "
        "count is unchanged. The CSV reports each changed collider and conflict "
        "resolution policy. Arrows are compelled within the selected equivalence "
        "class; they are not bootstrap-supported directions or demonstrated causal "
        "effects. For conflict-resolved PC panels the class depends on the documented "
        "orientation policy, not on jointly compatible test evidence alone.\n")


if __name__ == '__main__':
    main()
