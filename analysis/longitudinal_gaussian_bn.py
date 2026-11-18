"""Small, exact Gaussian BN sensitivity analysis for continuous trait scores.

This module does not select a transformation, alpha, or method to recover a
preferred substantive result. The fixed defaults use untransformed continuous
domain scores, Gaussian PC-stable (Fisher z, alpha=.05), and a globally optimal
Gaussian BIC DAG. These assumptions differ from the earlier discrete models.
Observational edge directions are not causal conclusions. In particular, a
single deterministic optimal DAG can represent a score-equivalent family.

Run this file directly for numerical and synthetic validation only. No study
dataset is read or analyzed by the validation command.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import ndtr


DEFAULT_NODES = ("H", "E", "X", "A", "C", "O")
TIE_TOLERANCE = 1e-9


def _bit_count(value: int) -> int:
    # The project may run under the system Python 3.9 runtime.
    return bin(value).count("1")


def _validate_nodes(nodes: Sequence[str], p: int) -> tuple[str, ...]:
    nodes = tuple(nodes)
    if len(nodes) != p or len(set(nodes)) != p:
        raise ValueError("Supply exactly one unique label for each column.")
    if not all(isinstance(node, str) for node in nodes):
        raise ValueError("Node labels must be strings.")
    return nodes


def _validate_covariance(cov: np.ndarray, n: int) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 1:
        raise ValueError("Expected a square covariance/correlation matrix.")
    if int(n) != n or n <= cov.shape[0]:
        raise ValueError("Sample size must be an integer greater than node count.")
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Covariance must be finite and symmetric.")
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Positive definite covariance required; no ridge is added.")
    return cov


def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MLE covariance (divide by N) and Pearson correlation, without filtering."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] <= values.shape[1]:
        raise ValueError("Expected N-by-P scores with N > P.")
    if not np.isfinite(values).all():
        raise ValueError("Provide explicitly selected complete finite rows.")
    centered = values - values.mean(axis=0)
    cov = np.einsum("ni,nj->ij", centered, centered) / len(values)
    cov = _validate_covariance(cov, len(values))
    sd = np.sqrt(np.diag(cov))
    return cov, cov / np.outer(sd, sd)


def gaussian_local_scores(cov: np.ndarray, n: int) -> np.ndarray:
    """Return log-likelihood minus half the BIC penalty for every parent set.

    For k parents, local MLE log L is -N/2*(log(2*pi)+1+log(RSS/N)).
    The free parameter count is k+2: intercept, k slopes, residual variance.
    Use an MLE covariance, not the unbiased covariance, for absolute fit scores.
    Correlations yield the same selected structure and gains vs independence,
    but a graph-independent shift of absolute log-likelihood and BIC.
    """
    cov = _validate_covariance(cov, n)
    p = len(cov)
    if p > 16:
        raise ValueError("Exact subset search is intended for small networks.")
    scores = np.full((p, 1 << p), -np.inf)
    for family in range(1, 1 << p):
        members = [i for i in range(p) if family & (1 << i)]
        precision = np.linalg.inv(cov[np.ix_(members, members)])
        variances = 1.0 / np.diag(precision)
        if np.any(variances <= 0) or not np.isfinite(variances).all():
            raise ValueError("Nonpositive residual variance in Gaussian fit.")
        parameters = len(members) + 1
        local = (-0.5 * n * (math.log(2 * math.pi) + 1 + np.log(variances))
                 - 0.5 * parameters * math.log(n))
        for child, score in zip(members, local):
            scores[child, family ^ (1 << child)] = score
    return scores


def exact_gaussian_bic(cov: np.ndarray, n: int,
                       nodes: Sequence[str] = DEFAULT_NODES):
    """Return (max logL-.5*k*logN, one optimal named DAG).

    Subset dynamic programming searches all DAGs exactly. Floating-point ties
    (within 1e-9 score units) prefer fewer edges, then deterministic index order.
    This tie rule is not a substantive direction identification rule.
    """
    local = gaussian_local_scores(cov, n)
    p, size = local.shape
    nodes = _validate_nodes(nodes, p)
    best_score = local.copy()
    best_parents = np.tile(np.arange(size, dtype=np.int64), (p, 1))
    for child in range(p):
        for allowed in range(size):
            if allowed & (1 << child):
                continue
            for parent in range(p):
                if not allowed & (1 << parent):
                    continue
                reduced = allowed ^ (1 << parent)
                candidate = best_score[child, reduced]
                current = best_score[child, allowed]
                new_parents = int(best_parents[child, reduced])
                old_parents = int(best_parents[child, allowed])
                preferable_tie = (abs(candidate - current) <= TIE_TOLERANCE
                                  and (_bit_count(new_parents), new_parents)
                                  < (_bit_count(old_parents), old_parents))
                if candidate > current + TIE_TOLERANCE or preferable_tie:
                    best_score[child, allowed] = candidate
                    best_parents[child, allowed] = new_parents
    dp = np.full(size, -np.inf)
    edge_count = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    dp[0], edge_count[0] = 0.0, 0
    choices = {}
    for subset in range(1, size):
        for sink in range(p):
            if not subset & (1 << sink):
                continue
            remaining = subset ^ (1 << sink)
            parents = int(best_parents[sink, remaining])
            score = dp[remaining] + best_score[sink, remaining]
            edges = int(edge_count[remaining]) + _bit_count(parents)
            if (score > dp[subset] + TIE_TOLERANCE
                    or (abs(score - dp[subset]) <= TIE_TOLERANCE
                        and edges < edge_count[subset])):
                dp[subset] = score
                edge_count[subset] = edges
                choices[subset] = (sink, parents)
    edges = []
    subset = size - 1
    while subset:
        child, parents = choices[subset]
        edges.extend((nodes[parent], nodes[child]) for parent in range(p)
                     if parents & (1 << parent))
        subset ^= 1 << child
    return float(dp[-1]), tuple(sorted(edges))


def skeleton(dag):
    return tuple(sorted(tuple(sorted(edge)) for edge in dag))


def partial_correlation(corr, i, j, conditioning):
    if len(conditioning) == 0:
        return float(corr[i, j])
    family = [i, j, *conditioning]
    precision = np.linalg.inv(corr[np.ix_(family, family)])
    return float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))


def fisher_pvalue(r, n, conditioning_size):
    if n <= conditioning_size + 3:
        raise ValueError("Insufficient observations for the Fisher-z test.")
    r = float(np.clip(r, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps))
    statistic = np.arctanh(r) * np.sqrt(n - conditioning_size - 3)
    return float(2 * ndtr(-abs(statistic)))


def pc_stable_details(corr, n, alpha=.05, nodes=DEFAULT_NODES,
                      include_tests=False):
    """Gaussian PC-stable skeleton with separating sets and optional test log.

    Both endpoints' frozen adjacency sets are considered at each conditioning
    depth. Node names determine a fixed test order. This is skeleton discovery;
    causal sufficiency, faithfulness, and causal directions are not established.
    """
    corr = _validate_covariance(corr, n)
    if not np.allclose(np.diag(corr), 1, atol=1e-10, rtol=1e-10):
        raise ValueError("PC expects a correlation matrix with unit diagonal.")
    if n <= len(corr) + 1 or not 0 < alpha < 1:
        raise ValueError("Invalid sample size or alpha for PC-stable.")
    nodes = _validate_nodes(nodes, len(corr))
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = {node: set(nodes) - {node} for node in nodes}
    precision_cache = {}
    sepsets, tests = [], []
    n_tests = 0
    for level in range(max(1, len(nodes) - 1)):
        snapshot = {node: set(neighbors) for node, neighbors in adjacency.items()}
        pairs = sorted((a, b) for a in nodes for b in snapshot[a] if a < b)
        if not any(len(snapshot[a] - {b}) >= level or
                   len(snapshot[b] - {a}) >= level for a, b in pairs):
            break
        for x, y in pairs:
            candidates = set()
            for base in (snapshot[x] - {y}, snapshot[y] - {x}):
                if len(base) >= level:
                    candidates.update(itertools.combinations(sorted(base), level))
            for conditioning in sorted(candidates):
                family = tuple(sorted(index[node] for node in (x, y, *conditioning)))
                if level == 0:
                    r = float(corr[index[x], index[y]])
                else:
                    if family not in precision_cache:
                        precision_cache[family] = np.linalg.inv(corr[np.ix_(family, family)])
                    precision = precision_cache[family]
                    i, j = family.index(index[x]), family.index(index[y])
                    r = float(-precision[i, j] / np.sqrt(precision[i, i] * precision[j, j]))
                pvalue = fisher_pvalue(r, n, level)
                n_tests += 1
                if include_tests:
                    tests.append(dict(pair=(x, y), conditioning=conditioning,
                                      partial_r=r, pvalue=pvalue))
                if pvalue > alpha:
                    adjacency[x].discard(y)
                    adjacency[y].discard(x)
                    sepsets.append(dict(pair=(x, y), conditioning=conditioning,
                                        partial_r=r, pvalue=pvalue))
                    break
    edges = tuple(sorted((a, b) for a in nodes for b in adjacency[a] if a < b))
    result = dict(skeleton=edges, edge_count=len(edges), alpha=float(alpha),
                  n_tests=n_tests, separating_sets=sepsets)
    if include_tests:
        result["tests"] = tests
    return result


def pc_stable_gaussian(corr, n, alpha=.05, nodes=DEFAULT_NODES):
    return pc_stable_details(corr, n, alpha, nodes)["skeleton"]


def bic_details(cov, n, nodes=DEFAULT_NODES):
    score, dag = exact_gaussian_bic(cov, n, nodes)
    p = len(cov)
    independent_score = float(gaussian_local_scores(cov, n)[:, 0].sum())
    parameters = 2 * p + len(dag)
    log_likelihood = score + 0.5 * parameters * math.log(n)
    return dict(dag=dag, skeleton=skeleton(dag), edge_count=len(dag),
                score=score, independent_score=independent_score,
                score_gain_vs_independence=score - independent_score,
                conventional_bic=-2 * score,
                independent_conventional_bic=-2 * independent_score,
                conventional_bic_improvement=2 * (score - independent_score),
                log_likelihood=log_likelihood, free_parameters=parameters,
                direction_interpretation="One deterministic optimum; observational directions are not causal.")


def fit_gaussian_bn(values, nodes=DEFAULT_NODES, alpha=.05, include_tests=False):
    """Fit both fixed Gaussian models; return a serializable dictionary."""
    values = np.asarray(values, dtype=float)
    cov, corr = moments(values)
    nodes = _validate_nodes(nodes, values.shape[1])
    off_diagonal = corr[np.triu_indices(len(nodes), 1)]
    return dict(n=len(values), nodes=nodes, means=values.mean(axis=0).tolist(),
                sd=np.sqrt(np.diag(cov)).tolist(), covariance=cov.tolist(),
                correlation=corr.tolist(),
                max_abs_marginal_correlation=float(np.max(np.abs(off_diagonal))) if len(off_diagonal) else 0.0,
                pc=pc_stable_details(corr, len(values), alpha, nodes, include_tests),
                bic=bic_details(cov, len(values), nodes))


def _ols_reference(values, child, parents):
    design = np.column_stack([np.ones(len(values)), values[:, parents]])
    beta = np.linalg.lstsq(design, values[:, child], rcond=None)[0]
    residual = values[:, child] - np.einsum("ij,j->i", design, beta)
    variance = float(np.sum(residual ** 2) / len(values))
    return (-0.5 * len(values) * (math.log(2 * math.pi) + 1 + math.log(variance))
            - 0.5 * (len(parents) + 2) * math.log(len(values)))


def validate():
    """Independent OLS, exhaustive DAG, residual correlation, and known-graph checks."""
    import networkx as nx
    import original_continuous_bn_sensitivity as previous

    rng = np.random.default_rng(73592)
    # Construct exact sample covariance for clear deterministic recovery tests.
    z = rng.normal(size=(2000, 6))
    z -= z.mean(axis=0)
    q, _ = np.linalg.qr(z)
    independent = q * np.sqrt(len(q))
    chain = independent.copy()
    chain[:, 1] += .75 * chain[:, 0]
    chain[:, 2] += .65 * chain[:, 1]
    common_cause = independent.copy()
    common_cause[:, 1] += .8 * common_cause[:, 0]
    common_cause[:, 2] -= .7 * common_cause[:, 0]
    collider = independent.copy()
    collider[:, 2] += .7 * collider[:, 0] + .65 * collider[:, 1]
    cases = {"independent": (independent, ()),
             "chain": (chain, (("V0", "V1"), ("V1", "V2"))),
             "common_cause": (common_cause, (("V0", "V1"), ("V0", "V2"))),
             "collider": (collider, (("V0", "V2"), ("V1", "V2")))}
    results = []
    max_local_error = max_partial_error = max_bic_error = 0.0
    local_count = partial_count = exhaustive_count = 0
    for label, (values, expected) in cases.items():
        nodes = tuple(f"V{i}" for i in range(6))
        result = fit_gaussian_bn(values, nodes, include_tests=True)
        assert result["pc"]["skeleton"] == expected
        assert result["bic"]["skeleton"] == expected
        cov, corr = moments(values)
        local = gaussian_local_scores(cov, len(values))
        for child in range(6):
            possible = [i for i in range(6) if i != child]
            for k in range(6):
                for parents in itertools.combinations(possible, k):
                    actual = local[child, sum(1 << i for i in parents)]
                    error = abs(actual - _ols_reference(values, child, list(parents)))
                    max_local_error = max(max_local_error, error)
                    local_count += 1
                    assert error < 1e-8
        for i, j in itertools.combinations(range(6), 2):
            possible = [k for k in range(6) if k not in (i, j)]
            for k in range(5):
                for conditioning in itertools.combinations(possible, k):
                    design = np.column_stack([np.ones(len(values)), values[:, conditioning]])
                    pair = values[:, [i, j]]
                    beta = np.linalg.lstsq(design, pair, rcond=None)[0]
                    residuals = pair - np.einsum("ij,jk->ik", design, beta)
                    reference = np.corrcoef(residuals.T)[0, 1]
                    error = abs(reference - partial_correlation(corr, i, j, conditioning))
                    partial_count += 1
                    max_partial_error = max(max_partial_error, error)
                    assert error < 1e-10
        # Raw OLS scoring of all 543 possible four-node DAGs, independent of DP.
        four = values[:, :4]
        local_ols = {(i, parents): _ols_reference(four, i, list(parents))
                     for i in range(4) for k in range(4)
                     for parents in itertools.combinations([j for j in range(4) if j != i], k)}
        pairs = list(itertools.combinations(range(4), 2))
        enumerated = []
        for states in itertools.product((0, 1, 2), repeat=len(pairs)):
            dag = [pair if state == 1 else pair[::-1]
                   for pair, state in zip(pairs, states) if state]
            graph = nx.DiGraph()
            graph.add_nodes_from(range(4)); graph.add_edges_from(dag)
            if nx.is_directed_acyclic_graph(graph):
                enumerated.append(sum(local_ols[i, tuple(sorted(graph.predecessors(i)))] for i in range(4)))
        assert len(enumerated) == 543
        exhaustive_count += len(enumerated)
        score, _ = exact_gaussian_bic(moments(four)[0], len(four), nodes[:4])
        error = abs(score - max(enumerated))
        max_bic_error = max(max_bic_error, error)
        assert error < 1e-8
        old_score, old_dag = previous.exact_gaussian_bic(cov, len(values), nodes)
        assert abs(old_score - result["bic"]["score"]) < 1e-8
        assert skeleton(old_dag) == result["bic"]["skeleton"]
        assert previous.pc_stable_gaussian(corr, len(values), nodes=nodes) == expected
        assert pc_stable_gaussian(corr[::-1, ::-1], len(values), nodes=nodes[::-1]) == expected
        standardized = bic_details(corr, len(values), nodes)
        assert abs(standardized["score_gain_vs_independence"] - result["bic"]["score_gain_vs_independence"]) < 1e-8
        results.append(dict(case=label, n=len(values), expected_skeleton=expected,
                            pc_skeleton=result["pc"]["skeleton"],
                            bic_skeleton=result["bic"]["skeleton"],
                            bic_score_gain_vs_independence=result["bic"]["score_gain_vs_independence"]))
    # Random correlated samples check nontrivial variable magnitudes and means.
    for p in (5, 6):
        transform = rng.normal(size=(p, p)) + np.eye(p) * 2
        values = (np.einsum("ij,jk->ik", rng.normal(size=(429, p)), transform)
                  * np.linspace(.4, 2.5, p) + np.arange(p) * 1.7)
        cov, corr = moments(values)
        nodes = tuple(f"V{i}" for i in range(p))
        score, dag = exact_gaussian_bic(cov, len(values), nodes)
        reference, _ = previous.exact_gaussian_bic(cov, len(values), nodes)
        assert abs(score - reference) < 1e-8
        assert pc_stable_gaussian(corr, len(values), nodes=nodes) == previous.pc_stable_gaussian(corr, len(values), nodes=nodes)
        graph = nx.DiGraph(); graph.add_nodes_from(nodes); graph.add_edges_from(dag)
        assert nx.is_directed_acyclic_graph(graph)
        raw_score = sum(_ols_reference(values, i, [nodes.index(parent) for parent in graph.predecessors(node)])
                        for i, node in enumerate(nodes))
        assert abs(score - raw_score) < 1e-8
    out = Path(__file__).resolve().parents[1] / "outputs" / "longitudinal_bn_validation"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(passed=True, synthetic_cases=results,
                  local_ols_comparisons=local_count, max_local_ols_absolute_error=max_local_error,
                  partial_correlation_residual_comparisons=partial_count,
                  max_partial_correlation_absolute_error=max_partial_error,
                  exhaustive_four_node_dags_scored=exhaustive_count,
                  max_exhaustive_vs_dynamic_bic_error=max_bic_error,
                  previous_implementation_matches=True, pc_reversed_node_order_matches=True,
                  covariance_vs_correlation_bic_gains_match=True,
                  random_five_and_six_node_dags_acyclic_and_raw_ols_scores_match=True,
                  model_selection="Defaults fixed; no selection to recover preferred scientific results.",
                  direction_interpretation="One deterministic optimum; observational edge directions are not causal.")
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    validate()
