"""Bounded independent audit of manuscript cross-sectional Gaussian engines.

No original output is changed. This checks every reported full fit, reconstructs
all equal-N memberships, refits three draws per group, and independently
rebuilds all mean/SD/edge-frequency summaries. It does not rerun 8,000 samples.
"""
from __future__ import annotations
import hashlib
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import networkx as nx
import original_continuous_bn_sensitivity as engine
import fast_exact_bic
import hexaco_bic_exhaustive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/submission_code_audit/core"
NODES = engine.DOMAINS
SAPA_BANDS = ["18 and under", "19-39", "40-59", "60 and over"]
COLLAPSE = dict(zip(["18 and under", "19-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60 and over"],
                    [SAPA_BANDS[0]] + [SAPA_BANDS[1]] * 4 + [SAPA_BANDS[2]] * 2 + [SAPA_BANDS[3]]))


def subsets(items):
    return [s for k in range(len(items) + 1) for s in itertools.combinations(items, k)]


def independent_local_ols(x):
    """Direct raw-value least squares, separate from covariance/precision code."""
    n, p = x.shape
    result = {}
    for target in range(p):
        for parents in subsets([j for j in range(p) if j != target]):
            design = np.column_stack((np.ones(n), x[:, parents]))
            beta = np.linalg.lstsq(design, x[:, target], rcond=None)[0]
            residual = x[:, target] - np.einsum("ij,j->i", design, beta)
            variance = np.sum(residual ** 2) / n
            assert np.isfinite(variance) and variance > 0
            score = -n / 2 * (math.log(2 * math.pi) + 1 + math.log(variance))
            score -= (len(parents) + 2) / 2 * math.log(n)
            result[target, parents] = float(score)
    return result


def independent_discrete_scores(x):
    """Log-likelihood from joint-minus-parent count entropies, five states."""
    n, p = x.shape
    result = {}
    def count_entropy(columns):
        if not columns:
            return n * math.log(n)
        codes = np.ravel_multi_index(x[:, columns].T, (5,) * len(columns))
        counts = np.bincount(codes)
        positive = counts[counts > 0]
        return float(np.sum(positive * np.log(positive)))
    entropies = {tuple(s): count_entropy(tuple(s)) for s in subsets(list(range(p)))}
    for target in range(p):
        for parents in subsets([j for j in range(p) if j != target]):
            likelihood = entropies[tuple(sorted((*parents, target)))] - entropies[parents]
            result[target, parents] = likelihood - .5 * (4 * 5 ** len(parents)) * math.log(n)
    return result


def maximum_over_orders(local, p):
    """Exhaust every topological order; no subset DP from fitted engine."""
    best = -math.inf
    for order in itertools.permutations(range(p)):
        score = sum(max(local[target, tuple(sorted(ps))] for ps in subsets(order[:i]))
                    for i, target in enumerate(order))
        best = max(best, score)
    return best


def independent_pvalues(x):
    """Fisher z using raw OLS residuals for all 240 pair/conditioning queries."""
    n, p = x.shape
    result = {}
    for a, b in itertools.combinations(range(p), 2):
        for cond in subsets([i for i in range(p) if i not in (a, b)]):
            design = np.column_stack((np.ones(n), x[:, cond]))
            y = x[:, [a, b]]
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - np.einsum("ij,jk->ik", design, beta)
            r = float(np.corrcoef(residual.T)[0, 1])
            z = math.atanh(np.clip(r, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps)) * math.sqrt(n - len(cond) - 3)
            result[a, b, cond] = float(2 * norm.sf(abs(z)))
    return result


def independent_pc(pvalues, alpha=.05):
    """Independent simultaneous level-deletion implementation."""
    graph = {i: set(range(6)) - {i} for i in range(6)}
    for level in range(5):
        removals = []
        for a, b in itertools.combinations(range(6), 2):
            if b not in graph[a]:
                continue
            candidates = set()
            for endpoint, other in [(a, b), (b, a)]:
                candidates.update(itertools.combinations(sorted(graph[endpoint] - {other}), level))
            if any(pvalues[a, b, cond] > alpha for cond in candidates):
                removals.append((a, b))
        for a, b in removals:
            graph[a].remove(b)
            graph[b].remove(a)
    return tuple(sorted(tuple(sorted((NODES[a], NODES[b]))) for a in graph for b in graph[a] if a < b))


def edge_set(value):
    if not isinstance(value, str) or not value:
        return set()
    return {tuple(sorted(e.split("-"))) for e in value.split(", ")}


def saved_checks(replicates, summaries, frequencies, group_col, count_col):
    for row in replicates.itertuples():
        edges = edge_set(row.skeleton_edges)
        assert len(edges) == row.edge_count
        if "BIC" in row.method:
            dag = [tuple(s.split("->")) for s in row.one_optimal_dag.split(", ")] if row.one_optimal_dag else []
            graph = nx.DiGraph(dag)
            assert nx.is_directed_acyclic_graph(graph)
            assert {tuple(sorted(e)) for e in dag} == edges
    for (group, method), block in replicates.groupby([group_col, "method"]):
        summary = summaries.loc[summaries[group_col].eq(group) & summaries.method.eq(method)].iloc[0]
        assert len(block) == summary.fits == 1000
        assert np.isclose(block.edge_count.mean(), summary.mean_edges, rtol=0, atol=1e-12)
        assert np.isclose(block.edge_count.std(ddof=1), summary.sd_edges, rtol=0, atol=1e-12)
        lo, hi = np.quantile(block.edge_count, [.025, .975], method="nearest")
        assert lo == summary.percentile_2_5 and hi == summary.percentile_97_5
        parsed = [edge_set(v) for v in block.skeleton_edges]
        subset = frequencies.loc[frequencies[group_col].eq(group) & frequencies.method.eq(method)]
        assert len(subset) == 15
        for row in subset.itertuples():
            count = sum(tuple(row.edge.split("-")) in edges for edges in parsed)
            assert count == getattr(row, count_col)
            assert np.isclose(count / len(block), row.selection_fraction, rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-draws", action="store_true", help="Refit all 8,000 equal-N samples rather than 24 spot checks.")
    args = parser.parse_args()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(ROOT / "outputs/original_hexaco_scored_domains.csv")
    sapa = pd.read_csv(ROOT / "outputs/sapa_hexaco_scored_domains.csv")
    figure_sources = pd.read_csv(ROOT / "outputs/main_paper_cpdags/network_cpdag_results.csv").fillna("")
    categorical_sources = pd.read_csv(ROOT / "outputs/categorical_full_sample_appendix/full_sample_categorical_edges.csv").fillna("")
    sapa["age_group"] = sapa.age_band.map(COLLAPSE)
    assert len(primary) == 98917 and len(sapa) == 18616 and sapa.RID.is_unique
    assert pd.cut(primary.age, [2, 12, 19, 39, 64, 100], labels=engine.AGE_LABELS).astype(str).eq(primary.age_group).all()
    specs = [
        ("Primary", primary, engine.AGE_LABELS, "original_continuous_bn_sensitivity", "matched_n3000", 3000, 20261015),
        ("SAPA", sapa, SAPA_BANDS, "sapa_gaussian_wrapup", "matched_n300", 300, 20261020),
    ]
    full_rows, categorical_rows, draw_rows, summary_rows = [], [], [], []
    max_local_error = max_p_error = 0.
    for dataset, data, bands, folder, prefix, sample_n, seed in specs:
        base = ROOT / "outputs" / folder
        full = pd.read_csv(base / "full_sample_results.csv").fillna("")
        if dataset == "SAPA":
            full = full.loc[full.scheme.eq("pooled_four")]
        group_col = "age_group" if dataset == "Primary" else "age_band"
        for group in bands:
            values = data.loc[data.age_group.eq(group), NODES].to_numpy(float)
            assert np.isfinite(values).all() and np.all((values >= 1) & (values <= 5))
            cov, corr = engine.moments(values)
            assert np.linalg.eigvalsh(corr).min() > 0
            local = independent_local_ols(values)
            expected_best = maximum_over_orders(local, 6)
            local_actual = engine.gaussian_local_scores(cov, len(values))
            for (target, parents), score in local.items():
                error = abs(score - local_actual[target, sum(1 << j for j in parents)])
                max_local_error = max(max_local_error, error)
                assert error < 1e-7
            probabilities = independent_pvalues(values)
            for (a, b, cond), expected in probabilities.items():
                actual = engine.fisher_pvalue(engine.partial_correlation(corr, a, b, cond), len(values), len(cond))
                max_p_error = max(max_p_error, abs(expected - actual))
                assert abs(expected - actual) < 1e-10
            independent_edges = independent_pc(probabilities)
            actual_pc, actual_score, actual_dag = engine.fit_sample(values)
            assert independent_edges == actual_pc
            assert abs(expected_best - actual_score) < 1e-7
            graph = nx.DiGraph(actual_dag)
            assert nx.is_directed_acyclic_graph(graph)
            subset = full.loc[full[group_col].eq(group)]
            for row in subset.itertuples():
                actual = set(actual_pc) if "PC" in row.method else {tuple(sorted(e)) for e in actual_dag}
                assert actual == edge_set(row.skeleton_edges) and len(actual) == row.edge_count
            panel_rows = figure_sources.loc[figure_sources.dataset.eq(dataset) & figure_sources.group.eq(group)]
            assert len(panel_rows) == (0 if group == engine.AGE_LABELS[0] else 2)
            for row in panel_rows.itertuples():
                actual = set(actual_pc) if "PC" in row.method else {tuple(sorted(e)) for e in actual_dag}
                assert actual == edge_set(row.skeleton_edges) and len(actual) == row.edge_count and row.n == len(values)
            full_rows.append(dict(dataset=dataset, age_group=group, n=len(values), pc_count=len(actual_pc), bic_count=len(actual_dag),
                                  independent_ols_all_orders_best=expected_best, production_dp_score=actual_score,
                                  minimum_correlation_eigenvalue=float(np.linalg.eigvalsh(corr).min()),
                                  all_192_local_scores_match=True, all_240_ci_tests_match=True, saved_edges_match=True))
            discrete = pd.DataFrame(np.column_stack([pd.cut(values[:, j], [1, 1.8, 2.6, 3.4, 4.2, 5.0000001], labels=False, include_lowest=True) + 1 for j in range(6)]), columns=NODES).astype(int)
            independent_discrete_best = maximum_over_orders(independent_discrete_scores(discrete.to_numpy()-1), 6)
            fast_score, fast_dag = fast_exact_bic.exact_bic_dag(discrete)
            reference_score, reference_dag = hexaco_bic_exhaustive.exact_bic_dag(discrete)
            assert abs(independent_discrete_best - fast_score) < 1e-6
            assert abs(reference_score - fast_score) < 1e-8
            assert fast_dag == reference_dag
            category_rows = categorical_sources.loc[categorical_sources.dataset.eq(dataset) & categorical_sources.age_group.eq(group)
                                                    & categorical_sources.method.eq("Categorical BIC")]
            assert len(category_rows) == (0 if group == engine.AGE_LABELS[0] else 1)
            for row in category_rows.itertuples():
                assert {tuple(sorted(e)) for e in fast_dag} == edge_set(row.skeleton_edges)
            categorical_rows.append(dict(dataset=dataset, age_group=group, n=len(values), bic_edges=len(fast_dag),
                                         independent_best=independent_discrete_best, production_score=fast_score,
                                         optimized_equals_reference=True, independent_all_orders_score_matches=True))
            print(dataset, group, len(actual_pc), len(actual_dag), flush=True)
        replicates = pd.read_csv(base / f"{prefix}_replicate_results.csv").fillna("")
        summaries = pd.read_csv(base / f"{prefix}_summary.csv")
        frequency_name = "edge_selection_frequencies.csv" if dataset == "Primary" else f"{prefix}_edge_selection_frequencies.csv"
        frequencies = pd.read_csv(base / frequency_name)
        saved_checks(replicates, summaries, frequencies, group_col, "times_selected" if dataset == "Primary" else "times_retained")
        indices_col = "sample_indices_sha256" if dataset == "Primary" else "source_row_indices_sha256"
        lookup = {(getattr(row, group_col), row.replicate, row.method): row for row in replicates.itertuples()}
        matched_bands = bands[1:] if dataset == "Primary" else bands
        for group_index, group in enumerate(matched_bands, 1):
            source = data.loc[data.age_group.eq(group)]
            for rep in range(1, 1001):
                rng = np.random.default_rng(np.random.SeedSequence([seed, group_index, rep]))
                chosen = np.sort(rng.choice(source.index.to_numpy(), sample_n, replace=False))
                assert len(np.unique(chosen)) == sample_n
                digest = hashlib.sha256(chosen.astype("<i8").tobytes()).hexdigest()
                rows = [lookup[group, rep, method] for method in replicates.method.unique()]
                assert all(getattr(row, indices_col) == digest for row in rows)
                if dataset == "SAPA":
                    participant_digest = hashlib.sha256(data.loc[chosen].RID.to_numpy(dtype="<i8").tobytes()).hexdigest()
                    assert all(row.participant_ids_sha256 == participant_digest for row in rows)
                if args.all_draws or rep in [1, 500, 1000]:
                    values = data.loc[chosen, NODES].to_numpy(float)
                    actual_pc, actual_score, dag = engine.fit_sample(values)
                    for row in rows:
                        actual_edges = set(actual_pc) if "PC" in row.method else {tuple(sorted(e)) for e in dag}
                        assert edge_set(row.skeleton_edges) == actual_edges and row.edge_count == len(actual_edges)
                        if "BIC" in row.method:
                            assert abs(row.bic_score - actual_score) < 1e-7
                    draw_rows.append(dict(dataset=dataset, age_group=group, replicate=rep, sample_n=sample_n,
                                          pc_count=len(actual_pc), bic_count=len(dag), saved_membership_edges_scores_match=True))
        summary_rows.append(dict(dataset=dataset, saved_fits_checked=len(replicates), memberships_rebuilt=4000,
                                 refitted_samples=4000 if args.all_draws else 12, summaries_checked=len(summaries), edge_frequency_rows_checked=len(frequencies)))
    pd.DataFrame(full_rows).to_csv(OUT / "full_sample_independent_checks.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(OUT / "selected_draw_refits.csv", index=False)
    pd.DataFrame(categorical_rows).to_csv(OUT / "categorical_bic_independent_checks.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "saved_output_integrity.csv", index=False)
    scripts = ["original_continuous_bn_sensitivity.py", "sapa_gaussian_wrapup.py", "original_hexaco_pc_bic.py",
               "sapa_hexaco_pc_bic.py", "hexaco_bic_exhaustive.py", "fast_exact_bic.py", "fast_pc_stable.py", "audit_submission_core.py"]
    hashes = {f"analysis/{name}": hashlib.sha256((ROOT / "analysis" / name).read_bytes()).hexdigest() for name in scripts}
    result = dict(status="All scoped numerical checks passed", full_groups=9, local_ols_scores=9*192, ci_probabilities=9*240,
                  independent_six_node_orders_per_group=720, max_local_score_error=max_local_error, max_ci_probability_error=max_p_error,
                  saved_equal_n_method_rows_checked=16000, memberships_rebuilt=8000, equal_n_sample_refits=len(draw_rows),
                  categorical_bic_full_groups_independently_checked=9,
                  gaussian_main_figure_panels_matched=16, categorical_bic_figure_panels_matched=8,
                  summary_rows=16, edge_frequency_rows=240, python=platform.python_version(), numpy=np.__version__,
                  pandas=pd.__version__, scipy=scipy.__version__, script_hashes=hashes, runtime_seconds=time.monotonic()-started)
    (OUT / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "script_hashes"}, indent=2))


if __name__ == "__main__":
    main()
