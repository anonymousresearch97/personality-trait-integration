"""Gaussian PC/BIC on the unchanged manuscript SAPA sample.

The four pooled labels state actual SAPA age bands. Eight native bands are
also recorded for full fits. The authorized matched comparison draws N=300
without replacement 1,000 times per pooled band. Scores and filters are fixed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import scipy

from original_continuous_bn_sensitivity import (
    DOMAINS, edge_string, exact_gaussian_bic, fit_sample, moments,
    pc_stable_gaussian, raw_ols_score, skeleton,
)
from sapa_hexaco_pc_bic import AGE_VALUE_TO_LABEL, analytic_sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sapa_gaussian_wrapup"
SOURCE = ROOT / "outputs" / "sapa_hexaco_scored_domains.csv"
ALL_ROWS = ROOT / "outputs" / "sapa_hexaco_scored_domains_all_rows.csv"
CATEGORICAL = ROOT / "outputs" / "reviewer1_comment2_age_harmonization" / "sapa_rerun_results.csv"
NATIVE = list(AGE_VALUE_TO_LABEL.values())
POOLED = ["18 and under", "19-39", "40-59", "60 and over"]
COLLAPSE = dict(zip(NATIVE, [POOLED[0]] + [POOLED[1]] * 4 + [POOLED[2]] * 2 + [POOLED[3]]))
MANUSCRIPT_LABELS = ["Adolescence 13-19", "Early adulthood 20-39", "Middle adulthood 40-64", "Older adulthood 65-100"]
MATCHED_N = 300
REPS = 1000
SEED = 20261020


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_matched(scores):
    rows, warning_rows = [], []
    for idx, band in enumerate(POOLED, 1):
        source = scores.loc[scores.pooled_band.eq(band)]
        indices = source.index.to_numpy()
        for rep in range(1, REPS + 1):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, idx, rep]))
            selected = np.sort(rng.choice(indices, MATCHED_N, replace=False))
            assert len(selected) == MATCHED_N and len(np.unique(selected)) == MATCHED_N
            sample = scores.loc[selected]
            assert sample.pooled_band.eq(band).all()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pc, score, dag = fit_sample(sample[DOMAINS].to_numpy(dtype=float))
            for warning in caught:
                warning_rows.append({"age_band": band, "replicate": rep, "message": str(warning.message)})
            base = {"dataset": "SAPA IPIP-HEXACO", "scheme": "pooled_four", "age_band": band,
                    "group_index": idx, "replicate": rep, "sample_n": MATCHED_N, "source_n": len(source),
                    "min_items_per_domain": 3,
                    "source_row_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                    "participant_ids_sha256": hashlib.sha256(sample.RID.to_numpy(dtype="<i8").tobytes()).hexdigest()}
            for method, edges in [("Gaussian PC-stable", pc), ("Exact Gaussian BIC", skeleton(dag))]:
                rows.append({**base, "method": method, "edge_count": len(edges), "skeleton_edges": edge_string(edges),
                             "alpha": .05 if method == "Gaussian PC-stable" else np.nan,
                             "bic_score": score if method == "Exact Gaussian BIC" else np.nan,
                             "one_optimal_dag": edge_string(dag, directed=True) if method == "Exact Gaussian BIC" else ""})
        print(f"SAPA {band}: {REPS} matched draws at N={MATCHED_N} complete", flush=True)
    results = pd.DataFrame(rows)
    assert len(results) == 8000
    assert results.groupby(["group_index", "replicate"]).source_row_indices_sha256.nunique().eq(1).all()
    assert results.groupby(["group_index", "method"]).size().eq(REPS).all()
    results.to_csv(OUT / "matched_n300_replicate_results.csv", index=False)
    summaries, frequencies = [], []
    all_edges = sorted("-".join(sorted(pair)) for pair in itertools.combinations(DOMAINS, 2))
    for (idx, band, method), sub in results.groupby(["group_index", "age_band", "method"], sort=False):
        counts = sub.edge_count.to_numpy()
        low, high = np.quantile(counts, [.025, .975], method="nearest")
        summaries.append({"group_index": idx, "age_band": band, "method": method,
                          "sample_n": MATCHED_N, "source_n": int(sub.source_n.iloc[0]), "fits": len(sub),
                          "mean_edges": counts.mean(), "sd_edges": counts.std(ddof=1),
                          "median_edges": np.median(counts), "percentile_2_5": low, "percentile_97_5": high,
                          "min_edges": counts.min(), "max_edges": counts.max(), "empty_fraction": np.mean(counts == 0),
                          "monte_carlo_se_of_mean": counts.std(ddof=1) / np.sqrt(len(counts))})
        edge_sets = [set(v.split(", ")) if v else set() for v in sub.skeleton_edges]
        for edge in all_edges:
            retained = sum(edge in edges for edges in edge_sets)
            frequencies.append({"group_index": idx, "age_band": band, "method": method, "edge": edge,
                                "fits": len(sub), "times_retained": retained, "selection_fraction": retained / len(sub)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "matched_n300_summary.csv", index=False)
    pd.DataFrame(frequencies).to_csv(OUT / "matched_n300_edge_selection_frequencies.csv", index=False)
    pd.DataFrame(warning_rows, columns=["age_band", "replicate", "message"]).to_csv(OUT / "matched_n300_captured_warnings.csv", index=False)
    print(summary[["age_band", "method", "sample_n", "fits", "mean_edges"]].to_string(index=False), flush=True)
    return summary, {"method_rows": len(results), "samples": 4000, "fits_per_group_per_method": REPS,
                     "within_sample_distinct_participants": MATCHED_N, "same_membership_both_methods": True,
                     "warning_count": len(warning_rows),
                     "distinct_sample_hashes_by_group": results.loc[results.method.eq("Gaussian PC-stable")].groupby("group_index").source_row_indices_sha256.nunique().to_dict()}


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(SOURCE)
    all_rows = pd.read_csv(ALL_ROWS)
    reconstructed = analytic_sample(all_rows, 3).reset_index(drop=True)
    pd.testing.assert_frame_equal(scores, reconstructed, check_exact=True)
    assert len(scores) == 18616 and scores.RID.is_unique
    assert scores.age.map(AGE_VALUE_TO_LABEL).eq(scores.age_band).all()
    assert scores[DOMAINS].notna().all().all()
    assert scores[DOMAINS].ge(1).all().all() and scores[DOMAINS].le(5).all().all()
    assert scores[[d + "_item_count" for d in DOMAINS]].ge(3).all().all()
    scores["pooled_band"] = scores.age_band.map(COLLAPSE)
    assert scores.pooled_band.notna().all()
    assert scores.groupby("pooled_band").size().reindex(POOLED).tolist() == [4550, 11510, 2253, 303]
    categorical = pd.read_csv(CATEGORICAL)
    manuscript_pc = pd.read_csv(ROOT / "outputs" / "six_png_results_tables" / "sapa_pc_stable_manuscript_age_groups.csv").set_index("age_group")
    manuscript_bic = pd.read_csv(ROOT / "outputs" / "six_png_results_tables" / "sapa_bic_manuscript_age_groups.csv").set_index("age_group")
    rows, summary, validation_rows, warning_rows = [], [], [], []
    cache = {}
    for scheme, column, bands in [("pooled_four", "pooled_band", POOLED), ("native_eight", "age_band", NATIVE)]:
        assert sum(scores[column].eq(band).sum() for band in bands) == len(scores)
        for band in bands:
            group = scores.loc[scores[column].eq(band)]
            values = group[DOMAINS].to_numpy(dtype=float)
            n = len(group)
            key = tuple(group.index)
            reference = categorical.loc[categorical.scheme.eq(scheme) & categorical.comparison_band.eq(band)].iloc[0]
            assert reference.n == n
            if scheme == "pooled_four":
                old_label = MANUSCRIPT_LABELS[POOLED.index(band)]
                assert manuscript_pc.loc[old_label, "n"] == manuscript_bic.loc[old_label, "n"] == n
                assert manuscript_pc.loc[old_label, "edge_count"] == reference.pc_edges
                assert manuscript_bic.loc[old_label, "skeleton_edge_count"] == reference.bic_edges
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if key not in cache:
                    pc, bic_score, dag = fit_sample(values)
                    cov, corr = moments(values)
                    centered = values - values.mean(axis=0)
                    direct_cov = np.einsum("ni,nj->ij", centered, centered) / n
                    assert np.allclose(cov, direct_cov, rtol=0, atol=1e-12)
                    direct_sd = np.sqrt(np.diag(direct_cov))
                    direct_corr = direct_cov / np.outer(direct_sd, direct_sd)
                    assert pc == pc_stable_gaussian(direct_corr, n)
                    explicit_score, explicit_dag = exact_gaussian_bic(direct_cov, n)
                    assert skeleton(dag) == skeleton(explicit_dag)
                    assert np.isclose(bic_score, explicit_score, rtol=0, atol=1e-8)
                    reverse = pc_stable_gaussian(corr[::-1, ::-1], n, nodes=DOMAINS[::-1])
                    assert pc == reverse
                    child_scores = []
                    for child in DOMAINS:
                        parents = [DOMAINS.index(a) for a, b in dag if b == child]
                        child_scores.append(raw_ols_score(values, DOMAINS.index(child), parents))
                    ols_score = sum(child_scores)
                    assert np.isclose(bic_score, ols_score, rtol=0, atol=1e-8)
                    cache[key] = pc, bic_score, dag, float(np.linalg.eigvalsh(corr).min()), float(abs(bic_score - ols_score)), float(np.max(abs(cov - direct_cov)))
            pc, bic_score, dag, min_eig, ols_difference, cov_difference = cache[key]
            for warning in caught:
                warning_rows.append({"scheme": scheme, "age_band": band, "message": str(warning.message)})
            base = {"dataset": "SAPA IPIP-HEXACO", "scheme": scheme, "age_band": band, "n": n,
                    "min_items_per_domain": 3, "source_row_indices_sha256": hashlib.sha256(group.index.to_numpy().astype("<i8").tobytes()).hexdigest(),
                    "participant_ids_sha256": hashlib.sha256(group.RID.to_numpy(dtype="<i8").tobytes()).hexdigest()}
            for method, edges in [("Gaussian PC-stable", pc), ("Exact Gaussian BIC", skeleton(dag))]:
                rows.append({**base, "method": method, "edge_count": len(edges), "skeleton_edges": edge_string(edges),
                             "alpha": .05 if method == "Gaussian PC-stable" else np.nan,
                             "bic_score": bic_score if method == "Exact Gaussian BIC" else np.nan,
                             "one_optimal_dag": edge_string(dag, directed=True) if method == "Exact Gaussian BIC" else ""})
            summary.append({**base, "gaussian_pc_edges": len(pc), "gaussian_bic_edges": len(dag),
                            "categorical_pc_edges": int(reference.pc_edges), "categorical_bic_edges": int(reference.bic_edges)})
            validation_rows.append({"scheme": scheme, "age_band": band, "n": n, "minimum_correlation_eigenvalue": min_eig,
                                    "absolute_bic_ols_difference": ols_difference, "maximum_covariance_sum_difference": cov_difference})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "full_sample_results.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "group_validation.csv", index=False)
    pd.DataFrame(warning_rows, columns=["scheme", "age_band", "message"]).to_csv(OUT / "captured_warnings.csv", index=False)
    matched_summary, matched_validation = run_matched(scores)
    validation = {"analytic_cache_equals_three_item_filtered_all_rows_exactly": True,
                  "unique_participants": len(scores), "complete_six_domain_scores": True,
                  "all_12_group_sizes_equal_categorical_reference": True,
                  "all_four_pooled_N_and_categorical_counts_equal_manuscript_tables": True,
                  "explicit_covariance_same_pc_and_bic_skeleton": True,
                  "selected_bic_score_matches_raw_ols": True,
                  "pc_node_order_invariance": True, "unique_group_fits": len(cache),
                  "recorded_method_rows": len(results), "warning_count": len(warning_rows),
                  "matched_n300": matched_validation}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    provenance = {"analysis": "Full-sample SAPA Gaussian PC and exact Gaussian BIC on unchanged manuscript participants",
                  "sources": {str(p.relative_to(ROOT)): sha(p) for p in [SOURCE, ALL_ROWS, CATEGORICAL]},
                  "scripts": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / "analysis" / "original_continuous_bn_sensitivity.py", ROOT / "analysis" / "sapa_hexaco_pc_bic.py"]},
                  "pooled_bands": POOLED, "native_bands": NATIVE, "collapse_map": COLLAPSE,
                  "filter": "observed native age; at least 3 answered keyed items in each of all 6 domains; complete scores",
                  "scores": "unchanged cached keyed domain means, originally affine rescaled from 1-6 to 1-5",
                  "PC": "PC-stable, Gaussian partial-correlation Fisher-z test, alpha .05",
                  "BIC": "exact DAG optimization: log-likelihood minus half parameter count times log(n); intercept, slopes, residual variance per node",
                  "reported_networks": "undirected skeletons; one optimal BIC DAG recorded without causal interpretation",
                  "resampling": {"full_sample": "none", "matched_n": MATCHED_N, "draws_per_pooled_band": REPS,
                                 "seed": SEED, "seed_rule": "SeedSequence([20261020, pooled_group_index_1_to_4, replicate_1_to_1000])",
                                 "within_draw": "without replacement, sorted source CSV row indices",
                                 "all_four_groups_resampled": True, "across_draws": "overlap allowed"},
                  "python": platform.python_version(), "numpy": np.__version__,
                  "pandas": pd.__version__, "scipy": scipy.__version__, "runtime_seconds": time.monotonic() - started}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = """SAPA full-sample Gaussian comparison

All 18,616 participants are exactly those in the saved manuscript analytic
sample, requiring at least three answered keyed items per domain. Refiltering
the saved all-row scores reproduces that analytic cache exactly. Domain means
are used without discretization, rescoring, or additional exclusion.

Four pooled actual SAPA bands are <=18, 19-39, 40-59 and >=60. These are not
the original primary dataset's developmental age boundaries. A separate SAPA
childhood estimate is unavailable and must not be displayed as a zero-edge fit.
The native eight bands are included for records only. Endpoint fits are shared.

PC uses the existing validated Gaussian PC-stable function at alpha .05;
BIC uses the existing exact Gaussian DAG optimizer. Both summarize skeleton
edge counts. These are full-sample estimates with different sample sizes;
they do not establish age-group differences independently of sample size.
The separate equal-N comparison uses 1,000 samples of 300 distinct participants
from each pooled band, including random samples from the 303 older participants.
PC and BIC receive the same participants for every draw. Draws can overlap;
their empirical percentile ranges are not population confidence intervals.
Equal N is enforced within SAPA; this N=300 is not the primary study's N=3,000.
Gaussian assumptions remain, and numerical checks do not validate those
substantive modeling assumptions. BIC values are not compared across models
with differently transformed responses or discrete versus continuous scores.

""" + pd.DataFrame(summary).drop(columns=["source_row_indices_sha256", "participant_ids_sha256"]).to_string(index=False) + "\n\nMatched N=300\n" + matched_summary.to_string(index=False) + "\n"
    (OUT / "methods_results.txt").write_text(report)
    print(pd.DataFrame(summary)[["scheme", "age_band", "n", "gaussian_pc_edges", "gaussian_bic_edges", "categorical_pc_edges", "categorical_bic_edges"]].to_string(index=False), flush=True)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
