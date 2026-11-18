"""Bounded SAPA measurement audit: scoring, facet coverage and item overlap.

No reliability model, invariance model, network model or new exclusion is fit.
All summaries describe the existing main analytic participants.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_external" / "sapa" / "sapa_data.tab"
KEY = ROOT / "data_external" / "sapa" / "superKey696.csv"
CACHE = ROOT / "outputs" / "sapa_hexaco_scored_domains.csv"
OUT = ROOT / "outputs" / "measurement_resolution"
DOMAINS = ["H", "E", "X", "A", "C", "O"]
FACETS = {
    "H": ["HHFairness", "HHGreedAvoid", "HHModesty", "HHSincere"],
    "E": ["HEAnxiety", "HEDepend", "HEFear", "HESentiment"],
    "X": ["HXExpress", "HXLiveliness", "HXSociable", "HXSocBold"],
    "A": ["HAFlex", "HAForgive", "HAGentle", "HAPatient"],
    "C": ["HCDiligence", "HCOrgan", "HCPerfect", "HCPrudence"],
    "O": ["HOAesthetic", "HOCreative", "HOInquisite", "HOUnconvention"],
}
AGE_MAP = {"18andUnder": "18 and under", "19to24": "19-24", "25to29": "25-29",
           "30to34": "30-34", "35to39": "35-39", "40to49": "40-49",
           "50to59": "50-59", "60andOver": "60 and over"}
NATIVE = list(AGE_MAP.values())
POOLED = ["18 and under", "19-39", "40-59", "60 and over"]
COLLAPSE = dict(zip(NATIVE, [POOLED[0]] + [POOLED[1]] * 4 + [POOLED[2]] * 2 + [POOLED[3]]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    keys = pd.read_csv(KEY).set_index("Unnamed: 0")
    assert keys.index.is_unique
    item_lists = {d: keys.index[keys["HEXACO_" + d].ne(0)].tolist() for d in DOMAINS}
    item_map = []
    for domain, facets in FACETS.items():
        assert len(item_lists[domain]) == 40
        assert keys[facets].ne(0).sum().eq(10).all()
        assert keys[facets].sum(axis=1).eq(keys["HEXACO_" + domain]).all()
        assert keys.loc[item_lists[domain], facets].ne(0).sum(axis=1).eq(1).all()
        for item in item_lists[domain]:
            facet = next(f for f in facets if keys.loc[item, f] != 0)
            sign = int(keys.loc[item, "HEXACO_" + domain])
            assert sign in [-1, 1]
            item_map.append({"domain": domain, "facet_key": facet, "item": item, "key_sign": sign})
    all_items = sorted(set(item for items in item_lists.values() for item in items))
    raw = pd.read_csv(RAW, sep="\t", usecols=["RID", "gender", "age", *all_items])
    assert raw.RID.is_unique and len(raw) == 23679
    responses = raw[all_items].apply(pd.to_numeric, errors="raise")
    nonmissing = responses.stack().to_numpy()
    assert np.isfinite(nonmissing).all() and np.isin(nonmissing, [1, 2, 3, 4, 5, 6]).all()

    rebuilt = raw[["RID", "gender", "age"]].copy()
    rebuilt["age_band"] = raw.age.map(AGE_MAP)
    observed, facet_counts = {}, {}
    for domain, items in item_lists.items():
        keyed = responses[items].copy()
        for item in items:
            if keys.loc[item, "HEXACO_" + domain] < 0:
                keyed[item] = 7 - keyed[item]
        observed[domain] = keyed.notna().to_numpy()
        rebuilt[domain + "_item_count"] = keyed.notna().sum(axis=1)
        rebuilt[domain] = 1 + (keyed.mean(axis=1, skipna=True) - 1) * 4 / 5
        facet_counts[domain] = np.column_stack([
            keyed.loc[:, keys.loc[items, facet].ne(0).to_numpy()].notna().sum(axis=1).to_numpy()
            for facet in FACETS[domain]
        ])
    count_cols = [d + "_item_count" for d in DOMAINS]
    keep = rebuilt.age_band.notna() & rebuilt[count_cols].ge(3).all(axis=1) & rebuilt[DOMAINS].notna().all(axis=1)
    selected = rebuilt.loc[keep].reset_index(drop=True)
    cached = pd.read_csv(CACHE)
    assert len(selected) == len(cached) == 18616
    pd.testing.assert_frame_equal(selected[["RID", "gender", "age", "age_band", *count_cols]],
                                  cached[["RID", "gender", "age", "age_band", *count_cols]], check_exact=True)
    maximum_score_error = float(np.max(np.abs(selected[DOMAINS].to_numpy() - cached[DOMAINS].to_numpy())))
    assert maximum_score_error <= 1e-12
    assert np.array_equal(raw.loc[keep, "RID"].to_numpy(), cached.RID.to_numpy())
    selected["pooled_band"] = selected.age_band.map(COLLAPSE)
    assert selected.groupby("pooled_band").size().reindex(POOLED).tolist() == [4550, 11510, 2253, 303]
    observed = {d: values[keep.to_numpy()] for d, values in observed.items()}
    facet_counts = {d: values[keep.to_numpy()] for d, values in facet_counts.items()}

    coverage_rows, exposure_rows, pair_rows, facet_rows, group_rows = [], [], [], [], []
    for scheme, column, bands in [("pooled_four", "pooled_band", POOLED), ("native_eight", "age_band", NATIVE)]:
        assert sum(selected[column].eq(b).sum() for b in bands) == len(selected)
        for band in bands:
            in_group = selected[column].eq(band).to_numpy()
            n = int(in_group.sum())
            base = {"scheme": scheme, "age_band": band, "n": n}
            all_facets = np.column_stack([(facet_counts[d][in_group] > 0).all(axis=1) for d in DOMAINS])
            group_rows.append({**base, "n_with_all_four_facets_in_all_six_domains": int(all_facets.all(axis=1).sum()),
                               "percent_with_all_four_facets_in_all_six_domains": float(100 * all_facets.all(axis=1).mean())})
            for domain, items in item_lists.items():
                obs = observed[domain][in_group].astype(np.int64)
                exposure = obs.sum(axis=0)
                overlap = obs.T @ obs
                ii, jj = np.triu_indices(40, 1)
                pair_n = overlap[ii, jj]
                assert len(pair_n) == 780 and np.array_equal(np.diag(overlap), exposure)
                per_person_counts = obs.sum(axis=1)
                fc = facet_counts[domain][in_group]
                represented = (fc > 0).sum(axis=1)
                assert np.array_equal(fc.sum(axis=1), per_person_counts)
                coverage_rows.append({**base, "domain": domain, "keyed_items": 40,
                                      "item_count_mean": float(per_person_counts.mean()), "item_count_median": float(np.median(per_person_counts)),
                                      "represented_facets_mean": float(represented.mean()), "represented_facets_median": float(np.median(represented)),
                                      "percent_all_four_facets": float(100 * (represented == 4).mean()),
                                      "percent_at_most_two_facets": float(100 * (represented <= 2).mean()),
                                      "complete_40_item_rows": int((per_person_counts == 40).sum()),
                                      "item_exposure_n_min": int(exposure.min()), "item_exposure_n_median": float(np.median(exposure)),
                                      "item_exposure_n_max": int(exposure.max()), "item_pair_n_min": int(pair_n.min()),
                                      "item_pair_n_median": float(np.median(pair_n)), "item_pair_n_max": int(pair_n.max()),
                                      "item_pair_n_p25": float(np.quantile(pair_n, .25)), "item_pair_n_p75": float(np.quantile(pair_n, .75)),
                                      "pairs_n_zero": int((pair_n == 0).sum()), "pairs_n_below_5": int((pair_n < 5).sum()),
                                      "pairs_n_below_10": int((pair_n < 10).sum()), "pairs_n_below_20": int((pair_n < 20).sum()),
                                      "pairs_n_below_30": int((pair_n < 30).sum()), "possible_item_pairs": 780,
                                      "unique_answered_item_subsets": int(len(np.unique(np.packbits(obs, axis=1), axis=0)))})
                for j, item in enumerate(items):
                    facet = next(f for f in FACETS[domain] if keys.loc[item, f] != 0)
                    exposure_rows.append({**base, "domain": domain, "facet_key": facet, "item": item,
                                          "answered_n": int(exposure[j]), "answered_percent": float(100 * exposure[j] / n)})
                for i, j, co_n in zip(ii, jj, pair_n):
                    pair_rows.append({**base, "domain": domain, "item_1": items[i], "item_2": items[j], "jointly_answered_n": int(co_n)})
                for j, facet in enumerate(FACETS[domain]):
                    facet_rows.append({**base, "domain": domain, "facet_key": facet, "keyed_items": 10,
                                       "answered_items_mean": float(fc[:, j].mean()), "answered_items_median": float(np.median(fc[:, j])),
                                       "at_least_one_answer_n": int((fc[:, j] >= 1).sum()),
                                       "at_least_one_answer_percent": float(100 * (fc[:, j] >= 1).mean()),
                                       "at_least_two_answers_percent": float(100 * (fc[:, j] >= 2).mean())})

    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"key_map": item_map, "coverage_by_age": coverage_rows, "item_exposure_by_age": exposure_rows,
              "item_pair_overlap_by_age": pair_rows, "facet_coverage_by_age": facet_rows,
              "all_domain_facet_coverage_by_age": group_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(OUT / f"sapa_{name}.csv", index=False)
    verification = {"raw_rows": len(raw), "analytic_rows": len(cached), "raw_distinct_relevant_items": len(all_items),
                    "keyed_items_per_domain": 40, "facets_per_domain": 4, "keyed_items_per_facet": 10,
                    "domain_items_partition_into_facets_and_signs_agree": True,
                    "all_observed_responses_are_integer_1_to_6": True,
                    "raw_reconstructed_analytic_IDs_order_demographics_ages_item_counts_match_cache": True,
                    "maximum_absolute_score_difference": maximum_score_error,
                    "pooled_age_ns": [4550, 11510, 2253, 303], "unchanged_filter": "at least3 answered items in every domain, observed age, complete domain scores",
                    "item_pair_counts_computed_without_covariance_estimation": True,
                    "models_fitted": "none: no reliability, invariance, or networks",
                    "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [RAW, KEY, CACHE, Path(__file__)]},
                    "output_rows": {name: len(rows) for name, rows in tables.items()},
                    "elapsed_seconds": time.monotonic() - started}
    (OUT / "sapa_raw_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    coverage = pd.DataFrame(coverage_rows)
    concise = coverage.loc[coverage.scheme.eq("pooled_four"), ["age_band", "domain", "n", "percent_all_four_facets", "percent_at_most_two_facets", "item_exposure_n_min", "item_exposure_n_max", "item_pair_n_median", "pairs_n_below_10", "pairs_n_below_20", "pairs_n_zero"]]
    groups = pd.DataFrame(group_rows).loc[lambda d: d.scheme.eq("pooled_four")]
    note = """SAPA measurement-resolution audit (descriptive, no model refits)

The raw item responses and local superKey696 were used to reconstruct the
EXISTING analytic sample and scores. All 18,616 participant IDs, their order,
demographics, age categories and item counts match the cache exactly. Score
differences are at floating-point precision. The >=3-item/domain inclusion
rule and all networks remain unchanged.

Each 40-item IPIP-HEXACO domain key partitions exactly into four 10-item facet
keys with identical scoring directions. A represented facet means at least
one answered item; this is only a minimal content indicator, not adequate
facet measurement. Item counts indicate observed responses; the raw file does
not distinguish an item not presented from a presented item left unanswered.
Counts therefore must not be interpreted as verified exposure or dropout.

Within each actual age band, we tabulated observed item coverage, item-pair
co-observation and the facets represented in each person's scored mean.
Zero/low co-observation counts diagnose information available for item-level
modeling; cutoffs of 5/10/20/30 are descriptive only, not established pass/fail
rules for measurement invariance. No correlations, alpha, omega or invariance
tests were estimated. Complete 40-item cases are separately counted.

""" + concise.to_string(index=False, float_format=lambda x: f"{x:.2f}") + "\n\n" + groups.to_string(index=False, float_format=lambda x: f"{x:.2f}") + """

Interpretation: reproducing scores confirms scoring correctness, not score
reliability or equivalence. The actual domain means use participant-specific
item subsets and often incomplete facet coverage. Reliability for a hypothetical
40-item composite reconstructed from pairwise covariances would not establish
reliability of those variable-subset means. Estimating their reliability would
require an explicit measurement model and missingness assumptions; a pooled
coefficient cannot substitute for age-specific measurement comparability.

The small older sample and sparse item overlaps limit a quick defensible
multi-group item-invariance analysis. This audit does not prove invariance or
noninvariance, nor establish that a more assumption-intensive model is impossible.
The justified manuscript treatment is a partial external comparison using
different items and incomplete, variable content coverage, with this unresolved
measurement limitation stated. A >=3-item threshold alone is not a validation
of full domain content, score reliability, or measurement invariance.

Reproduce: python3 analysis/sapa_measurement_resolution.py
"""
    (OUT / "sapa_measurement_note.txt").write_text(note)
    print(concise.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print(groups.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
