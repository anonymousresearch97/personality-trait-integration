"""Complete cross-sectional descriptive tables requested in reviewer 1 minor 8.

Run from any directory with Python, pandas, numpy and pyreadstat installed.
No network fitting, participant sampling, imputation, or manuscript editing.
Raw-data reconstructions are checked against the frozen analysis caches.
Childhood is reported separately from the four-group primary main sample.
"""
from __future__ import annotations

import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/submission_code_audit/descriptives"
DOMAINS = ["H", "E", "X", "A", "C", "O"]
NAMES = dict(zip(DOMAINS, ["Honesty-Humility", "Emotionality", "Extraversion",
                         "Agreeableness", "Conscientiousness", "Openness"]))
SCORE_COLS = dict(zip(DOMAINS, ["hones16", "emoti16", "extra16", "agree16", "consc16", "openn16"]))
PRIMARY_GROUPS = ["Childhood 3-12", "Adolescence 13-19", "Early adulthood 20-39",
                  "Middle adulthood 40-64", "Older adulthood 65-100"]
AGE_MAP = {"18andUnder": "18 and under", "19to24": "19-24", "25to29": "25-29",
           "30to34": "30-34", "35to39": "35-39", "40to49": "40-49",
           "50to59": "50-59", "60andOver": "60 and over"}
POOLED_MAP = {"18 and under": "18 and under", "19-24": "19-39", "25-29": "19-39",
              "30-34": "19-39", "35-39": "19-39", "40-49": "40-59",
              "50-59": "40-59", "60 and over": "60 and over"}
SAPA_GROUPS = ["18 and under", "19-39", "40-59", "60 and over"]
FILES = {
    "primary_raw": ROOT / "Lee & Ashton 2018  online sample data.sav",
    "primary_cache": ROOT / "outputs/original_hexaco_scored_domains.csv",
    "sapa_raw": ROOT / "data_external/sapa/sapa_data.tab",
    "sapa_key": ROOT / "data_external/sapa/superKey696.csv",
    "sapa_cache": ROOT / "outputs/sapa_hexaco_scored_domains.csv",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe(x):
    x = pd.Series(x).dropna().astype(float)
    return {"n_observed": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
            "median": x.median(), "minimum": x.min(), "q1": x.quantile(.25),
            "q3": x.quantile(.75), "maximum": x.max()}


def save(rows, name):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False)
    return df


def primary():
    raw, meta = pyreadstat.read_sav(str(FILES["primary_raw"]), apply_value_formats=False)
    item_map = {d: [c for c in meta.column_names[2:102]
                    if c.startswith(d) and not c.startswith("Alt")] for d in DOMAINS}
    assert all(len(items) == 16 for items in item_map.values())
    errors = {}
    for d, items in item_map.items():
        x = raw[items]
        assert x.notna().all().all() and x.ge(1).all().all() and x.le(5).all().all()
        errors[d] = float((x.mean(axis=1) - raw[SCORE_COLS[d]]).abs().max())
        assert errors[d] == 0
    scored = raw.rename(columns={"user_age": "age", **{v: k for k, v in SCORE_COLS.items()}}).copy()
    valid = scored.loc[scored.age.between(3, 100)].copy()
    valid["age_group"] = pd.cut(valid.age, [2, 12, 19, 39, 64, 100], labels=PRIMARY_GROUPS).astype(str)
    cached = pd.read_csv(FILES["primary_cache"])
    pd.testing.assert_frame_equal(valid[["age", *DOMAINS, "age_group"]].reset_index(drop=True),
                                  cached, check_exact=True)
    labels = meta.variable_value_labels.get("sexnumeric", {})
    assert labels == {1.0: "men", 2.0: "women"}
    valid["sex_gender_label"] = valid.sexnumeric.map(labels).fillna("Missing")
    assert not ((valid.sexnumeric.notna()) & (~valid.sexnumeric.isin(labels))).any()
    return raw, valid, labels, item_map, errors


def sapa():
    key = pd.read_csv(FILES["sapa_key"])
    item_col = key.columns[0]
    keys = {}
    for d in DOMAINS:
        signs = pd.to_numeric(key[f"HEXACO_{d}"], errors="coerce").fillna(0).astype(int)
        assert signs.isin([-1, 0, 1]).all()
        keys[d] = dict(zip(key.loc[signs.ne(0), item_col], signs[signs.ne(0)]))
        assert len(keys[d]) == 40
    all_items = sorted(set().union(*(set(k) for k in keys.values())))
    raw = pd.read_csv(FILES["sapa_raw"], sep="\t", usecols=["RID", "gender", "age", *all_items])
    assert raw.RID.is_unique
    values = raw[all_items]
    assert (values.isna() | (values.ge(1) & values.le(6))).all().all()
    scored = raw[["RID", "gender", "age"]].copy()
    scored["age_band"] = scored.age.map(AGE_MAP)
    assert not (scored.age.notna() & scored.age_band.isna()).any()
    for d, signs in keys.items():
        keyed = pd.DataFrame({col: raw[col] if sign == 1 else 7 - raw[col]
                              for col, sign in signs.items()})
        scored[f"{d}_item_count"] = keyed.notna().sum(axis=1)
        scored[d] = 1 + (keyed.mean(axis=1) - 1) * 4 / 5
    count_cols = [f"{d}_item_count" for d in DOMAINS]
    mask = scored.age_band.notna() & scored[count_cols].ge(3).all(axis=1)
    valid = scored.loc[mask].dropna(subset=DOMAINS).copy()
    cached = pd.read_csv(FILES["sapa_cache"])
    assert np.array_equal(valid.RID.to_numpy(), cached.RID.to_numpy())
    errors = {}
    for d in DOMAINS:
        assert np.array_equal(valid[f"{d}_item_count"], cached[f"{d}_item_count"])
        errors[d] = float(np.max(np.abs(valid[d].to_numpy() - cached[d].to_numpy())))
        np.testing.assert_allclose(valid[d], cached[d], rtol=0, atol=1e-14)
    for col in ["gender", "age", "age_band"]:
        assert valid[col].reset_index(drop=True).equals(cached[col])
    valid["age_group"] = valid.age_band.map(POOLED_MAP)
    valid["sex_gender_label"] = valid.gender.fillna("Missing")
    return raw, scored, valid, keys, errors


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    praw, pall, sex_labels, pitems, perrors = primary()
    sraw, sall, svalid, skeys, serrors = sapa()
    pvalid = pall.loc[pall.age.ge(13)].copy()
    child = pall.loc[pall.age.lt(13)].copy()
    assert len(praw) == 100318 and len(pall) == 98917 and len(pvalid) == 98819 and len(child) == 98
    assert len(sraw) == 23679 and len(svalid) == 18616

    age_rows, age_counts = [], []
    child_age_rows, child_age_counts = [], []
    for label in PRIMARY_GROUPS:
        group = pall.loc[pall.age_group.eq(label)]
        age_row = {"age_group": label, **describe(group.age)}
        counts = group.age.value_counts().sort_index()
        count_rows = [{"age_group": label, "age": float(age), "n": int(n),
                       "percent_within_group": 100 * n / len(group)} for age, n in counts.items()]
        if label == PRIMARY_GROUPS[0]:
            child_age_rows.append(age_row)
            child_age_counts.extend(count_rows)
        else:
            age_rows.append(age_row)
            age_counts.extend(count_rows)
    age_rows.append({"age_group": "All primary ages 13-100", **describe(pvalid.age)})
    save(age_rows, "primary_age_summary.csv")
    save(age_counts, "primary_age_distribution.csv")
    save(child_age_rows, "childhood_age_summary.csv")
    save(child_age_counts, "childhood_age_distribution.csv")

    domain_rows, composition = [], []
    child_domains, child_sex = [], []
    missing = []
    for dataset, valid, labels, raw, source_var in [
        ("Primary", pvalid, PRIMARY_GROUPS[1:], praw, "sexnumeric"),
        ("SAPA", svalid, SAPA_GROUPS, sraw, "gender"),
        ("Primary childhood (separate archive)", child, PRIMARY_GROUPS[:1], praw, "sexnumeric")
    ]:
        extra = [] if dataset.endswith("archive)") else ["All analytic participants"]
        for label in labels + extra:
            group = valid if label == "All analytic participants" else valid.loc[valid.age_group.eq(label)]
            for d in DOMAINS:
                row = {"dataset": dataset, "age_group": label, "domain": d,
                       "domain_name": NAMES[d], "n_group": len(group),
                       "missing_scores": int(group[d].isna().sum()), **describe(group[d])}
                (child_domains if dataset.endswith("archive)") else domain_rows).append(row)
            # Include zero missing counts explicitly; use the entire age-group N as denominator.
            known = ["men", "women", "Missing"] if source_var == "sexnumeric" else ["female", "male", "Missing"]
            counts = group.sex_gender_label.value_counts(dropna=False)
            assert set(counts.index).issubset(known)
            for label_value in known:
                n = int(counts.get(label_value, 0))
                row = {"dataset": dataset, "age_group": label, "source_variable": source_var,
                       "category": label_value, "n": n, "n_group": len(group),
                       "percent_within_group": 100 * n / len(group)}
                (child_sex if dataset.endswith("archive)") else composition).append(row)
        if not dataset.endswith("archive)"):
            columns = ["user_age", "sexnumeric", *SCORE_COLS.values()] if dataset == "Primary" else ["age", "gender"]
            for col in columns:
                missing.append({"dataset": dataset, "sample": "Raw source", "variable": col,
                                "n": len(raw), "n_missing": int(raw[col].isna().sum())})
            for col in [*DOMAINS, source_var]:
                missing.append({"dataset": dataset, "sample": "Analytic main sample", "variable": col,
                                "n": len(valid), "n_missing": int(valid[col].isna().sum())})
    save(domain_rows, "domain_descriptives.csv")
    wide = pd.DataFrame(domain_rows)
    wide = wide.loc[wide.age_group.ne("All analytic participants")].copy()
    wide["mean_sd"] = [f"{m:.3f} ({s:.3f})" for m, s in zip(wide["mean"], wide.sd)]
    wide = wide.pivot(index=["dataset", "age_group", "n_group"], columns="domain", values="mean_sd")
    desired_index = [("Primary", g, int(pvalid.age_group.eq(g).sum())) for g in PRIMARY_GROUPS[1:]]
    desired_index += [("SAPA", g, int(svalid.age_group.eq(g).sum())) for g in SAPA_GROUPS]
    wide = wide.reindex(pd.MultiIndex.from_tuples(desired_index, names=wide.index.names))[DOMAINS].reset_index()
    save(wide, "domain_mean_sd_wide.csv")
    save(composition, "sex_gender_composition.csv")
    save(child_domains, "childhood_domain_descriptives.csv")
    save(child_sex, "childhood_sex_composition.csv")
    save(missing, "missingness_summary.csv")

    native_rows = []
    for native in AGE_MAP.values():
        pooled = POOLED_MAP[native]
        n = int(svalid.age_band.eq(native).sum())
        pooled_n = int(svalid.age_group.eq(pooled).sum())
        native_rows.append({"pooled_age_band": pooled, "constituent_public_age_band": native,
                            "n": n, "pooled_n": pooled_n, "percent_within_pooled_band": 100 * n / pooled_n,
                            "percent_of_sapa_analytic": 100 * n / len(svalid)})
    save(native_rows, "sapa_constituent_age_bands.csv")
    item_rows = []
    for grouping, labels in [("age_group", SAPA_GROUPS), ("age_band", list(AGE_MAP.values()))]:
        for label in labels:
            group = svalid.loc[svalid[grouping].eq(label)]
            for d in DOMAINS:
                counts = group[f"{d}_item_count"]
                item_rows.append({"dataset": "SAPA", "grouping": grouping, "age_group": label,
                                  "domain": d, "available_keyed_items": len(skeys[d]),
                                  "n_group": len(group), **describe(counts),
                                  "n_exactly_three_answered": int(counts.eq(3).sum()),
                                  "percent_exactly_three_answered": 100 * counts.eq(3).mean(),
                                  "mean_percent_of_keyed_items_answered": 100 * counts.mean() / len(skeys[d])})
    save(item_rows, "sapa_item_coverage.csv")

    missing_age = int(praw.user_age.isna().sum())
    outside = int((praw.user_age.notna() & ~praw.user_age.between(3, 100)).sum())
    s_age_ok = sall.age_band.notna()
    s_item_ok = sall[[f"{d}_item_count" for d in DOMAINS]].ge(3).all(axis=1)
    s_complete = sall[DOMAINS].notna().all(axis=1)
    flow = [
        {"dataset": "Primary", "stage": "Source", "n_retained": len(praw), "n_removed_at_stage": 0},
        {"dataset": "Primary", "stage": "Nonmissing age", "n_retained": len(praw)-missing_age, "n_removed_at_stage": missing_age},
        {"dataset": "Primary", "stage": "Age 3-100 (historical analytic sample)", "n_retained": len(pall), "n_removed_at_stage": outside},
        {"dataset": "Primary", "stage": "Age 13-100 (main sample; childhood archived separately)", "n_retained": len(pvalid), "n_removed_at_stage": len(child)},
        {"dataset": "SAPA", "stage": "Source", "n_retained": len(sraw), "n_removed_at_stage": 0},
        {"dataset": "SAPA", "stage": "Recognized nonmissing public age band", "n_retained": int(s_age_ok.sum()), "n_removed_at_stage": int((~s_age_ok).sum())},
        {"dataset": "SAPA", "stage": "At least 3 answered keyed items in every domain", "n_retained": int((s_age_ok & s_item_ok).sum()), "n_removed_at_stage": int((s_age_ok & ~s_item_ok).sum())},
        {"dataset": "SAPA", "stage": "Nonmissing six domain scores (final analytic sample)", "n_retained": len(svalid), "n_removed_at_stage": int((s_age_ok & s_item_ok & ~s_complete).sum())},
    ]
    save(flow, "sample_flow.csv")
    assert missing_age == 1373 and outside == 28
    for dataset, groups, expected in [(pvalid, PRIMARY_GROUPS[1:], [7738, 51670, 35496, 3915]),
                                      (svalid, SAPA_GROUPS, [4550, 11510, 2253, 303])]:
        assert [int(dataset.age_group.eq(g).sum()) for g in groups] == expected
    for rows in [composition, child_sex]:
        df = pd.DataFrame(rows)
        sums = df.groupby(["dataset", "age_group"]).agg(n_sum=("n", "sum"), n_group=("n_group", "first"), percent_sum=("percent_within_group", "sum"))
        assert (sums.n_sum == sums.n_group).all()
        np.testing.assert_allclose(sums.percent_sum, 100, atol=1e-12)
    provenance = {
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "pyreadstat": pyreadstat.__version__, "script_sha256": sha256(Path(__file__)),
        "inputs": {k: {"path_relative_to_project": str(p.relative_to(ROOT)), "sha256": sha256(p)} for k,p in FILES.items()},
        "primary_sex_labels_from_sav_metadata": sex_labels,
        "primary_item_score_max_abs_errors": perrors, "sapa_score_max_abs_errors_vs_cache": serrors,
        "primary_cache_exactly_reproduced": True, "sapa_RID_order_and_item_counts_exactly_reproduced": True,
        "group_sample_sizes_reproduced": True, "composition_counts_and_percentages_checked": True,
        "primary_source_n": len(praw), "primary_main_n": len(pvalid), "childhood_n": len(child),
        "sapa_source_n": len(sraw), "sapa_analytic_n": len(svalid),
        "sample_sd_ddof": 1, "primary_items_per_domain": {d: len(v) for d,v in pitems.items()},
        "sapa_keyed_items_per_domain": {d: len(v) for d,v in skeys.items()},
        "primary_keyed_items_missing": 0,
        "raw_data_redistribution_permission_verified": False,
    }
    (OUT / "validation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    note = """Descriptive-statistics completion, 12 November 2026

Reviewer 1 minor 8 requested primary mean/SD/median/range/age distributions,
SAPA constituent age-band distributions, and sex/gender and domain means/SDs
by age group. All numerical tables for those requests are now supplied here.
These tables have not been inserted into a manuscript or Google document.

Primary main sample: ages 13-19 N=7,738; 20-39 N=51,670; 40-64 N=35,496;
65-100 N=3,915; total N=98,819. Childhood N=98 is in separate childhood_* files.
The source has 100,318 records: 1,373 missing age and 28 ages outside 3-100.
Source domain scores exactly equal means of the 16 stored keyed items per
domain in all 100,318 records. Those 96 items and six source scores are complete.
No additional reverse scoring or imputation was used. File completeness
cannot establish whether any imputation occurred before data release.

Primary sexnumeric labels are verified directly from SPSS SAV metadata:
1=men, 2=women. These replace the unlabeled codes in the older audit table.
SAPA retains source gender labels female/male. Percentages use every participant
in the specified age group as denominator; missing counts are explicit.

SAPA main sample: 18 and under N=4,550; 19-39 N=11,510; 40-59 N=2,253;
60 and over N=303; total N=18,616. Raw 1-6 responses were independently keyed,
averaged within each 40-item domain key, and mapped linearly to 1-5. At least
three answered keyed items in every domain and an observed public age band
were required. Raw reconstruction reproduced participant IDs, item counts,
and domain scores (maximum numerical discrepancy below 1e-14) in the cache.
SAPA means/SDs here are on that analyzed 1-5 scale. Item coverage is reported
separately; sparse item availability is not evidence of equivalent reliability.

Exact SAPA individual ages, mean age, median age, SD age, or observed age
range cannot be recovered from these public categories. No category midpoints
are substituted for ages. SAPA 18-and-under may include children; age 19 and
ages 60-64 are grouped differently than in the primary sample. The endpoint
bands are open-ended. The two sets of developmental categories are non-equivalent.

Contents:
primary_age_summary.csv: age means, sample SDs, medians, observed minima/maxima,
  quartiles and Ns for the four primary main groups and their combined sample.
primary_age_distribution.csv: single-year observed age frequencies by main group.
sapa_constituent_age_bands.csv: all eight constituent public age-band counts,
  with percentages within the pooled band and within the SAPA analytic sample.
domain_descriptives.csv: six domain means, sample SDs, medians, quartiles,
  observed ranges and missingness for each dataset/group and overall main sample.
domain_mean_sd_wide.csv: compact eight-row presentation of mean (SD) by group.
sex_gender_composition.csv: verified source labels, counts and percentages.
childhood_*.csv: all corresponding original childhood descriptions separately.
sapa_item_coverage.csv: answered-item counts across the four pooled/eight native bands.
sample_flow.csv and missingness_summary.csv: exclusions and missingness checks.
validation.json: versions, source hashes and reconstruction checks.

Language: Python (pandas/numpy/pyreadstat); no R or statistical network fitting
was needed for these descriptive tables. SDs are sample SDs (denominator N-1).

Data-deposition note: no explicit raw-data redistribution license was verified
from the local materials used in this audit. Package the code and aggregate
tables, and identify the original data sources; do not claim a public code
deposit, DOI, repository URL, or permission to redistribute participant-level
raw/cached data merely because this local package exists.
"""
    (OUT / "README.txt").write_text(note)
    sections = ["primary_age_summary.csv", "sapa_constituent_age_bands.csv", "sex_gender_composition.csv", "domain_mean_sd_wide.csv", "childhood_age_summary.csv", "childhood_age_distribution.csv", "sample_flow.csv"]
    body = ["<!doctype html><html><head><meta charset='utf-8'><title>Verified descriptive statistics</title><style>body{font:15px Arial,sans-serif;margin:32px;color:#17242d}table{border-collapse:collapse;font-size:13px;margin-bottom:32px}td,th{padding:6px 10px;border:1px solid #ccd5dc;text-align:right}th{background:#eef3f7}pre{white-space:pre-wrap;max-width:1000px}</style></head><body>",
            "<h1>Verified descriptive statistics</h1><pre>" + html.escape(note) + "</pre>"]
    for filename in sections:
        body.extend(["<h2>" + html.escape(filename) + "</h2>", pd.read_csv(OUT / filename).to_html(index=False, float_format=lambda x: f"{x:.3f}")])
    body.append("</body></html>")
    (OUT / "descriptive_tables.html").write_text("\n".join(body))
    print("Descriptive tables complete. Primary N=98,819 + childhood archive N=98; SAPA N=18,616.")
    print(pd.DataFrame(age_rows).to_string(index=False))
    print(pd.DataFrame(native_rows).to_string(index=False))
    print("Independent raw scoring and analysis-cache/sample-size/composition validations passed.")


if __name__ == "__main__":
    main()
