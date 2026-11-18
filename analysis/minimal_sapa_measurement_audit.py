"""Describe existing SAPA item coverage; no raw-item reads or model refits."""
from pathlib import Path
import hashlib

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "sapa_hexaco_scored_domains.csv"
OUT = ROOT / "outputs" / "minimal_revision_checks"
DOMAINS = ["H", "E", "X", "A", "C", "O"]
NATIVE = ["18 and under", "19-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60 and over"]
POOLED = ["18 and under", "19-39", "40-59", "60 and over"]
COLLAPSE = dict(zip(NATIVE, [POOLED[0]] + [POOLED[1]] * 4 + [POOLED[2]] * 2 + [POOLED[3]]))


def main():
    scores = pd.read_csv(SOURCE)
    assert len(scores) == 18616 and scores.RID.is_unique
    assert scores.age_band.isin(NATIVE).all()
    assert scores[DOMAINS].notna().all().all()
    counts = scores[[domain + "_item_count" for domain in DOMAINS]]
    assert counts.notna().all().all() and counts.ge(3).all().all()
    assert counts.mod(1).eq(0).all().all()
    scores["pooled_band"] = scores.age_band.map(COLLAPSE)
    pooled_ns = scores.groupby("pooled_band").size().reindex(POOLED)
    assert pooled_ns.tolist() == [4550, 11510, 2253, 303]

    rows = []
    for scheme, column, bands in [("pooled_four", "pooled_band", POOLED), ("native_eight", "age_band", NATIVE)]:
        assert sum(scores[column].eq(band).sum() for band in bands) == len(scores)
        for band in bands:
            group = scores.loc[scores[column].eq(band)]
            for domain in DOMAINS:
                coverage = group[domain + "_item_count"]
                rows.append({"scheme": scheme, "age_band": band, "domain": domain, "n": len(group),
                             "item_count_min": int(coverage.min()), "item_count_median": float(coverage.median()),
                             "item_count_mean": float(coverage.mean()), "item_count_max": int(coverage.max()),
                             "percent_exactly_3_items": float(100 * coverage.eq(3).mean()),
                             "percent_at_most_4_items": float(100 * coverage.le(4).mean()),
                             "score_mean": float(group[domain].mean()), "score_sd": float(group[domain].std(ddof=1))})
    table = pd.DataFrame(rows)
    assert len(table) == 72
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "sapa_item_coverage.csv", index=False)

    pooled = table.loc[table.scheme.eq("pooled_four")]
    concise = []
    for band in POOLED:
        sub = pooled.loc[pooled.age_band.eq(band)]
        concise.append(f"{band}: N={int(sub.n.iloc[0]):,}; across the six domains, mean answered items "
                       f"{sub.item_count_mean.min():.2f}-{sub.item_count_mean.max():.2f}; "
                       f"exactly 3 items {sub.percent_exactly_3_items.min():.1f}-{sub.percent_exactly_3_items.max():.1f}%; "
                       f"at most 4 items {sub.percent_at_most_4_items.min():.1f}-{sub.percent_at_most_4_items.max():.1f}%.")
    note = """Minimal SAPA item-coverage audit

This descriptive audit reads only outputs/sapa_hexaco_scored_domains.csv.
All 18,616 existing participants are retained. All six domains meet the original
minimum of three answered keyed items, and the four pooled age-band Ns remain
4,550 / 11,510 / 2,253 / 303. No exclusion, rescoring, raw-item access, or network
refit was performed. Endpoint age bands remain open-ended public categories.

The CSV contains 72 domain-by-age rows: four pooled bands and all eight native
bands, each for six domains. Percentages use participants within that age band
as their denominator. Score SD is the sample SD (ddof=1); scores remain the
cached 1-5 rescaled means. Item counts describe answered items, not distinct
facet coverage or whether participants answered the same item subset.

""" + "\n".join(concise) + """

This establishes item-coverage distributions only. It does not estimate score
reliability or establish measurement invariance, equivalent construct coverage,
or measurement equivalence across age bands or datasets. Differences in score
means/SDs likewise do not establish measurement invariance or its failure.

Source SHA-256: """ + hashlib.sha256(SOURCE.read_bytes()).hexdigest() + "\n"
    (OUT / "sapa_coverage_note.txt").write_text(note)
    print("\n".join(concise), flush=True)
    print("Verified: N=18,616; 72 output rows; all item counts >=3; no model refits.", flush=True)


if __name__ == "__main__":
    main()
