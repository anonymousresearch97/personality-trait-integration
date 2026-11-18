"""Reviewer 1, comment 2: rerun existing models with SAPA-compatible age bands.

Uses the saved analytic domain scores used by the existing figure pipeline.
Only the age grouping changes; scoring, exclusions and model implementations
are preserved. Existing manuscript outputs are not overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from sapa_hexaco_pc_bic import (
    AGE_VALUE_TO_LABEL, DOMAINS, discretize_equal_width, pc_stable_edges,
)
from hexaco_bic_exhaustive import exact_bic_dag, skeleton

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reviewer1_comment2_age_harmonization"
NATIVE = list(AGE_VALUE_TO_LABEL.values())
POOLED = ["18 and under", "19-39", "40-59", "60 and over"]
COLLAPSE = dict(zip(NATIVE, [POOLED[0]] + [POOLED[1]] * 4 + [POOLED[2]] * 2 + [POOLED[3]]))
PRIMARY_RANGES = dict(zip(NATIVE, ["3-18", "19-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60-100"]))
PRIMARY_RANGES.update({"19-39": "19-39", "40-59": "40-59"})
SOURCE_FILES = {
    "primary": ROOT / "outputs" / "original_hexaco_scored_domains.csv",
    "sapa": ROOT / "outputs" / "sapa_hexaco_scored_domains.csv",
}


def primary_bands(ages: pd.Series) -> pd.Series:
    return pd.cut(ages, [-np.inf, 18, 24, 29, 34, 39, 49, 59, np.inf], labels=NATIVE)


def load_sample(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(SOURCE_FILES[dataset])
    assert df[DOMAINS].notna().all().all()
    assert df[DOMAINS].ge(1).all().all() and df[DOMAINS].le(5).all().all()
    if dataset == "primary":
        assert df.age.between(3, 100).all() and (df.age % 1 == 0).all()
        df["native_band"] = primary_bands(df.age).astype(str)
        probes = pd.Series([3, 12, 13, 18, 19, 24, 25, 29, 30, 34, 35, 39, 40, 49, 50, 59, 60, 64, 65, 100])
        expected = [NATIVE[0]] * 4 + [NATIVE[1]] * 2 + [NATIVE[2]] * 2 + [NATIVE[3]] * 2 + [NATIVE[4]] * 2 + [NATIVE[5]] * 2 + [NATIVE[6]] * 2 + [NATIVE[7]] * 4
        assert primary_bands(probes).astype(str).tolist() == expected
        transitions = pd.crosstab(df.age_group, df.native_band.map(COLLAPSE)).reindex(columns=POOLED, fill_value=0)
        transitions.to_csv(OUT / "primary_original_to_harmonized_counts.csv")
        df.groupby("age").size().rename("n").to_csv(OUT / "primary_exact_age_counts.csv")
    else:
        assert df.RID.is_unique
        assert df[[f"{d}_item_count" for d in DOMAINS]].ge(3).all().all()
        assert df.age.map(AGE_VALUE_TO_LABEL).eq(df.age_band).all()
        df["native_band"] = df.age_band
    df["pooled_band"] = df.native_band.map(COLLAPSE)
    assert df.native_band.isin(NATIVE).all() and df.pooled_band.notna().all()
    return df


def run_dataset(dataset: str) -> None:
    df = load_sample(dataset)
    rows = []
    cache = {}
    for scheme, col, bands in [("pooled_four", "pooled_band", POOLED), ("native_eight", "native_band", NATIVE)]:
        assert sum(int(df[col].eq(b).sum()) for b in bands) == len(df)
        for band in bands:
            group = df.loc[df[col].eq(band)]
            key = tuple(group.index)
            print(f"{dataset}: {scheme}, {band}, n={len(group):,}", flush=True)
            if key not in cache:
                discrete = discretize_equal_width(group[DOMAINS])
                edges = pc_stable_edges(discrete, alpha=0.05)
                bic, dag = exact_bic_dag(discrete)
                cache[key] = edges, bic, dag, skeleton(dag)
            edges, bic, dag, skel = cache[key]
            rows.append({
                "dataset": dataset, "scheme": scheme, "comparison_band": band,
                "actual_age_coverage": PRIMARY_RANGES[band] if dataset == "primary" else band,
                "n": len(group), "pc_alpha": 0.05, "pc_edges": len(edges),
                "pc_skeleton": ", ".join(f"{a}-{b}" for a, b in edges),
                "bic_edges": len(skel), "bic_skeleton": ", ".join(f"{a}-{b}" for a, b in skel),
                "bic_score": bic,
                "bic_one_optimal_dag": ", ".join(f"{a}->{b}" for a, b in dag),
                "age_min_observed": group.age.min() if dataset == "primary" else np.nan,
                "age_max_observed": group.age.max() if dataset == "primary" else np.nan,
                "age_mean": group.age.mean() if dataset == "primary" else np.nan,
                "age_sd": group.age.std() if dataset == "primary" else np.nan,
                "age_median": group.age.median() if dataset == "primary" else np.nan,
            })
    pd.DataFrame(rows).to_csv(OUT / f"{dataset}_rerun_results.csv", index=False)


def plot_counts(table: pd.DataFrame, scheme: str, bands: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharey=True)
    for ax, metric, title in zip(axes, ["pc_edges", "bic_edges"], ["PC-stable (alpha = .05)", "Exact discrete BIC"]):
        for dataset, label, color, marker in [("primary", "Primary HEXACO-100", "#245782", "o"), ("sapa", "SAPA IPIP-HEXACO", "#c26628", "s")]:
            sub = table[(table.dataset == dataset) & (table.scheme == scheme)].set_index("comparison_band").loc[bands]
            ax.plot(range(len(bands)), sub[metric], marker=marker, color=color, label=label, linewidth=1.8)
        ax.set_xticks(range(len(bands)))
        labels = [b.replace("18 and under", "18 and\nunder*").replace("60 and over", "60 and\nover*") for b in bands]
        ax.set_xticklabels(labels, fontsize=9, rotation=35 if len(bands) > 4 else 0)
        ax.set_ylim(-0.4, 15.5)
        ax.set_yticks(range(0, 16, 3))
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Age category (unequal widths)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Retained skeleton edges (maximum 15)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Age boundaries aligned to SAPA: " + ("four pooled groups" if scheme == "pooled_four" else "eight public bands"), fontsize=14)
    fig.text(0.07, 0.04, "* Primary sample: ages 3-18 and 60-100; SAPA: open-ended public categories.\nDescriptive estimates with unequal sample sizes; these are not tests of age-group differences.", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.13, 1, 0.94])
    fig.savefig(OUT / f"{scheme}_edge_counts.png", dpi=220)
    fig.savefig(OUT / f"{scheme}_edge_counts.svg")
    plt.close(fig)


def summarize() -> None:
    table = pd.concat([pd.read_csv(OUT / f"{d}_rerun_results.csv") for d in SOURCE_FILES], ignore_index=True)
    table.to_csv(OUT / "all_rerun_results.csv", index=False)
    for scheme, bands in [("pooled_four", POOLED), ("native_eight", NATIVE)]:
        rows = []
        for band in bands:
            pair = table[(table.scheme == scheme) & (table.comparison_band == band)].set_index("dataset")
            rows.append({"comparison_band": band, "primary_ages": PRIMARY_RANGES[band], "sapa_public_ages": band,
                         **{f"{d}_{field}": int(pair.loc[d, field]) for d in SOURCE_FILES for field in ["n", "pc_edges", "bic_edges"]}})
        wide = pd.DataFrame(rows)
        wide.to_csv(OUT / f"{scheme}_comparison.csv", index=False)
        plot_counts(table, scheme, bands)
        print("\n" + wide.to_string(index=False), flush=True)
    provenance = {
        "analysis": "Reviewer 1, major comment 2: primary ages regrouped to SAPA public boundaries",
        "inputs": {str(path.relative_to(ROOT)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in SOURCE_FILES.values()},
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "model": {"discretization": "5 equal-width categories over 1-5; unchanged implementation", "pc_alpha": 0.05,
                  "pc_test": "G^2 asymptotic chi-square, full 5-level state space", "max_conditioning_set": 4,
                  "bic": "loglik - 0.5 * parameter_count * log(n)", "max_parents": 5,
                  "bic_search": "existing exact dynamic-programming search", "reported_edges": "undirected skeleton"},
        "analysis_sources": {f"analysis/{name}": hashlib.sha256((ROOT / "analysis" / name).read_bytes()).hexdigest() for name in [Path(__file__).name, "sapa_hexaco_pc_bic.py", "hexaco_bic_exhaustive.py"]},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    pooled = pd.read_csv(OUT / "pooled_four_comparison.csv")
    native = pd.read_csv(OUT / "native_eight_comparison.csv")
    peak_lines = []
    for scheme in ["pooled_four", "native_eight"]:
        for dataset in SOURCE_FILES:
            sub = table[(table.scheme == scheme) & (table.dataset == dataset)]
            for metric in ["pc_edges", "bic_edges"]:
                maximum = int(sub[metric].max())
                peaks = ", ".join(sub.loc[sub[metric].eq(maximum), "comparison_band"])
                peak_lines.append(f"{scheme}, {dataset}, {metric}: maximum {maximum} at {peaks}.")
    report = """Reviewer 1, major comment 2: SAPA-compatible age regrouping

Purpose and scope
The primary sample contains exact integer ages. Its observations can be grouped
using SAPA's public cut points, without inventing individual ages in SAPA.
Four pooled groups use <=18, 19-39, 40-59 and >=60. An additional analysis uses
all eight native SAPA bands. Existing manuscript outputs are preserved.

Inputs and unchanged choices
Both datasets use the existing saved analytic domain-score files. Primary N is
98,917 with ages 3-100. SAPA N is 18,616 and requires at least three answered
items per domain. The existing five equal-width bins over scores 1-5,
PC-stable at alpha=.05, and exact discrete BIC functions are reused.
Each unique group is refit using both methods. The identical endpoint groups
are reused between the four-group and eight-band summaries.
No sampling or random seed is involved. No model tuning is based on results.
The BIC skeleton derives from one optimal DAG; directions are not interpreted.

What regrouping changes
The primary youngest group now includes all 98 participants aged 3-12 along
with those aged 13-18. The 2,224 participants aged 19 join the 19-39 group.
The 4,094 participants aged 60-64 join the 60-100 group.

Remaining limitations
The primary endpoint ranges are 3-18 and 60-100. SAPA's <=18 and >=60 categories
do not reveal exact ages, so identical endpoint support and within-band age
distributions cannot be verified. A separate SAPA childhood estimate remains
impossible. Age midpoints are not substituted for individual ages.
This aligns observable cut points, not measurements, sample composition,
within-band age distributions, or group sizes. PC/BIC sample-size sensitivity,
sparse discrete tables, measurement comparability and structural uncertainty
are not resolved by this rerun. These are descriptive partial external checks,
not formal tests of age differences or proof of within-person development.

Four-group comparison
""" + pooled.to_string(index=False) + "\n\nEight-band comparison\n" + native.to_string(index=False) + "\n\nDescriptive maxima (not inferential comparisons)\n" + "\n".join(peak_lines) + "\n\nBaseline reproduction was independently checked; see baseline_validation.txt.\n\nReproduce from the project root:\nMPLCONFIGDIR=/private/tmp/psychology-mpl python3 analysis/reviewer1_comment2_age_harmonization.py\n"
    (OUT / "methods_results.txt").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["primary", "sapa", "all"], default="all")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for dataset in SOURCE_FILES if args.dataset == "all" else [args.dataset]:
            run_dataset(dataset)
    if args.summarize_only or args.dataset == "all":
        summarize()


if __name__ == "__main__":
    main()
