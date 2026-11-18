"""Controlled entry points for the archived cross-sectional analysis.

Run --help for stages. Saved references are copied only where an output does
not yet exist; numerical fitting stages then replace their own derived files.
This private student bundle includes raw inputs and prepared analysis data.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".runtime/cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")


def references():
    """Initialize absent comparison files, preserving existing recomputed ones."""
    copied = 0
    for source in sorted((ROOT / "reference_outputs").rglob("*")):
        if source.is_file():
            target = ROOT / "outputs" / source.relative_to(ROOT / "reference_outputs")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
    print(f"Initialized {copied} absent output files from immutable reference_outputs.", flush=True)


def run(name, *args):
    subprocess.run([sys.executable, str(ROOT / "analysis" / name), *args], cwd=ROOT, check=True)


def prepare():
    import original_hexaco_pc_bic as p
    import sapa_hexaco_pc_bic as s
    required = [p.DATA_FILE, s.DATA_FILE, s.KEY_FILE]
    missing = [str(f.relative_to(ROOT)) for f in required if not f.is_file()]
    if missing:
        raise FileNotFoundError("Supply separately obtained inputs at: " + "; ".join(missing))
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    primary = p.load_original()
    assert len(primary) == 98917
    primary.to_csv(out / "original_hexaco_scored_domains.csv", index=False)
    all_rows = s.score_domains(s.load_sapa_raw(s.load_key()), s.load_key())
    selected = s.analytic_sample(all_rows, 3)
    assert len(all_rows) == 23679 and len(selected) == 18616
    all_rows.to_csv(out / "sapa_hexaco_scored_domains_all_rows.csv", index=False)
    selected.to_csv(out / "sapa_hexaco_scored_domains.csv", index=False)
    prepare_measurement()
    print("Reconstructed domain scores AND data_prepared/primary_items.csv from included raw inputs.")



def prepare_measurement():
    """Generate the visible item CSV, mapping and models; no R fitting."""
    cache = ROOT / "outputs/original_hexaco_scored_domains.csv"
    if not cache.is_file():
        import original_hexaco_pc_bic as p
        cache.parent.mkdir(parents=True, exist_ok=True)
        p.load_original().to_csv(cache, index=False)
    run("prepare_measurement_resolution.py")


def check_inputs():
    """Check bundled data and exact derived item matrix using only stdlib."""
    import csv
    import hashlib
    manifest = json.loads((ROOT / "INPUT_MANIFEST.json").read_text())
    for record in manifest:
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError("Missing bundled input: " + record["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError("Bundled input hash differs: " + record["path"])
    with (ROOT / "data_prepared/primary_items.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        expected = [f"{d}{facet}{item}" for d in "HEXACO" for facet in range(1,5) for item in range(1,5)] + ["age_group"]
        if reader.fieldnames != expected:
            raise ValueError("Unexpected primary_items.csv columns")
        counts = {}
        for row in reader:
            counts[row["age_group"]] = counts.get(row["age_group"], 0) + 1
    expected_counts = {"Adolescence 13-19":7738,"Early adulthood 20-39":51670,
                       "Middle adulthood 40-64":35496,"Older adulthood 65-100":3915}
    if counts != expected_counts:
        raise ValueError(f"Incorrect primary item sample: {counts}")
    print(f"PASS: {len(manifest)} input files verified; primary_items.csv has 98,819 rows, 96 keyed item columns and age_group.")


def measurement_audit():
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Install R and lavaan, then put Rscript on PATH. See START_HERE.txt.")
    subprocess.run([rscript, str(ROOT / "analysis/audit_submission_measurement.R")], cwd=ROOT, check=True)


def categorical():
    import pandas as pd
    import original_hexaco_pc_bic as p
    import reviewer1_comment2_age_harmonization as h
    scores = pd.read_csv(ROOT / "outputs/original_hexaco_scored_domains.csv")
    p.run_pc(scores).to_csv(ROOT / "outputs/original_pc_stable_edge_counts.csv", index=False)
    p.run_exact_bic(scores).to_csv(ROOT / "outputs/original_bic_exact_edge_counts.csv", index=False)
    h.OUT.mkdir(parents=True, exist_ok=True)
    h.run_dataset("sapa")
    rows = pd.read_csv(h.OUT / "sapa_rerun_results.csv").query("scheme == 'pooled_four'")
    pc, bic = [], []
    for row, legacy in zip(rows.itertuples(), p.AGE_LABELS[1:]):
        common = dict(dataset="SAPA IPIP-HEXACO", age_group=legacy,
                      age_note=f"SAPA {row.comparison_band}", n=row.n, status="estimated")
        pc.append(dict(**common, edge_count=row.pc_edges, edges=row.pc_skeleton))
        bic.append(dict(**common, bic=row.bic_score, skeleton_edge_count=row.bic_edges,
                        dag_edge_count=row.bic_edges, dag_edges=row.bic_one_optimal_dag,
                        skeleton_edges=row.bic_skeleton))
    dest = ROOT / "outputs/six_png_results_tables"
    dest.mkdir(parents=True, exist_ok=True)
    # Legacy age_group column is retained for original plotting compatibility;
    # age_note and all new figure labels use the actual public SAPA bands.
    pd.DataFrame(pc).to_csv(dest / "sapa_pc_stable_manuscript_age_groups.csv", index=False)
    pd.DataFrame(bic).to_csv(dest / "sapa_bic_manuscript_age_groups.csv", index=False)


def gaussian_full():
    import numpy as np
    import pandas as pd
    import original_continuous_bn_sensitivity as e
    import sapa_gaussian_wrapup as s
    run("original_continuous_bn_sensitivity.py", "--full-only")
    scores = pd.read_csv(s.SOURCE)
    scores["pooled_band"] = scores.age_band.map(s.COLLAPSE)
    rows = []
    for scheme, column, bands in [("pooled_four", "pooled_band", s.POOLED),
                                   ("native_eight", "age_band", s.NATIVE)]:
        for band in bands:
            group = scores.loc[scores[column].eq(band)]
            pc, bic, dag = e.fit_sample(group[e.DOMAINS].to_numpy(float))
            for method, edges in [("Gaussian PC-stable", pc), ("Exact Gaussian BIC", e.skeleton(dag))]:
                rows.append(dict(dataset="SAPA IPIP-HEXACO", scheme=scheme, age_band=band,
                                 n=len(group), min_items_per_domain=3, method=method,
                                 edge_count=len(edges), skeleton_edges=e.edge_string(edges),
                                 alpha=.05 if "PC" in method else np.nan,
                                 bic_score=bic if "BIC" in method else np.nan,
                                 one_optimal_dag=e.edge_string(dag, True) if "BIC" in method else ""))
    s.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(s.OUT / "full_sample_results.csv", index=False)
    print("Fresh primary and SAPA full-sample Gaussian fits complete; no subsampling.")


def gaussian_equal_n(workers):
    run("original_continuous_bn_sensitivity.py", "--reps", "1000", "--workers", str(workers))
    run("sapa_gaussian_wrapup.py")


def cpdag_strict():
    import pandas as pd
    import revision_cpdag_reconstruction as r
    import sapa_gaussian_wrapup as s
    r.rows.clear()
    r.sepsets_log.clear()
    r.OUT.mkdir(parents=True, exist_ok=True)
    for dataset, cache, source, group_col in [
        ("Primary", "original_hexaco_scored_domains.csv", "original_continuous_bn_sensitivity/full_sample_results.csv", "age_group"),
        ("SAPA", "sapa_hexaco_scored_domains.csv", "sapa_gaussian_wrapup/full_sample_results.csv", "age_band"),
    ]:
        data = pd.read_csv(ROOT / "outputs" / cache)
        refpath = ROOT / "outputs" / source
        fits = pd.read_csv(refpath)
        if dataset == "Primary":
            fits = fits.loc[~fits.age_group.str.contains("Childhood")]
            data["fit_group"] = data.age_group
        else:
            fits = fits.query("scheme == 'pooled_four'")
            data["fit_group"] = data.age_band.map(s.COLLAPSE)
        for group in fits[group_col].unique():
            selected = fits.loc[fits[group_col].eq(group)]
            pc = selected.loc[selected.method.str.contains("PC")].iloc[0]
            bic = selected.loc[selected.method.str.contains("BIC")].iloc[0]
            values = data.loc[data.fit_group.eq(group), r.DOMAINS].to_numpy(float)
            r.process(dataset, group, values, pc, bic, refpath)
    assert len(r.rows) == 16
    (r.OUT / "pc_separating_sets.json").write_text(json.dumps(r.sepsets_log, indent=2) + "\n")
    print("Strict CPDAG audit complete. Incompatible PC collider sets remain explicitly unresolved.")


def cpdag_resolved():
    """Optional documented change to PC orientation policy; skeletons unchanged."""
    import pandas as pd
    import revision_cpdag_reconstruction as r
    import complete_main_pc_cpdags as c
    source = ROOT / "outputs/revision_cpdag"
    rows = pd.read_csv(source / "network_cpdag_results.csv").fillna("").to_dict("records")
    sep_map = {(x["dataset"], x["group"]): x for x in json.loads((source / "pc_separating_sets.json").read_text())}
    assert len(rows) == 16 and all(x["dataset"] in ("Primary", "SAPA") for x in rows)
    for row in rows:
        edges = r.parse(row["skeleton_edges"])
        pc = row["method"] == "Gaussian PC-stable"
        valid = row["cpdag_valid"] in (True, "True")
        raw = r.pc_colliders(edges, sep_map[row["dataset"], row["group"]]["separating_sets"]) if pc else frozenset()
        row.update(orientation_policy="existing_verified", retries=0, fallback=False,
                   fallback_kind="none", deleted_colliders="", added_colliders="",
                   resolution_seed="not_used_deterministic",
                   original_collider_suggestions=c.collider_text(raw) if pc else row["unshielded_colliders"])
        if valid:
            arrows = set(r.parse(row["directed_edges"], True))
            dag = next(d for d, cols in c.all_dags(edges)
                       if arrows.issubset(d) and c.collider_text(cols) == row["unshielded_colliders"])
        else:
            assert pc
            dag, details = c.retry_pc(edges, sep_map[row["dataset"], row["group"]]["separating_sets"])
            row.update(details)
            cols = r.colliders(dag)
            result, equivalents = r.enumerate_cpdag(edges, cols)
            assert result["cpdag_valid"] and dag in equivalents
            row.update(result, status="verified_CPDAG_after_PC_conflict_resolution",
                       unshielded_colliders=c.collider_text(cols), collider_count=len(cols),
                       deleted_colliders=c.collider_text(raw-cols), added_colliders=c.collider_text(cols-raw))
        row["representative_dag"] = r.edge_text(dag, True)
        idir, iundir = r.independent_dag2cpdag(dag)
        assert idir == row["directed_edges"] and iundir == row["undirected_edges"]
        assert r.skeleton(dag) == edges
        row["independent_dag2cpdag_matches"] = True
    dest = ROOT / "outputs/main_paper_cpdags"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "network_cpdag_results.csv", index=False)
    (dest / "packaged_orientation_policy.json").write_text(json.dumps({
        "networks": 16, "conflict_resolved": sum(x["orientation_policy"] != "existing_verified" for x in rows),
        "no_resampling": True, "all_skeletons_unchanged": True,
        "warning": "Optional policy-dependent PC orientations; see README and per-row added/deleted colliders."
    }, indent=2) + "\n")


def figures(cpdag_policy):
    import pandas as pd
    import matplotlib.pyplot as plt
    import plot_publication_skeletons as sk
    import plot_revision_cpdags as cp
    # Skeletons do not depend on the choice of orientation policy.
    source = ROOT / "outputs" / ("main_paper_cpdags" if cpdag_policy == "resolved" else "revision_cpdag") / "network_cpdag_results.csv"
    frame = pd.read_csv(source)
    assert len(frame) == 16
    sk.OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    for dataset, groups, labels, stem in [
        ("Primary", cp.PRIMARY_GROUPS, cp.PRIMARY_TITLES, "primary_gaussian"),
        ("SAPA", cp.SAPA_GROUPS, cp.SAPA_TITLES, "sapa_gaussian")
    ]:
        sk.draw_figure(frame, dataset, groups, labels, stem + "_skeletons")
        cp.OUT = ROOT / "outputs/supplementary_cpdags" / cpdag_policy
        cp.OUT.mkdir(parents=True, exist_ok=True)
        note = ("PC conflicts are shown as skeletons when no jointly compatible CPDAG exists."
                if cpdag_policy == "strict" else
                "PC conflicts use the documented retry/fallback policy; arrows depend on that policy.")
        cp.draw_cross_sectional(frame, dataset, groups, labels, stem + "_cpdags", note)
    run("plot_categorical_full_sample_appendix.py")


def measurement():
    run("minimal_primary_measurement_audit.py")
    run("minimal_sapa_measurement_audit.py")
    prepare_measurement()
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Install R with lavaan before requesting the optional measurement stage.")
    for domain in "HEXACO":
        subprocess.run([rscript, str(ROOT / "analysis/primary_measurement_resolution.R"), domain], cwd=ROOT, check=True)
    run("sapa_measurement_resolution.py")
    run("summarize_measurement_resolution.py")
    subprocess.run([rscript, str(ROOT / "analysis/audit_submission_measurement.R")], cwd=ROOT, check=True)


def main():
    if not __debug__:
        raise RuntimeError("Run without Python -O/-OO; numerical validation assertions must remain enabled.")
    choices = ["check-inputs", "prepare-measurement", "measurement-audit", "references", "prepare", "categorical", "gaussian-full", "gaussian-equal-n",
               "sparse", "descriptives", "alpha", "associations", "measurement",
               "cpdag-strict", "cpdag-resolved", "figures", "audit"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=choices)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--all-draws", action="store_true", help="For audit: refit every standard equal-N draw, not only selected draws.")
    parser.add_argument("--cpdag-policy", choices=["strict", "resolved"], default="strict", help="For figures: strict keeps incompatible PC panels as skeletons.")
    args = parser.parse_args()
    os.chdir(ROOT)
    references()
    if args.stage == "check-inputs":
        check_inputs(); return
    if args.stage == "prepare-measurement":
        prepare_measurement(); return
    if args.stage == "measurement-audit":
        measurement_audit(); return
    if args.stage == "references": return
    if args.workers < 1:
        parser.error("--workers must be positive")
    caches = ["original_hexaco_scored_domains.csv", "sapa_hexaco_scored_domains.csv", "sapa_hexaco_scored_domains_all_rows.csv"]
    if args.stage != "prepare" and any(not (ROOT / "outputs" / x).is_file() for x in caches):
        prepare()
    if args.stage == "prepare": prepare()
    elif args.stage == "categorical": categorical()
    elif args.stage == "gaussian-full": gaussian_full()
    elif args.stage == "gaussian-equal-n": gaussian_equal_n(args.workers)
    elif args.stage == "sparse": run("sparse_gaussian_comparison.py")
    elif args.stage == "descriptives": run("complete_submission_descriptives.py")
    elif args.stage == "alpha": run("audit_categorical_alpha.py")
    elif args.stage == "associations": run("original_continuous_age_detail.py")
    elif args.stage == "measurement": measurement()
    elif args.stage == "cpdag-strict": cpdag_strict()
    elif args.stage == "cpdag-resolved": cpdag_resolved()
    elif args.stage == "figures": figures(args.cpdag_policy)
    elif args.stage == "audit":
        run("audit_submission_core.py", *(["--all-draws"] if args.all_draws else []))
        run("audit_sapa_categorical_discrepancy.py")
        run("audit_submission_associations.py")


if __name__ == "__main__":
    main()
