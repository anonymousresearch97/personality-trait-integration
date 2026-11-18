"""Redraw primary and SAPA saved categorical skeletons; no estimation."""
from pathlib import Path
import hashlib
import json
import os
import zipfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import pandas as pd
from plot_publication_skeletons import draw_skeleton
from plot_revision_cpdags import PRIMARY_GROUPS, COLORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/categorical_full_sample_appendix"
PC = ROOT / "outputs/original_pc_stable_edge_counts.csv"
BIC = ROOT / "outputs/original_bic_exact_edge_counts.csv"
STEM = "primary_categorical_full_sample_skeletons"
SAPA_PC = ROOT / "outputs/six_png_results_tables/sapa_pc_stable_manuscript_age_groups.csv"
SAPA_BIC = ROOT / "outputs/six_png_results_tables/sapa_bic_manuscript_age_groups.csv"


def load_results(dataset, pc_path, bic_path, expected_n, pc_counts, bic_counts):
    pc, bic = pd.read_csv(pc_path), pd.read_csv(bic_path)
    expected_counts = {"Categorical PC": pc_counts, "Categorical BIC": bic_counts}
    rows = []
    for method, source, group_key, edge_key, count_key in (
        ("Categorical PC", pc, "age_group", "edges", "edge_count"),
        ("Categorical BIC", bic, "group" if dataset == "Primary" else "age_group", "skeleton_edges", "skeleton_edge_count"),
    ):
        for i, group in enumerate(PRIMARY_GROUPS):
            fit = source.loc[source[group_key].eq(group)]
            assert len(fit) == 1
            row = fit.iloc[0]
            assert int(row.n) == expected_n[i]
            assert int(row[count_key]) == expected_counts[method][i]
            edge_text = "" if pd.isna(row[edge_key]) else row[edge_key]
            pairs = [tuple(sorted(item.strip().split("-"))) for item in edge_text.split(",") if item.strip()]
            assert len(set(pairs)) == int(row[count_key])
            assert all(len(pair) == 2 and len(set(pair)) == 2 and set(pair) <= set("HEXACO") for pair in pairs)
            display_group = group if dataset == "Primary" else ["18 and under", "19-39", "40-59", "60 and over"][i]
            rows.append(dict(dataset=dataset, method=method, age_group=display_group,
                             source_group=group, n=int(row.n), edge_count=int(row[count_key]), skeleton_edges=edge_text))
    return pd.DataFrame(rows)


def draw_figure(frame, labels, stem, dataset):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(13.2, 8.0), facecolor="white")
    for method, color, bottom, heading_y, count_y in (
        ("Categorical PC", COLORS["Gaussian PC-stable"], .535, .975, .520),
        ("Categorical BIC", COLORS["Gaussian BIC"], .055, .475, .040),
    ):
        fig.text(.025, heading_y, method, fontsize=15,
                 color=color, fontweight="bold", va="top")
        for col, (group, label) in enumerate(zip(PRIMARY_GROUPS, labels)):
            row = frame.loc[frame.method.eq(method) & frame.source_group.eq(group)].iloc[0]
            x = .022 + .245 * col
            ax = fig.add_axes([x, bottom, .235, .335])
            draw_skeleton(ax, row, color)
            center = x + .235 / 2
            if method == "Categorical PC":
                fig.text(center, .918, label, ha="center", va="center", fontsize=12.5, color="#172b3a")
                fig.text(center, .886, f"N = {int(row.n):,}", ha="center", va="center", fontsize=10.5, color="#53616c")
            count = int(row.edge_count)
            fig.text(center, count_y, f"{count} link" + ("s" if count != 1 else ""),
                     ha="center", va="center", fontsize=11.5, color=color)
    for ext in ("png", "svg", "pdf"):
        opts = {"metadata": {"Title": f"{dataset} categorical network skeletons"}} if ext == "pdf" else {}
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, facecolor="white", **opts)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary = load_results("Primary", PC, BIC, [7738, 51670, 35496, 3915], [4, 5, 2, 1], [4, 7, 7, 4])
    sapa = load_results("SAPA", SAPA_PC, SAPA_BIC, [4550, 11510, 2253, 303], [2, 0, 0, 0], [2, 5, 0, 0])
    pd.concat([primary, sapa], ignore_index=True).to_csv(OUT / "full_sample_categorical_edges.csv", index=False)
    draw_figure(primary, ["Ages 13-19", "Ages 20-39", "Ages 40-64", "Ages 65-100"], STEM, "Primary")
    draw_figure(sapa, ["18 and under", "19-39", "40-59", "60 and over"], "sapa_categorical_full_sample_skeletons", "SAPA")
    (OUT / "caption_and_provenance.txt").write_text(
        "PRIMARY AND SAPA CATEGORICAL NETWORK SKELETONS\n\n"
        "The eight panels show saved full-sample categorical results for adolescence "
        "(13-19), early adulthood (20-39), middle adulthood (40-64), and older "
        "adulthood (65-100). Blue: PC-stable, alpha=.05; brown: exact discrete BIC. "
        "Domain scores were discretized into five equal-width intervals on the "
        "1-5 scale. PC used asymptotic conditional mutual-information/G-squared "
        "tests with nominal full-state degrees of freedom. All edges are displayed "
        "as undirected skeleton adjacencies. Fixed positions do not encode distance "
        "or association strength.\n\n"
        "H: Honesty-Humility; E: Emotionality; X: Extraversion; A: Agreeableness; "
        "C: Conscientiousness; O: Openness to Experience.\n\n"
        "N: 7738, 51670, 35496, 3915. PC links: 4, 5, 2, 1. "
        "BIC links: 4, 7, 7, 4. Childhood is outside the requested figure.\n\n"
        "SAPA: actual pooled public age bands are 18 and under, 19-39, 40-59, "
        "and 60 and over. N: 4550, 11510, 2253, 303. PC links: 2, 0, 0, 0; "
        "BIC links: 2, 5, 0, 0. Zero-link panels are fitted empty skeletons, "
        "not missing estimates. SAPA bands differ from primary age ranges. "
        "No unavailable childhood category is plotted for SAPA.\n\n"
        "Provenance: redrawn from saved Python reruns of the original primary "
        "dataset, produced by analysis/original_hexaco_pc_bic.py. These files "
        "are not copies of the submitted manuscript artwork; exact agreement "
        "with its displayed edge sets has not been checked against that artwork. "
        "SAPA edge lists are read from the saved pooled categorical results "
        "under outputs/six_png_results_tables/. Only their displayed age labels "
        "are corrected to actual SAPA bands. No network was fitted and no "
        "participants were resampled for these figures.\n\n"
        "Full-sample displays alone do not adjust for unequal group sizes. "
        "This figure request does not modify or invalidate previously saved "
        "resampling diagnostics.\n"
    )
    validation = dict(networks=16, requested_counts_match=True, no_model_fitting=True,
                      no_resampling=True, graph_type="undirected skeleton",
                      exact_submitted_artwork_identity_verified=False,
                      sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (PC, BIC, SAPA_PC, SAPA_BIC)})
    (OUT / "figure_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    archive_path = OUT / "categorical_full_sample_appendix.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path != archive_path:
                archive.write(path, path.name)
    print(f"Created matching primary and SAPA PNG/SVG/PDF figures in {OUT}; 16 verified skeletons.")


if __name__ == "__main__":
    main()
