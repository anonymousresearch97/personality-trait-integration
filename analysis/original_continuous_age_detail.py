"""Continuous PC and weighted age comparisons requested for the revision wrap-up.

All age bands and contrasts are fixed below. This is exploratory follow-up,
not a new preregistration or an attempt to optimize a peak location.
"""
from pathlib import Path
import hashlib
import itertools
import json
import time

import numpy as np
import pandas as pd

from longitudinal_gaussian_bn import moments, pc_stable_gaussian, exact_gaussian_bic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/longitudinal_peak_wrapup/original_age_detail"
SOURCE = ROOT / "outputs/original_hexaco_scored_domains.csv"
NODES = ["H", "E", "X", "A", "C", "O"]
PAIRS = list(itertools.combinations(range(6), 2))
PAIR_NAMES = [NODES[i] + "–" + NODES[j] for i, j in PAIRS]
GROUPS = ["13–19", "20–39", "40–64", "65–100"]
BANDS = ["13–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80–100"]
SEED, BOOT_REPS, MATCH_REPS, MATCH_N = 2026101521, 2000, 1000, 1000


def statistics(values):
    _, corr = moments(values)
    prec = np.linalg.inv(corr)
    partial = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
    edges = np.array([partial[i, j] for i, j in PAIRS])
    metrics = np.array([np.abs(edges).mean(),
                        np.mean([abs(corr[i, j]) for i, j in PAIRS]),
                        -0.5*np.linalg.slogdet(corr)[1]])
    return edges, metrics, corr


def networks(values, corr):
    pc = pc_stable_gaussian(corr, len(values), alpha=.05, nodes=NODES)
    _, bic = exact_gaussian_bic(corr, len(values), nodes=NODES)
    return len(pc), len(bic)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"analysis_plan.txt").write_text(
        "Exploratory continuation, 15 October 2026. Standard Gaussian PC-stable at alpha=.05 "
        "and exact Gaussian BIC use continuous scores, no discretization. Original four "
        "nonchild age groups support weighted comparisons. The fixed primary weighted "
        "metric is mean absolute full-order partial correlation across all15 pairs; "
        "mean absolute Pearson correlation and Gaussian total-correlation proxy are "
        "reported alongside it, not selected by outcome.\n"
        "Two thousand independent participant bootstraps per age group estimate sampling "
        "uncertainty. All45 signed edge differences across adjacent original groups are "
        "reported with centered, fixed-SE max-|t| bootstrap simultaneous95% intervals; "
        "this is an approximate bootstrap comparison, not permutation exchangeability. "
        "Participant sampling within each group is assumed independent; selection and "
        "measurement equivalence remain unaddressed.\n"
        "A fixed broad-adult contrast uses the equal-weight average of strengths at20–39 "
        "and40–64, compared separately to13–19 and65+. It is not pooled-score covariance "
        "and does not test a unique peak age or unimodality.\n"
        "Ten-year sensitivity bands are13–19,20s,30s,40s,50s,60s,70s,80–100. Groups "
        "with at least1000 people enter1000 equal-N1000 subsamples without replacement. "
        "The80+ group (N133) is descriptive only. Narrow-band observations remain "
        "cross-sectional, not within-person development.\n")
    raw = pd.read_csv(SOURCE)
    raw = raw[raw.age >= 13].copy()
    raw["group"] = pd.cut(raw.age, [12, 19, 39, 64, 100], labels=GROUPS)
    full_edges, full_metrics, edge_draws, metric_draws, descriptive = [], [], [], [], []
    metric_names = ["mean_absolute_partial_r", "mean_absolute_r", "gaussian_total_correlation"]
    for gi, label in enumerate(GROUPS):
        group = raw[raw.group == label]
        values = group[NODES].to_numpy()
        edges, metrics, _ = statistics(values)
        full_edges.append(edges); full_metrics.append(metrics)
        ed = np.empty((BOOT_REPS, 15)); md = np.empty((BOOT_REPS, 3))
        for rep in range(BOOT_REPS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, gi, rep]))
            selected = rng.integers(0, len(values), len(values))
            ed[rep], md[rep], _ = statistics(values[selected])
        edge_draws.append(ed); metric_draws.append(md)
        for mi, name in enumerate(metric_names):
            descriptive.append(dict(age_group=label, n=len(values), mean_age=group.age.mean(),
                                    metric=name, estimate=metrics[mi],
                                    bootstrap_p2_5=np.quantile(md[:, mi], .025),
                                    bootstrap_p97_5=np.quantile(md[:, mi], .975)))
        print(f"Weighted participant bootstraps complete: {label}, n={len(values)}", flush=True)
    full_edges = np.array(full_edges); full_metrics = np.array(full_metrics)
    edge_draws = np.array(edge_draws); metric_draws = np.array(metric_draws)
    pd.DataFrame(descriptive).to_csv(OUT/"original_group_strengths.csv", index=False)
    pd.DataFrame([dict(age_group=label, edge=PAIR_NAMES[j], partial_r=full_edges[i, j])
                  for i, label in enumerate(GROUPS) for j in range(15)]).to_csv(OUT/"signed_partial_correlations.csv", index=False)

    # All 45 adjacent-age signed-edge contrasts, simultaneous over the fixed family.
    observed = (full_edges[1:] - full_edges[:-1]).ravel()
    draws = (edge_draws[1:] - edge_draws[:-1]).transpose(1, 0, 2).reshape(BOOT_REPS, 45)
    se = draws.std(axis=0, ddof=1)
    centered = (draws-observed)/se
    null_max = np.abs(centered).max(axis=1)
    critical = np.quantile(null_max, .95)
    rows = []
    for i in range(3):
        for j in range(15):
            k = i*15+j
            lo, hi = observed[k]-critical*se[k], observed[k]+critical*se[k]
            rows.append(dict(contrast=f"{GROUPS[i+1]} minus {GROUPS[i]}", edge=PAIR_NAMES[j],
                             younger_partial_r=full_edges[i,j], older_partial_r=full_edges[i+1,j],
                             difference=observed[k], bootstrap_se=se[k], simultaneous_lower=lo,
                             simultaneous_upper=hi, excludes_zero=bool(lo>0 or hi<0),
                             adjusted_bootstrap_p=(1+np.sum(null_max>=abs(observed[k]/se[k])))/(BOOT_REPS+1)))
    pd.DataFrame(rows).to_csv(OUT/"adjacent_edge_differences_simultaneous.csv", index=False)

    broad = []
    for mi, metric in enumerate(metric_names):
        adult = .5*(full_metrics[1,mi]+full_metrics[2,mi])
        b_adult = .5*(metric_draws[1,:,mi]+metric_draws[2,:,mi])
        first = b_adult-metric_draws[0,:,mi]
        last = b_adult-metric_draws[3,:,mi]
        for label, estimate, bvalues in [
                ("adult average minus adolescents",adult-full_metrics[0,mi],first),
                ("adult average minus older adults",adult-full_metrics[3,mi],last),
                ("minimum of both side contrasts",min(adult-full_metrics[0,mi],adult-full_metrics[3,mi]),np.minimum(first,last))]:
            broad.append(dict(metric=metric, contrast=label, estimate=estimate,
                              bootstrap_p2_5=np.quantile(bvalues,.025),
                              bootstrap_p97_5=np.quantile(bvalues,.975),
                              fraction_positive=np.mean(bvalues>0)))
    pd.DataFrame(broad).to_csv(OUT/"broad_adult_hump_contrasts.csv", index=False)
    np.savez_compressed(OUT/"group_bootstrap_arrays.npz", edge_draws=edge_draws,metric_draws=metric_draws,
                        full_edges=full_edges,full_metrics=full_metrics)

    raw["band"] = pd.cut(raw.age,[12,19,29,39,49,59,69,79,100],labels=BANDS)
    band_rows, matched = [], []
    for bi, label in enumerate(BANDS):
        group = raw[raw.band == label]; values = group[NODES].to_numpy()
        _, metrics, corr = statistics(values)
        pc, bic = networks(values,corr)
        band_rows.append(dict(age_band=label,n=len(values),mean_age=group.age.mean(),
                              PC_links=pc,BIC_links=bic,**dict(zip(metric_names,metrics)),
                              matched_included=len(values)>=MATCH_N))
        if len(values)<MATCH_N:
            continue
        for rep in range(MATCH_REPS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED,100+bi,rep]))
            selected = rng.choice(len(values),MATCH_N,replace=False)
            sample = values[selected]; _,metrics,corr=statistics(sample)
            pc,bic=networks(sample,corr)
            matched.append(dict(age_band=label,replicate=rep+1,sample_n=MATCH_N,source_n=len(values),
                                PC_links=pc,BIC_links=bic,**dict(zip(metric_names,metrics))))
        print(f"Continuous PC/BIC decade fits complete: {label}",flush=True)
    pd.DataFrame(band_rows).to_csv(OUT/"decade_full_sample.csv",index=False)
    matched=pd.DataFrame(matched)
    matched.to_csv(OUT/"decade_matched_draws.csv",index=False)
    summaries=[]
    for label in BANDS[:-1]:
        group=matched[matched.age_band==label]
        for metric in ["PC_links","BIC_links",*metric_names]:
            v=group[metric]
            summaries.append(dict(age_band=label,metric=metric,sample_n=MATCH_N,draws=MATCH_REPS,
                                  mean=v.mean(),sd=v.std(),p2_5=v.quantile(.025),p97_5=v.quantile(.975)))
    pd.DataFrame(summaries).to_csv(OUT/"decade_matched_summary.csv",index=False)
    metadata=dict(seed=SEED,bootstrap_replicates=BOOT_REPS,matched_replicates=MATCH_REPS,matched_n=MATCH_N,
                  signed_edge_contrast_family=45,max_t_critical=float(critical),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.time()-started)
    (OUT/"provenance.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("\nBROAD ADULT HUMP\n"+pd.DataFrame(broad).to_string(index=False),flush=True)
    print("\nDECADE ESTIMATES\n"+pd.DataFrame(band_rows).to_string(index=False),flush=True)
    print("\nSimultaneous edge differences excluding zero:",sum(row["excludes_zero"] for row in rows),"/45",flush=True)


if __name__=="__main__":
    main()
