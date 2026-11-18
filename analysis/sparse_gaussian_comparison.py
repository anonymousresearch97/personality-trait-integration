"""One fixed stronger-penalty comparison, preserving earlier samples and fits.

PC alpha=.01 and exact Gaussian BIC penalty multiplier=2. Full samples and
the same 1,000 within-dataset equal-N draws as the preceding comparison.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import platform
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import scipy
import networkx as nx
from original_continuous_bn_sensitivity import (
    DOMAINS, AGE_LABELS, moments, pc_stable_gaussian, exact_gaussian_bic,
    skeleton, edge_string, gaussian_local_scores,
)
from sapa_gaussian_wrapup import POOLED, COLLAPSE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sparse_gaussian_comparison'
ALPHA = .01
PENALTY = 2.0
REPS = 1000
SPECS = {
    'Primary': dict(source='original_hexaco_scored_domains.csv', seed=20261015, n=3000,
                    baseline='original_continuous_bn_sensitivity', labels=AGE_LABELS,
                    expected_ns=[98, 7738, 51670, 35496, 3915]),
    'SAPA': dict(source='sapa_hexaco_scored_domains.csv', seed=20261020, n=300,
                 baseline='sapa_gaussian_wrapup', labels=POOLED,
                 expected_ns=[4550, 11510, 2253, 303]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def sources(dataset):
    spec = SPECS[dataset]
    data = pd.read_csv(ROOT / 'outputs' / spec['source'])
    if dataset == 'Primary':
        data['band'] = data.age_group
        assert pd.cut(data.age, [2, 12, 19, 39, 64, 100], labels=AGE_LABELS).astype(str).eq(data.band).all()
        groups = {i: data.loc[data.band.eq(band)] for i, band in enumerate(spec['labels'])}
    else:
        data['band'] = data.age_band.map(COLLAPSE)
        assert data.RID.is_unique and data.band.notna().all()
        groups = {i: data.loc[data.band.eq(band)] for i, band in enumerate(spec['labels'], 1)}
    assert [len(g) for g in groups.values()] == spec['expected_ns']
    assert np.isfinite(data[DOMAINS]).all().all()
    return groups


def baselines(dataset, analysis):
    spec = SPECS[dataset]
    name = 'full_sample_results.csv' if analysis == 'full' else f"matched_n{spec['n']}_replicate_results.csv"
    frame = pd.read_csv(ROOT / 'outputs' / spec['baseline'] / name).fillna('')
    if dataset == 'Primary':
        frame = frame.rename(columns={'original_group_index': 'group_index', 'age_group': 'age_band',
                                      'sample_indices_sha256': 'source_row_indices_sha256'})
    elif analysis == 'full':
        frame = frame.loc[frame.scheme.eq('pooled_four')].copy()
        frame['group_index'] = frame.age_band.map({band: i for i, band in enumerate(POOLED, 1)})
    frame['method'] = frame.method.map(lambda s: 'PC' if 'PC' in s else 'BIC')
    if analysis == 'full':
        frame['replicate'] = 0
    return frame.set_index(['group_index', 'replicate', 'method'])


def fit_rows(dataset, idx, group, rep, selected, analysis, baseline):
    sample = group.loc[selected]
    values = sample[DOMAINS].to_numpy(dtype=float)
    n = len(values)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cov, corr = moments(values)
        pc = pc_stable_gaussian(corr, n, alpha=ALPHA)
        score, dag = exact_gaussian_bic(cov, n, penalty_multiplier=PENALTY)
    assert np.isfinite(score)
    graph = nx.DiGraph(); graph.add_nodes_from(DOMAINS); graph.add_edges_from(dag)
    assert nx.is_directed_acyclic_graph(graph)
    row_hash = digest(selected)
    participant_hash = digest(sample.RID) if dataset == 'SAPA' else ''
    rows = []
    for method, edges in [('PC', pc), ('BIC', skeleton(dag))]:
        old = baseline.loc[(idx, rep, method)]
        if analysis == 'equal_n':
            assert row_hash == old.source_row_indices_sha256
            assert n == old.sample_n and len(group) == old.source_n
            if dataset == 'SAPA':
                assert participant_hash == old.participant_ids_sha256
        else:
            assert old.n == n
            if method == 'PC':
                standard_edges = pc_stable_gaussian(corr, n, alpha=.05)
            else:
                old_score, old_dag = exact_gaussian_bic(cov, n)
                standard_edges = skeleton(old_dag)
                assert np.isclose(old_score, float(old.bic_score), atol=1e-7, rtol=0)
            assert edge_string(standard_edges) == old.skeleton_edges
        if method == 'BIC':
            assert len(edges) <= old.edge_count
        assert len(edges) == len(set(edges)) and 0 <= len(edges) <= 15
        rows.append(dict(dataset=dataset, analysis=analysis, group_index=idx, age_band=group.band.iloc[0],
                         source_n=len(group), sample_n=n, replicate=rep, method=method,
                         edge_count=len(edges), standard_edge_count=int(old.edge_count),
                         change=len(edges)-int(old.edge_count), skeleton_edges=edge_string(edges),
                         standard_skeleton_edges=old.skeleton_edges,
                         alpha=ALPHA if method == 'PC' else np.nan,
                         penalty_multiplier=PENALTY if method == 'BIC' else np.nan,
                         bic_score=score if method == 'BIC' else np.nan,
                         one_optimal_dag=edge_string(dag, directed=True) if method == 'BIC' else '',
                         source_row_indices_sha256=row_hash, participant_ids_sha256=participant_hash))
    warnings_out = [dict(dataset=dataset, group_index=idx, replicate=rep, message=str(w.message)) for w in caught]
    return rows, warnings_out


def run_group(dataset, idx, group):
    spec = SPECS[dataset]
    baseline = baselines(dataset, 'equal_n')
    indices = group.index.to_numpy()
    rows, warning_rows = [], []
    for rep in range(1, REPS + 1):
        rng = np.random.default_rng(np.random.SeedSequence([spec['seed'], idx, rep]))
        selected = np.sort(rng.choice(indices, spec['n'], replace=False))
        assert len(selected) == len(np.unique(selected)) == spec['n']
        fitted, caught = fit_rows(dataset, idx, group, rep, selected, 'equal_n', baseline)
        rows.extend(fitted); warning_rows.extend(caught)
    print(f"{dataset} {group.band.iloc[0]}: {REPS:,} draws finished", flush=True)
    return rows, warning_rows


def summaries(results):
    rows, frequencies, maxima = [], [], []
    keys = ['dataset', 'analysis', 'group_index', 'age_band', 'method']
    for key, sub in results.groupby(keys, sort=False):
        counts = sub.edge_count.to_numpy()
        low, high = np.quantile(counts, [.025, .975], method='nearest')
        rows.append(dict(zip(keys, key)) | dict(source_n=int(sub.source_n.iloc[0]), sample_n=int(sub.sample_n.iloc[0]),
                    fits=len(sub), standard_edges=sub.standard_edge_count.mean(), sparse_edges=counts.mean(),
                    change=sub.change.mean(), sd_edges=counts.std(ddof=1) if len(sub)>1 else np.nan,
                    percentile_2_5=low, percentile_97_5=high,
                    fraction_decreased=(sub.change<0).mean(), fraction_increased=(sub.change>0).mean()))
        if key[1] == 'equal_n':
            sets = [set(e.split(', ')) if e else set() for e in sub.skeleton_edges]
            for edge in sorted('-'.join(sorted(pair)) for pair in itertools.combinations(DOMAINS, 2)):
                frequencies.append(dict(zip(keys, key)) | dict(edge=edge, fits=len(sub),
                                  selection_fraction=sum(edge in es for es in sets)/len(sets)))
    for (dataset, method), sub in results.loc[results.analysis.eq('equal_n')].groupby(['dataset','method']):
        for setting, column in [('standard', 'standard_edge_count'), ('sparse', 'edge_count')]:
            wide = sub.pivot(index='replicate', columns='group_index', values=column)
            competing = wide[[1,3,4]].max(axis=1)
            maxima.append(dict(dataset=dataset, method=method, setting=setting, draws=len(wide),
                               fraction_early_strict_maximum=(wide[2]>competing).mean(),
                               fraction_early_tied_or_strict_maximum=(wide[2]>=competing).mean()))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT/'comparison.csv', index=False)
    pd.DataFrame(frequencies).to_csv(OUT/'edge_selection_frequencies.csv', index=False)
    pd.DataFrame(maxima).to_csv(OUT/'descriptive_early_maximum_fractions.csv', index=False)
    return comparison


def plot(comparison):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for i, dataset in enumerate(SPECS):
        labels = ['3–12','13–19','20–39','40–64','65–100'] if dataset=='Primary' else ['≤18','19–39','40–59','≥60']
        offset = 0 if dataset=='Primary' else 1
        for j, method in enumerate(['PC','BIC']):
            ax = axes[i,j]
            sub = comparison.loc[comparison.dataset.eq(dataset)&comparison.method.eq(method)]
            for analysis, color, name in [('full','#7b8590','Full sample'),('equal_n','#126c8b',f"Equal N={SPECS[dataset]['n']:,}")]:
                part = sub.loc[sub.analysis.eq(analysis)].sort_values('group_index')
                x = part.group_index.to_numpy()-offset
                ax.plot(x, part.standard_edges, 'o--', color=color, alpha=.55, label=name+' · standard')
                ax.plot(x, part.sparse_edges, 'o-', color=color, label=name+' · stricter')
            ax.set_title(f"{dataset} · Gaussian {method} (equal N={SPECS[dataset]['n']:,})")
            ax.set_xticks(range(len(labels)), labels)
            ax.set_xlabel('Actual age band (years)')
            ax.set_ylim(-.3,15.6); ax.set_yticks(range(0,16,3)); ax.grid(axis='y', alpha=.18)
            ax.spines[['top','right']].set_visible(False)
            if j==0: ax.set_ylabel('Number of links (maximum 15)')
    fig.suptitle('Stricter Gaussian settings: PC α=.01; BIC penalty ×2', fontsize=16)
    handles, _ = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, ['Full sample · standard', 'Full sample · stricter',
                         'Equal N · standard', 'Equal N · stricter'],
               ncol=4, loc='upper center', bbox_to_anchor=(.5,.94), frameon=False, fontsize=9)
    fig.text(.07,.025,'Dashed: previous PC α=.05 / ordinary BIC. Solid: stricter settings. Equal-N points average the same 1,000 draws.\nEqual N applies within each dataset. Age bands differ between datasets. Lines describe selected graphs, not developmental trajectories.',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.88])
    for ext in ['png','svg']: fig.savefig(OUT/f'standard_vs_sparse.{ext}', dpi=200)
    plt.close(fig)


def main():
    start = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    groups = {dataset:sources(dataset) for dataset in SPECS}
    rows, warning_rows = [], []
    for dataset, dataset_groups in groups.items():
        baseline = baselines(dataset, 'full')
        for idx, group in dataset_groups.items():
            fitted, caught = fit_rows(dataset, idx, group, 0, group.index.to_numpy(), 'full', baseline)
            rows.extend(fitted); warning_rows.extend(caught)
    full = pd.DataFrame(rows)
    full.to_csv(OUT/'full_sample_results.csv', index=False)
    print(full[['dataset','age_band','method','standard_edge_count','edge_count']].to_string(index=False),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run_group, dataset, idx, group) for dataset,dataset_groups in groups.items()
                for idx,group in dataset_groups.items() if not (dataset=='Primary' and idx==0)]
        for job in as_completed(jobs):
            fitted, caught = job.result(); rows.extend(fitted); warning_rows.extend(caught)
    results = pd.DataFrame(rows).sort_values(['dataset','analysis','group_index','replicate','method'])
    assert len(results)==16018
    matched = results.loc[results.analysis.eq('equal_n')]
    assert matched.groupby(['dataset','group_index','method']).size().eq(REPS).all()
    assert matched.groupby(['dataset','group_index','replicate']).source_row_indices_sha256.nunique().eq(1).all()
    matched.to_csv(OUT/'matched_replicate_results.csv', index=False)
    comparison = summaries(results)
    plot(comparison)
    pd.DataFrame(warning_rows,columns=['dataset','group_index','replicate','message']).to_csv(OUT/'captured_warnings.csv',index=False)
    checks = dict(full_method_fits=18, equal_n_method_fits=16000, identical_previous_sample_hashes=8000,
                  all_bic_dags_acyclic=True, all_stronger_bic_edge_counts_no_greater_than_standard=True,
                  original_full_fits_reproduced=True, warning_count=len(warning_rows))
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    paths = [Path(__file__), ROOT/'analysis/original_continuous_bn_sensitivity.py']
    for dataset,spec in SPECS.items():
        paths += [ROOT/'outputs'/spec['source'],ROOT/'outputs'/spec['baseline']/'full_sample_results.csv',
                  ROOT/'outputs'/spec['baseline']/f"matched_n{spec['n']}_replicate_results.csv"]
    provenance = dict(settings=dict(pc_alpha=ALPHA,bic_penalty_multiplier=PENALTY),
        selection='One fixed post-review sensitivity requested before seeing its results; no outcome-based tuning.',
        specs=SPECS,repetitions_per_group=REPS,files={str(p.relative_to(ROOT)):sha(p) for p in paths},
        sampling='Same seeded samples as earlier comparison; without replacement within each draw; overlap across draws.',
        seed_rule='default_rng(SeedSequence([dataset seed, group_index, replicate]))',
        objective='sum logL_i - multiplier/2 * (parents_i+2) * log(N), maximized over all DAGs by exact subset DP',
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        runtime_seconds=time.monotonic()-start)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(comparison[['dataset','analysis','age_band','method','standard_edges','sparse_edges','change']].to_string(index=False),flush=True)
    print(json.dumps(checks),flush=True)
    print(f'Elapsed {time.monotonic()-start:.1f} seconds',flush=True)


if __name__=='__main__': main()
