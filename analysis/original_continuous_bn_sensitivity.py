"""Exploratory Gaussian BN analysis on the original continuous domain scores.

Fits Gaussian PC-stable (Fisher z, alpha=.05) and the globally optimal Gaussian
BIC DAG. Original age groups and scores are unchanged; no discretization,
threshold tuning, or selection of outcomes is performed. This evaluates an
alternative distributional assumption, not a replication of the discrete BN.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import ndtr

DOMAINS = ["H", "E", "X", "A", "C", "O"]
AGE_LABELS = ["Childhood 3-12", "Adolescence 13-19", "Early adulthood 20-39",
              "Middle adulthood 40-64", "Older adulthood 65-100"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "original_continuous_bn_sensitivity"
SOURCE = ROOT / "outputs" / "original_hexaco_scored_domains.csv"
SEED = 20261015


def gaussian_local_scores(cov: np.ndarray, n: int, penalty_multiplier: float = 1.0) -> np.ndarray:
    """Local log likelihood minus BIC penalty, using MLE covariance.

For k parents, sigma2=RSS/n and log L=-n/2*(log(2*pi)+1+log(sigma2));
the parameter count is k+2 (intercept, k slopes, residual variance).
penalty_multiplier=1 is ordinary BIC; larger values strengthen its penalty.
Correlation matrices are also accepted: they score standardized variables,
changing scores by a graph-independent constant and leaving selection intact.
"""
    cov = np.asarray(cov, dtype=float)
    if not np.isfinite(penalty_multiplier) or penalty_multiplier <= 0:
        raise ValueError("penalty_multiplier must be finite and positive.")
    p = cov.shape[0]
    assert cov.shape == (p, p) and n > p + 3
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Gaussian BN requires a finite positive definite covariance.")
    scores = np.full((p, 1 << p), -np.inf)
    log_n = math.log(n)
    for family in range(1, 1 << p):
        members = [i for i in range(p) if family & (1 << i)]
        precision = np.linalg.inv(cov[np.ix_(members, members)])
        residual_variances = 1.0 / np.diag(precision)
        k = len(members) - 1
        local = -.5 * n * (math.log(2 * math.pi) + 1 + np.log(residual_variances)) - .5 * penalty_multiplier * (k + 2) * log_n
        for child, score in zip(members, local):
            scores[child, family ^ (1 << child)] = score
    return scores


def exact_gaussian_bic(corr: np.ndarray, n: int, nodes=DOMAINS, penalty_multiplier: float = 1.0):
    """Return globally optimal BIC score and one named DAG, for covariance or corr."""
    nodes = list(nodes)
    p = len(nodes)
    assert np.shape(corr) == (p, p)
    local = gaussian_local_scores(corr, n, penalty_multiplier=penalty_multiplier)
    size = 1 << p
    best_score = local.copy()
    best_parents = np.tile(np.arange(size), (p, 1))
    for child in range(p):
        for allowed in range(size):
            if allowed & (1 << child):
                continue
            for parent in range(p):
                if allowed & (1 << parent):
                    reduced = allowed ^ (1 << parent)
                    if best_score[child, reduced] > best_score[child, allowed] + 1e-10:
                        best_score[child, allowed] = best_score[child, reduced]
                        best_parents[child, allowed] = best_parents[child, reduced]
    dp = np.full(size, -np.inf)
    dp[0] = 0.0
    choices = {}
    for subset in range(1, size):
        for sink in range(p):
            if subset & (1 << sink):
                remaining = subset ^ (1 << sink)
                score = dp[remaining] + best_score[sink, remaining]
                if score > dp[subset] + 1e-10:
                    dp[subset] = score
                    choices[subset] = (sink, int(best_parents[sink, remaining]))
    edges = []
    subset = size - 1
    while subset:
        child, parents = choices[subset]
        edges.extend((nodes[parent], nodes[child]) for parent in range(p) if parents & (1 << parent))
        subset ^= 1 << child
    return float(dp[-1]), tuple(sorted(edges))


def partial_correlation(corr, i, j, conditioning):
    if not conditioning:
        return float(corr[i, j])
    chosen = [i, j, *conditioning]
    precision = np.linalg.inv(corr[np.ix_(chosen, chosen)])
    return float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))


def fisher_pvalue(r, n, conditioning_size):
    if n <= conditioning_size + 3:
        raise ValueError("Insufficient sample size for Fisher z.")
    r = float(np.clip(r, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps))
    z = np.arctanh(r) * np.sqrt(n - conditioning_size - 3)
    return float(2 * ndtr(-abs(z)))


def pc_stable_gaussian(corr, n, alpha=.05, nodes=DOMAINS):
    """Return the PC-stable skeleton using Gaussian partial correlations."""
    nodes = list(nodes)
    index = {name:i for i, name in enumerate(nodes)}
    adjacency = {node:set(nodes) - {node} for node in nodes}
    precision_cache = {}
    level = 0
    while True:
        snapshot = {node:set(neighbors) for node,neighbors in adjacency.items()}
        edges = sorted((a,b) for a in nodes for b in snapshot[a] if a < b)
        for x,y in edges:
            candidates = []
            for base in (snapshot[x] - {y}, snapshot[y] - {x}):
                if len(base) >= level:
                    candidates.extend(itertools.combinations(sorted(base), level))
            seen = set()
            for conditioning in candidates:
                if conditioning in seen:
                    continue
                seen.add(conditioning)
                if not conditioning:
                    r = corr[index[x],index[y]]
                else:
                    family = tuple(sorted(index[node] for node in (x,y,*conditioning)))
                    if family not in precision_cache:
                        precision_cache[family] = np.linalg.inv(corr[np.ix_(family,family)])
                    precision = precision_cache[family]
                    i,j = family.index(index[x]),family.index(index[y])
                    r = -precision[i,j] / np.sqrt(precision[i,i] * precision[j,j])
                if fisher_pvalue(r,n,len(conditioning)) > alpha:
                    adjacency[x].discard(y)
                    adjacency[y].discard(x)
                    break
        level += 1
        if not any(len(neighbors) >= level + 1 for neighbors in adjacency.values()) or level > len(nodes)-2:
            break
    return tuple(sorted((a,b) for a in nodes for b in adjacency[a] if a < b))


def skeleton(dag):
    return tuple(sorted(tuple(sorted(edge)) for edge in dag))


def moments(values):
    centered = values - values.mean(axis=0)
    cov = centered.T @ centered / len(values)
    sd = np.sqrt(np.diag(cov))
    return cov, cov / np.outer(sd,sd)


def fit_sample(values):
    cov,corr = moments(values)
    pc = pc_stable_gaussian(corr,len(values))
    score,dag = exact_gaussian_bic(cov,len(values))
    return pc,score,dag


def edge_string(edges, directed=False):
    sep = "->" if directed else "-"
    return ", ".join(f"{a}{sep}{b}" for a,b in edges)


def raw_ols_score(values, child, parents):
    design = np.column_stack([np.ones(len(values)), values[:,parents]]) if parents else np.ones((len(values),1))
    beta = np.linalg.lstsq(design,values[:,child],rcond=None)[0]
    residual = values[:,child] - np.einsum('ij,j->i',design,beta)
    sigma2 = float(np.sum(residual * residual) / len(values))
    assert np.isfinite(residual).all() and np.isfinite(sigma2) and sigma2 > 0
    return -.5 * len(values) * (np.log(2*np.pi) + 1 + np.log(sigma2)) - .5 * (len(parents)+2)*np.log(len(values))


def validate(groups):
    validations = []
    datasets = [groups[0][DOMAINS].to_numpy(), groups[1][DOMAINS].to_numpy()[:3000]]
    max_local_error = 0.0
    max_partial_error = 0.0
    for sample_id, values in enumerate(datasets):
        cov,corr = moments(values)
        scores = gaussian_local_scores(cov,len(values))
        for child in range(6):
            possible = [i for i in range(6) if i != child]
            for k in range(6):
                for parents in itertools.combinations(possible,k):
                    mask = sum(1<<i for i in parents)
                    expected = raw_ols_score(values,child,parents)
                    error = abs(scores[child,mask]-expected)
                    max_local_error = max(max_local_error,error)
                    assert error < 1e-8
        for i,j in itertools.combinations(range(6),2):
            possible = [k for k in range(6) if k not in (i,j)]
            for k in range(5):
                cond = tuple(possible[:k])
                design = np.column_stack([np.ones(len(values)), values[:,cond]]) if cond else np.ones((len(values),1))
                pair = values[:,[i,j]]
                beta = np.linalg.lstsq(design,pair,rcond=None)[0]
                residual = pair - np.einsum('ij,jk->ik',design,beta)
                assert np.isfinite(residual).all()
                expected = np.corrcoef(residual.T)[0,1]
                actual = partial_correlation(corr,i,j,cond)
                max_partial_error = max(max_partial_error,abs(actual-expected))
                assert abs(actual-expected) < 1e-10
        # Independent enumeration of every four-node DAG, scored by raw OLS.
        short = values[:,:4]
        cov4,_ = moments(short)
        pairs = list(itertools.combinations(range(4),2))
        best = -np.inf
        dag_count = 0
        independent_local = {(i,ps):raw_ols_score(short,i,ps) for i in range(4)
                             for k in range(4) for ps in itertools.combinations([j for j in range(4) if j != i],k)}
        for directions in itertools.product([0,1,2],repeat=6):
            edges = [pair if d == 1 else pair[::-1] for pair,d in zip(pairs,directions) if d]
            graph = nx.DiGraph();graph.add_nodes_from(range(4));graph.add_edges_from(edges)
            if not nx.is_directed_acyclic_graph(graph):
                continue
            dag_count += 1
            score = sum(independent_local[(i,tuple(sorted(graph.predecessors(i))))] for i in range(4))
            best = max(best,score)
        dp_score,dag = exact_gaussian_bic(cov4,len(short),nodes=DOMAINS[:4])
        assert dag_count == 543 and abs(dp_score-best)<1e-8
        graph=nx.DiGraph();graph.add_nodes_from(DOMAINS[:4]);graph.add_edges_from(dag)
        assert nx.is_directed_acyclic_graph(graph)
        validations.append(dict(sample=sample_id,n=len(values),enumerated_four_node_dags=dag_count,
                                dp_score=dp_score,enumerated_best_score=best))
    # Variable-order equivariance is checked for the PC skeleton, including all groups.
    for index,group in groups.items():
        values=group[DOMAINS].to_numpy();_,corr=moments(values)
        forward=pc_stable_gaussian(corr,len(values))
        backward=pc_stable_gaussian(corr[::-1,::-1],len(values),nodes=DOMAINS[::-1])
        assert forward==backward
        _,dag=exact_gaussian_bic(corr,len(values))
        graph=nx.DiGraph();graph.add_nodes_from(DOMAINS);graph.add_edges_from(dag)
        assert nx.is_directed_acyclic_graph(graph)
    result=dict(local_ols_scores_checked=384,max_local_score_absolute_error=max_local_error,
                partial_correlations_checked=150,max_partial_correlation_absolute_error=max_partial_error,
                exhaustive_dag_checks=validations,all_five_group_pc_order_checks_passed=True,
                all_five_group_bic_dags_acyclic=True)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def full_sample_results(groups):
    rows=[]
    for i,group in groups.items():
        pc,score,dag=fit_sample(group[DOMAINS].to_numpy())
        for method,edges in [('Gaussian PC-stable',pc),('Gaussian exact BIC',skeleton(dag))]:
            rows.append(dict(age_group=AGE_LABELS[i],original_group_index=i,n=len(group),method=method,
                             edge_count=len(edges),skeleton_edges=edge_string(edges),
                             bic_score=score if method=='Gaussian exact BIC' else np.nan,
                             one_optimal_dag=edge_string(dag,True) if method=='Gaussian exact BIC' else ''))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'full_sample_results.csv',index=False)
    print('FULL-SAMPLE CONTINUOUS-SCORE SENSITIVITY',flush=True)
    print(frame[['age_group','n','method','edge_count','skeleton_edges']].to_string(index=False),flush=True)
    return frame


def fit_group(task):
    i,group,reps,n=task
    source_ids=group.index.to_numpy(dtype=np.int64)
    rows=[]
    for rep in range(1,reps+1):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,i,rep]))
        ids=np.sort(rng.choice(source_ids,n,replace=False))
        pc,score,dag=fit_sample(group.loc[ids,DOMAINS].to_numpy())
        digest=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
        for method,edges in [('Gaussian PC-stable',pc),('Gaussian exact BIC',skeleton(dag))]:
            rows.append(dict(age_group=AGE_LABELS[i],original_group_index=i,replicate=rep,sample_n=n,
                             source_n=len(group),method=method,edge_count=len(edges),skeleton_edges=edge_string(edges),
                             bic_score=score if method=='Gaussian exact BIC' else np.nan,
                             one_optimal_dag=edge_string(dag,True) if method=='Gaussian exact BIC' else '',
                             sample_indices_sha256=digest))
        if rep%250==0:
            print(f'{AGE_LABELS[i]}: {rep}/{reps} draws',flush=True)
    return rows


def summarize(results):
    summary=[];frequencies=[]
    possible=sorted('-'.join(sorted(edge)) for edge in itertools.combinations(DOMAINS,2))
    for (label,method),sub in results.groupby(['age_group','method'],sort=False):
        values=sub.edge_count.to_numpy()
        lo,hi=np.quantile(values,[.025,.975],method='nearest')
        summary.append(dict(age_group=label,method=method,source_n=int(sub.source_n.iloc[0]),
                            sample_n=int(sub.sample_n.iloc[0]),fits=len(sub),mean_edges=values.mean(),
                            sd_edges=values.std(ddof=1),monte_carlo_se=values.std(ddof=1)/np.sqrt(len(sub)),
                            median_edges=np.median(values),percentile_2_5=lo,percentile_97_5=hi,
                            min_edges=int(values.min()),max_edges=int(values.max())))
        edge_sets=[set(value.split(', ')) if value else set() for value in sub.skeleton_edges]
        for edge in possible:
            count=sum(edge in edges for edges in edge_sets)
            frequencies.append(dict(age_group=label,method=method,edge=edge,times_selected=count,
                                    fits=len(sub),selection_fraction=count/len(sub)))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'matched_n3000_summary.csv',index=False)
    pd.DataFrame(frequencies).to_csv(OUT/'edge_selection_frequencies.csv',index=False)
    return summary


def contrasts_and_verification(results):
    """Descriptive cross-draw contrasts, with no inference about population peaks."""
    contrasts=[]; maxima=[]
    for method,sub in results.groupby('method'):
        wide=sub.pivot(index='replicate',columns='original_group_index',values='edge_count')
        for other in [1,3,4]:
            differences=wide[2]-wide[other]
            lo,hi=np.quantile(differences,[.025,.975],method='nearest')
            contrasts.append(dict(method=method,comparison=f'Early adulthood minus {AGE_LABELS[other]}',
                                  draws=len(wide),mean_edge_difference=differences.mean(),
                                  sd_edge_difference=differences.std(ddof=1),percentile_2_5=lo,percentile_97_5=hi,
                                  fraction_early_more_edges=(differences>0).mean(),
                                  fraction_tied=(differences==0).mean(),
                                  fraction_early_fewer_edges=(differences<0).mean()))
        competing=wide[[1,3,4]].max(axis=1)
        maxima.append(dict(method=method,draws=len(wide),
                           fraction_early_strict_maximum=(wide[2]>competing).mean(),
                           fraction_early_tied_or_strict_maximum=(wide[2]>=competing).mean()))
    pd.DataFrame(contrasts).to_csv(OUT/'descriptive_edge_count_contrasts.csv',index=False)
    pd.DataFrame(maxima).to_csv(OUT/'descriptive_early_maximum_fractions.csv',index=False)
    frequencies=pd.read_csv(OUT/'edge_selection_frequencies.csv')
    differences=[]
    for method,sub in frequencies.groupby('method'):
        wide=sub.pivot(index='edge',columns='age_group',values='selection_fraction')
        for edge,row in wide.iterrows():
            differences.append(dict(method=method,edge=edge,
                                    adolescent_selection_fraction=row[AGE_LABELS[1]],
                                    early_selection_fraction=row[AGE_LABELS[2]],
                                    early_minus_adolescent=row[AGE_LABELS[2]]-row[AGE_LABELS[1]]))
    pd.DataFrame(differences).to_csv(OUT/'edge_frequency_early_adolescent_contrasts.csv',index=False)
    for row in results.itertuples():
        edges=row.skeleton_edges.split(', ') if isinstance(row.skeleton_edges,str) and row.skeleton_edges else []
        assert len(edges)==row.edge_count and len(set(edges))==len(edges)
        if row.method=='Gaussian exact BIC':
            directed=[tuple(edge.split('->')) for edge in row.one_optimal_dag.split(', ')] if row.one_optimal_dag else []
            graph=nx.DiGraph();graph.add_nodes_from(DOMAINS);graph.add_edges_from(directed)
            assert nx.is_directed_acyclic_graph(graph) and len(directed)==row.edge_count
            assert edge_string(skeleton(directed))==row.skeleton_edges
    saved=pd.read_csv(OUT/'matched_n3000_summary.csv')
    for row in saved.itertuples():
        values=results.loc[results.age_group.eq(row.age_group)&results.method.eq(row.method),'edge_count']
        assert np.isclose(row.mean_edges,values.mean()) and np.isclose(row.sd_edges,values.std(ddof=1))
    discrete_file=ROOT/'outputs'/'original_matched_n3000_2000'/'replicate_results.csv'
    overlap_count=None
    if discrete_file.exists():
        previous=pd.read_csv(discrete_file)
        key=['original_group_index','replicate']
        old=previous[key+['sample_indices_sha256']].drop_duplicates(key)
        current=results[key+['sample_indices_sha256']].drop_duplicates(key)
        joined=current.merge(old,on=key,suffixes=('_continuous','_discrete'),validate='one_to_one')
        assert len(joined)==len(current)
        assert joined.sample_indices_sha256_continuous.eq(joined.sample_indices_sha256_discrete).all()
        overlap_count=len(joined)
    verification=dict(graph_edge_counts_verified=len(results),
                      bic_dags_acyclic_and_skeletons_verified=int(results.method.eq('Gaussian exact BIC').sum()),
                      all_summary_means_and_sds_independently_recomputed=True,
                      samples_identical_to_previous_discrete_n3000_draws=overlap_count,
                      contrasts='Descriptive conditional-on-source-data resampling comparisons; not population p-values.',
                      validation_arithmetic='Independent OLS predictions use einsum and squared residual sums; all finite. No BLAS matmul warnings in rerun.',
                      result_sha256=hashlib.sha256((OUT/'matched_n3000_replicate_results.csv').read_bytes()).hexdigest())
    (OUT/'result_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return pd.DataFrame(contrasts),pd.DataFrame(maxima)


def plot_results(full,summary,reps):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,method,color in zip(axes,['Gaussian PC-stable','Gaussian exact BIC'],['#245782','#9b4d20']):
        original=full[full.method.eq(method)].set_index('age_group').loc[AGE_LABELS[1:]]
        matched=summary[summary.method.eq(method)].set_index('age_group').loc[AGE_LABELS[1:]]
        x=np.arange(4)
        ax.plot(x,original.edge_count,'o--',color='#777777',label='Full original groups')
        ax.plot(x,matched.mean_edges,'o-',color=color,label=f'N = 3,000; {reps:,} draws')
        ax.vlines(x,matched.percentile_2_5,matched.percentile_97_5,color=color,alpha=.5)
        ax.set_title(method);ax.set_xticks(x)
        ax.set_xticklabels(['Adolescence\n13–19','Early adulthood\n20–39','Middle adulthood\n40–64','Older adulthood\n65–100'],fontsize=8)
        ax.set_ylim(-.3,15.8);ax.set_yticks(range(0,16,3));ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8,loc='lower center')
    axes[0].set_ylabel('Retained skeleton edges (maximum 15)')
    fig.suptitle('Continuous-score Gaussian sensitivity: original HEXACO domains',fontsize=13)
    fig.text(.07,.015,'Bars: empirical 2.5th–97.5th percentiles across subsamples, not confidence intervals.\nDifferent model assumptions from the original discretized analysis; childhood omitted from this matched-size comparison.',fontsize=8)
    fig.tight_layout(rect=[0,.10,1,.94]);fig.savefig(OUT/'mean_edge_counts.png',dpi=200);fig.savefig(OUT/'mean_edge_counts.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--full-only',action='store_true')
    args=parser.parse_args();started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    scored=pd.read_csv(SOURCE)
    expected=pd.cut(scored.age,[2,12,19,39,64,100],labels=AGE_LABELS).astype(str)
    assert expected.eq(scored.age_group).all() and scored[DOMAINS].notna().all().all()
    groups={i:scored.loc[scored.age_group.eq(label)] for i,label in enumerate(AGE_LABELS)}
    validation=validate(groups);full=full_sample_results(groups)
    if args.full_only:
        return
    rows=[]
    with ProcessPoolExecutor(max_workers=min(4,args.workers)) as pool:
        futures=[pool.submit(fit_group,(i,groups[i],args.reps,3000)) for i in range(1,5)]
        for future in as_completed(futures):rows.extend(future.result())
    results=pd.DataFrame(rows).sort_values(['original_group_index','replicate','method'])
    assert len(results)==args.reps*8
    results.to_csv(OUT/'matched_n3000_replicate_results.csv',index=False)
    summary=summarize(results);contrasts_and_verification(results);plot_results(full,summary,args.reps)
    metadata=dict(source_file=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sample_n=3000,
                  repetitions_per_group=args.reps,seed=SEED,
                  seed_rule='default_rng(SeedSequence([20261015, original_group_index, replicate]))',
                  sampling='Without replacement within a draw; repeats may overlap; all four nonchildhood groups sampled.',
                  models='Untransformed continuous scores; Gaussian PC-stable Fisher z alpha .05; exact Gaussian BIC subset DP.',
                  bic='Full OLS Gaussian log likelihood minus .5*(parents+2)*log(n) per node.',
                  interpretation='Alternative Gaussian model sensitivity, not a validation of discretized networks.',
                  validation=validation,runtime_seconds=time.monotonic()-started)
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report='''Continuous-score Gaussian BN sensitivity

All six original domain averages are used directly, without discretization or
transformation. Gaussian PC-stable uses partial correlations and Fisher z tests
at alpha .05. Exact Gaussian BIC maximizes the sum of Gaussian regression log
likelihoods minus half the parameter count times log(n); each local regression
counts an intercept, one slope per parent, and a residual variance. No
hyperparameters were chosen to recover the original age pattern. BIC DAG
directions are one optimum and must not be interpreted causally.

The five original age groups are first fitted in full, with childhood marked
as a descriptive N98 estimate. The matched-size comparison excludes childhood
and samples3000 from each of the other four groups, including older adulthood.
Both methods receive identical selected rows. Draws use the same original-row
seed rule as the N3000 discrete analysis. Repeated draws overlap.

This is an alternative model sensitivity requiring linear Gaussian conditional
relationships and approximately suitable Fisher-z inference. Domain scores are
bounded averages; multivariate normality is not established. Similar edge
counts do not imply similar edge identities or evidence for equal dependence
strength. Large samples can detect very small departures from independence.
Intervals describe subsampling variability, not confidence intervals for age
differences. More repetitions reduce simulation error, not model uncertainty.

Full-sample results
'''+full.to_string(index=False)+'\n\nMatched N3000 summary\n'+summary.to_string(index=False)
    (OUT/'methods_results.txt').write_text(report+'\n')
    print(summary.to_string(index=False),flush=True)
    print(f'Elapsed {time.monotonic()-started:.1f} seconds',flush=True)


if __name__=='__main__':
    main()
