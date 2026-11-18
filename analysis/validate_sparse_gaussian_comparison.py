"""Independent validation of alpha=.01 PC and doubled-penalty Gaussian BIC.

The independent score uses raw-response least squares and exhaustive ordering
search, not the production covariance score or subset-DP implementation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'sparse_gaussian_comparison'
DOMAINS=['H','E','X','A','C','O']
PRIMARY_LABELS=['Childhood 3-12','Adolescence 13-19','Early adulthood 20-39','Middle adulthood 40-64','Older adulthood 65-100']
SAPA_LABELS=['18 and under','19-39','40-59','60 and over']
COLLAPSE={'18 and under':'18 and under','19-24':'19-39','25-29':'19-39','30-34':'19-39','35-39':'19-39',
          '40-49':'40-59','50-59':'40-59','60 and over':'60 and over'}


def independent_pc(values,alpha=.01):
    nodes=list(range(6));adj={i:set(nodes)-{i} for i in nodes}
    minimum_margin=math.inf;tests=0
    for level in range(5):
        frozen={i:set(adj[i]) for i in nodes}
        for x,y in itertools.combinations(nodes,2):
            if y not in adj[x]:continue
            candidates=set()
            for endpoint,opposite in [(x,y),(y,x)]:
                choices=frozen[endpoint]-{opposite}
                candidates.update(itertools.combinations(sorted(choices),level))
            for cond in sorted(candidates):
                design=np.column_stack([np.ones(len(values)),values[:,cond]]) if cond else np.ones((len(values),1))
                observed=values[:,[x,y]]
                slopes=np.linalg.lstsq(design,observed,rcond=None)[0]
                residuals=observed-np.einsum('ij,jk->ik',design,slopes)
                residuals-=residuals.mean(axis=0)
                cross=float(np.sum(residuals[:,0]*residuals[:,1]))
                r=cross/math.sqrt(float(np.sum(residuals[:,0]**2))*float(np.sum(residuals[:,1]**2)))
                assert np.isfinite(r)
                r=max(-1+np.finfo(float).eps,min(1-np.finfo(float).eps,r))
                statistic=abs(math.atanh(r))*math.sqrt(len(values)-len(cond)-3)
                p=math.erfc(statistic/math.sqrt(2))
                tests+=1;minimum_margin=min(minimum_margin,abs(p-alpha))
                if p>alpha:adj[x].remove(y);adj[y].remove(x);break
        if not any(len(neighbors)>=level+2 for neighbors in adj.values()):break
    edges=tuple(sorted(tuple(sorted((DOMAINS[x],DOMAINS[y]))) for x,y in itertools.combinations(nodes,2) if y in adj[x]))
    return edges,tests,minimum_margin


def raw_ols_local(values,child,parents,multiplier=2.):
    design=np.column_stack([np.ones(len(values)),values[:,parents]]) if parents else np.ones((len(values),1))
    response=values[:,child]
    slopes=np.linalg.lstsq(design,response,rcond=None)[0]
    residuals=response-np.einsum('ij,j->i',design,slopes)
    variance=float(np.sum(residuals**2)/len(values))
    assert np.isfinite(variance) and variance>0
    likelihood=-len(values)/2*(math.log(2*math.pi)+1+math.log(variance))
    penalty=multiplier/2*(len(parents)+2)*math.log(len(values))
    return likelihood-penalty


def independent_all_order_bic(values,multiplier=2.):
    local={}
    for child in range(6):
        possible=[node for node in range(6) if node!=child]
        for size in range(6):
            for parents in itertools.combinations(possible,size):
                local[(child,frozenset(parents))]=raw_ols_local(values,child,parents,multiplier)
    maxima={}
    for child in range(6):
        possible=[node for node in range(6) if node!=child]
        for size in range(6):
            for allowed in itertools.combinations(possible,size):
                maxima[(child,frozenset(allowed))]=max(local[(child,frozenset(parents))]
                    for k in range(size+1) for parents in itertools.combinations(allowed,k))
    score=max(sum(maxima[(child,frozenset(order[:position]))] for position,child in enumerate(order))
              for order in itertools.permutations(range(6)))
    return score,local


def digest(values):
    return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()


def edge_text(edges):return ', '.join(f'{x}-{y}' for x,y in edges)


def validate_selected(values,pc_row,bic_row,description):
    edges,test_count,min_margin=independent_pc(values,.01)
    assert edge_text(edges)==pc_row.skeleton_edges,(description,edge_text(edges),pc_row.skeleton_edges)
    optimal_score,local=independent_all_order_bic(values,2.)
    saved_score=bic_row.bic_score
    assert np.isfinite(saved_score) and abs(optimal_score-saved_score)<1e-8,(description,optimal_score,saved_score)
    dag=[tuple(edge.split('->')) for edge in bic_row.one_optimal_dag.split(', ')] if bic_row.one_optimal_dag else []
    selected_score=sum(local[(child,frozenset(DOMAINS.index(parent) for parent,node in dag if node==DOMAINS[child]))] for child in range(6))
    assert abs(selected_score-saved_score)<1e-8
    print(f'{description}: PC alpha=.01 matches; doubled-penalty BIC optimum/selected DAG raw-OLS match (error={abs(optimal_score-saved_score):.3g})',flush=True)
    return dict(description=description,n=len(values),independent_pc_alpha=.01,pc_skeleton_matches=True,
                independent_pc_tests=test_count,nearest_test_p_distance_to_alpha=min_margin,
                independent_bic_multiplier=2.,enumerated_variable_orders=720,local_raw_ols_scores=192,
                optimal_objective_absolute_difference=abs(optimal_score-saved_score),
                selected_dag_ols_objective_absolute_difference=abs(selected_score-saved_score))


def baseline(dataset,analysis):
    if dataset=='Primary':
        directory=ROOT/'outputs'/'original_continuous_bn_sensitivity'
        name='full_sample_results.csv' if analysis=='full' else 'matched_n3000_replicate_results.csv'
        frame=pd.read_csv(directory/name).fillna({'skeleton_edges':''})
        frame=frame.rename(columns={'original_group_index':'group_index','sample_indices_sha256':'source_row_indices_sha256'})
    else:
        directory=ROOT/'outputs'/'sapa_gaussian_wrapup'
        name='full_sample_results.csv' if analysis=='full' else 'matched_n300_replicate_results.csv'
        frame=pd.read_csv(directory/name).fillna({'skeleton_edges':''})
        if analysis=='full':
            frame=frame[frame.scheme.eq('pooled_four')].copy()
            frame['group_index']=frame.age_band.map({label:i for i,label in enumerate(SAPA_LABELS,1)})
    frame['method']=frame.method.map(lambda text:'PC' if 'PC' in text else 'BIC')
    if analysis=='full':frame['replicate']=0
    return frame.set_index(['group_index','replicate','method'])


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(ROOT/'outputs/original_hexaco_scored_domains.csv')
    sapa=pd.read_csv(ROOT/'outputs/sapa_hexaco_scored_domains.csv')
    assert pd.cut(primary.age,[2,12,19,39,64,100],labels=PRIMARY_LABELS).astype(str).eq(primary.age_group).all()
    assert sapa.RID.is_unique and sapa[[f'{d}_item_count' for d in DOMAINS]].ge(3).all().all()
    sapa['pooled_band']=sapa.age_band.map(COLLAPSE)
    groups={'Primary':{i:primary.loc[primary.age_group.eq(label)] for i,label in enumerate(PRIMARY_LABELS)},
            'SAPA':{i:sapa.loc[sapa.pooled_band.eq(label)] for i,label in enumerate(SAPA_LABELS,1)}}
    assert [len(group) for group in groups['Primary'].values()]==[98,7738,51670,35496,3915]
    assert [len(group) for group in groups['SAPA'].values()]==[4550,11510,2253,303]
    full=pd.read_csv(OUT/'full_sample_results.csv').fillna({'skeleton_edges':'','standard_skeleton_edges':'','one_optimal_dag':'','participant_ids_sha256':''})
    assert len(full)==18 and full.analysis.eq('full').all() and full.replicate.eq(0).all()
    matched_path=OUT/'matched_replicate_results.csv'
    matched=pd.read_csv(matched_path).fillna({'skeleton_edges':'','standard_skeleton_edges':'','one_optimal_dag':'','participant_ids_sha256':''}) if matched_path.exists() else pd.DataFrame()
    if len(matched):
        assert len(matched)==16000 and matched.analysis.eq('equal_n').all()
        assert matched.groupby(['dataset','group_index','method']).size().eq(1000).all()
    results=pd.concat([full,matched],ignore_index=True)
    assert not results.duplicated(['dataset','analysis','group_index','replicate','method']).any()
    baselines={(dataset,analysis):baseline(dataset,analysis) for dataset in groups for analysis in ['full','equal_n']}
    selected_ids={}
    for (dataset,analysis,index,rep),sub in results.groupby(['dataset','analysis','group_index','replicate']):
        index=int(index);rep=int(rep);group=groups[dataset][index]
        assert len(sub)==2 and set(sub.method)=={'PC','BIC'}
        expected_label=PRIMARY_LABELS[index] if dataset=='Primary' else SAPA_LABELS[index-1]
        assert sub.age_band.eq(expected_label).all() and sub.source_n.eq(len(group)).all()
        if analysis=='full':ids=group.index.to_numpy()
        else:
            n=3000 if dataset=='Primary' else 300
            seed=20261015 if dataset=='Primary' else 20261020
            assert 1<=index<=4 and 1<=rep<=1000
            rng=np.random.default_rng(np.random.SeedSequence([seed,index,rep]))
            ids=np.sort(rng.choice(group.index.to_numpy(),n,replace=False))
        assert sub.sample_n.eq(len(ids)).all() and len(np.unique(ids))==len(ids)
        assert sub.source_row_indices_sha256.eq(digest(ids)).all()
        if dataset=='SAPA':assert sub.participant_ids_sha256.eq(digest(group.loc[ids,'RID'])).all()
        if rep in [0,1]:selected_ids[(dataset,analysis,index,rep)]=ids
        for row in sub.itertuples():
            old=baselines[(dataset,analysis)].loc[(index,rep,row.method)]
            assert row.standard_edge_count==old.edge_count and row.standard_skeleton_edges==old.skeleton_edges
            if analysis=='equal_n':assert row.source_row_indices_sha256==old.source_row_indices_sha256
            edges=set(row.skeleton_edges.split(', ')) if row.skeleton_edges else set()
            assert len(edges)==row.edge_count and row.change==row.edge_count-row.standard_edge_count
            if row.method=='PC':assert row.alpha==.01
            else:
                assert row.penalty_multiplier==2.
                dag=[tuple(edge.split('->')) for edge in row.one_optimal_dag.split(', ')] if row.one_optimal_dag else []
                graph=nx.DiGraph();graph.add_nodes_from(DOMAINS);graph.add_edges_from(dag)
                assert nx.is_directed_acyclic_graph(graph)
                assert {'-'.join(sorted(edge)) for edge in dag}==edges
                assert row.edge_count<=row.standard_edge_count
    full_hash=hashlib.sha256((OUT/'full_sample_results.csv').read_bytes()).hexdigest()
    existing=json.loads((OUT/'independent_validation.json').read_text()) if (OUT/'independent_validation.json').exists() else {}
    checks=existing.get('independent_model_checks',[]) if existing.get('full_results_sha256')==full_hash else []
    completed={row['description'] for row in checks}
    report=dict(full_results_sha256=full_hash,full_graph_rows_verified=len(full),matched_graph_rows_verified=len(matched),
                all_participant_hashes_reconstructed=True,standard_comparator_counts_and_skeletons_verified=True,
                all_skeleton_counts_and_bic_dag_acyclicity_verified=True,
                independent_model_checks=checks,settings=dict(pc_alpha=.01,gaussian_bic_penalty_multiplier=2.))
    for (dataset,analysis,index,rep),ids in selected_ids.items():
        description=f'{dataset} {analysis} group{index} replicate{rep}'
        if description in completed:continue
        saved=results[results.dataset.eq(dataset)&results.analysis.eq(analysis)&results.group_index.eq(index)&results.replicate.eq(rep)].set_index('method')
        checks.append(validate_selected(groups[dataset][index].loc[ids,DOMAINS].to_numpy(),saved.loc['PC'],saved.loc['BIC'],description))
        report['independent_model_checks']=checks
        (OUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    comparison_path=OUT/'comparison.csv'
    if len(matched) and comparison_path.exists():
        comparison=pd.read_csv(comparison_path)
        keys=['dataset','analysis','group_index','age_band','method']
        assert len(comparison)==34
        for row in comparison.to_dict('records'):
            mask=np.ones(len(results),dtype=bool)
            for key in keys:mask &= results[key].eq(row[key]).to_numpy()
            sub=results.loc[mask];counts=sub.edge_count.to_numpy()
            lo,hi=np.quantile(counts,[.025,.975],method='nearest')
            assert row['fits']==len(sub)
            assert np.isclose(row['sparse_edges'],counts.mean()) and np.isclose(row['standard_edges'],sub.standard_edge_count.mean())
            assert np.isclose(row['change'],sub.change.mean()) and row['percentile_2_5']==lo and row['percentile_97_5']==hi
            assert np.isclose(row['fraction_decreased'],sub.change.lt(0).mean())
            assert np.isclose(row['fraction_increased'],sub.change.gt(0).mean())
            assert np.isnan(row['sd_edges']) if len(sub)==1 else np.isclose(row['sd_edges'],counts.std(ddof=1))
        frequencies=pd.read_csv(OUT/'edge_selection_frequencies.csv');assert len(frequencies)==240
        for row in frequencies.to_dict('records'):
            sub=matched[matched.dataset.eq(row['dataset'])&matched.group_index.eq(row['group_index'])&matched.method.eq(row['method'])]
            fraction=sum(row['edge'] in set(value.split(', ')) for value in sub.skeleton_edges if value)/len(sub)
            assert np.isclose(fraction,row['selection_fraction'])
        report.update(all34_comparison_rows_recomputed=True,all240_edge_frequencies_recomputed=True)
    provenance_path=OUT/'provenance.json'
    if len(matched) and provenance_path.exists():
        provenance=json.loads(provenance_path.read_text())
        assert provenance['settings']=={'pc_alpha':.01,'bic_penalty_multiplier':2.}
        # The archived result provenance fingerprints the original source files.
        # Portable copies have explicitly documented path/comment-only edits.
        mapping_file=ROOT/'PACKAGING_CHANGES.json'
        mappings=json.loads(mapping_file.read_text()) if mapping_file.exists() else []
        provenance_checks=[]
        for file,sha in provenance['files'].items():
            actual=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
            if actual==sha:
                mode='exact_original_hash'
            else:
                compatible=[m for m in mappings if m['file']==file and
                            m['original_sha256']==sha and m['packaged_sha256']==actual]
                assert compatible, (file, 'Unexplained provenance mismatch')
                mode='documented_portable_source_copy'
            provenance_checks.append(dict(file=file,verification=mode))
        report['provenance_checks']=provenance_checks
        report['all_provenance_checks_passed']=True
    report.update(independently_checked_sample_count=len(checks),validation_runtime_seconds=time.monotonic()-started,
                  validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  objective='Independent raw OLS: logL -2/2*(parents+intercept+residual_variance)*log(n), optimized over all720 variable orders.',
                  scope_note='One fixed authorized sensitivity. Multiplier2 is stronger than ordinary BIC; PC adaptive searches need not yield nested skeletons as alpha changes.')
    (OUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='independent_model_checks'},indent=2),flush=True)


if __name__=='__main__':main()
