"""Reproduce the categorical PC alpha check without sampling or parameter tuning.

Independent G-squared calculation uses count entropies. A separate, level-frozen
PC skeleton search is checked against the production NumPy implementation.
Input: the fixed primary and SAPA scored-domain tables. Outputs include every
CI test, level snapshots, original-statistic spot checks, and actual SAPA labels.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import platform
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/submission_code_audit/categorical_alpha'
DOMAINS = ['H', 'E', 'X', 'A', 'C', 'O']
ALPHAS = (.01, .05, .10)
PRIMARY_BANDS = ['Childhood 3-12', 'Adolescence 13-19', 'Early adulthood 20-39',
                 'Middle adulthood 40-64', 'Older adulthood 65-100']
SAPA_BANDS = ['18 and under', '19-39', '40-59', '60 and over']
SAPA_MAP = dict(zip(['18 and under', '19-24', '25-29', '30-34', '35-39',
                     '40-49', '50-59', '60 and over'],
                    [SAPA_BANDS[0]] + [SAPA_BANDS[1]]*4 + [SAPA_BANDS[2]]*2 + [SAPA_BANDS[3]]))


def discretize(group):
    out = pd.DataFrame({d: pd.cut(group[d], [1, 1.8, 2.6, 3.4, 4.2, 5.0000001],
                                labels=False, include_lowest=True) for d in DOMAINS})
    assert out.notna().all().all()
    return out.astype(int)


def independent_pc(values, alpha):
    """Return skeleton, complete test trace, and immutable per-level snapshots."""
    p = values.shape[1]
    @lru_cache(None)
    def entropy_term(columns):
        if not columns:
            c = np.array([len(values)])
        else:
            code = np.zeros(len(values), dtype=np.int64)
            for col in columns:
                code = code*5 + values[:, col]
            c = np.bincount(code, minlength=5**len(columns))
            c = c[c > 0]
        return float(np.sum(c*np.log(c)))
    def statistic(i, j, cond):
        g2 = 2*(entropy_term(tuple(sorted((i,j,*cond)))) + entropy_term(cond)
                -entropy_term(tuple(sorted((i,*cond)))) -entropy_term(tuple(sorted((j,*cond)))))
        df = 16*5**len(cond)
        return max(g2, 0.), df, float(chi2.sf(max(g2, 0.), df))
    adjacency = [set(range(p)) - {i} for i in range(p)]
    trace, stages = [], []
    def edges():
        return tuple(sorted(tuple(sorted((DOMAINS[i],DOMAINS[j])))
                            for i,j in itertools.combinations(range(p),2) if j in adjacency[i]))
    for k in range(p-1):
        snapshot = [frozenset(ns) for ns in adjacency]
        if not any(len(ns) > k for ns in snapshot):
            break
        stages.append(dict(level=k,phase='before',edge_count=len(edges()),edges=format_edges(edges())))
        for i,j in itertools.combinations(range(p),2):
            if j not in snapshot[i]:
                continue
            candidates = sorted(set(itertools.chain.from_iterable(
                itertools.combinations(sorted(snapshot[a]-{b}), k)
                for a,b in ((i,j),(j,i)))))
            for cond in candidates:
                g2,df,pvalue = statistic(i,j,cond)
                remove = pvalue > alpha
                trace.append(dict(level=k,x=DOMAINS[i],y=DOMAINS[j],
                    conditioning=','.join(DOMAINS[z] for z in cond),
                    g2=g2,df=df,pvalue=pvalue,removed=remove,
                    snapshot_x=','.join(sorted(DOMAINS[z] for z in snapshot[i]-{j})),
                    snapshot_y=','.join(sorted(DOMAINS[z] for z in snapshot[j]-{i}))))
                if remove:
                    adjacency[i].remove(j)
                    adjacency[j].remove(i)
                    break
        stages.append(dict(level=k,phase='after',edge_count=len(edges()),edges=format_edges(edges())))
    return edges(), trace, stages


def format_edges(edges):
    return ', '.join('-'.join(edge) for edge in edges)


def write_publication_companion(results):
    """Plot verified fits for adolescence/adulthood without refitting models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.3))
    specifications = [
        ('Primary', PRIMARY_BANDS[1:], ['13–19', '20–39', '40–64', '65–100']),
        ('SAPA', SAPA_BANDS, ['≤18', '19–39', '40–59', '≥60']),
    ]
    for ax, (dataset, bands, labels) in zip(axs, specifications):
        selections = {
            alpha: results.loc[results.dataset.eq(dataset) & results.alpha.eq(alpha)]
                          .set_index('age_group').loc[bands]
            for alpha in ALPHAS
        }
        # Identical count curves are combined explicitly; no values are jittered.
        assert np.array_equal(selections[.05].edge_count, selections[.10].edge_count)
        for alpha, marker, color, label in [
                (.01, 'o', '#245c82', 'α = 0.01'),
                (.05, 's', '#8b4d28', 'α = 0.05 / 0.10')]:
            ax.plot(range(4), selections[alpha].edge_count, marker=marker,
                    color=color, linewidth=1.8, markersize=6, label=label)
        ax.set_title(dataset)
        ax.set_xticks(range(4), labels)
        ax.set_ylim(-.25, 5.7)
        ax.set_yticks(range(6))
        ax.set_xlabel('Age group (years)')
        ax.set_ylabel('Number of skeleton edges')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, loc='upper right')
    fig.suptitle('Categorical PC: alpha sensitivity')
    fig.text(.02, .015, 'The α = 0.05 and α = 0.10 count curves coincide in both panels.', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(OUT/'categorical_alpha_adolescence_adulthood.png', dpi=300)
    fig.savefig(OUT/'categorical_alpha_adolescence_adulthood.svg')
    plt.close(fig)
    (OUT/'categorical_alpha_adolescence_adulthood_caption.txt').write_text(
        'Categorical PC-stable sensitivity to alpha (.01, .05, .10), using the full '
        'observed samples and the original five equal-width score categories. '
        'No bootstrap or subsampling was performed. Primary childhood is excluded. '
        'Actual SAPA age bands are shown; they are non-equivalent to the primary '
        'age groups, and the SAPA 18-and-under category can include children. '
        'The alpha .05 and .10 edge-count curves coincide and are drawn together. '
        'The separately retained categorical_alpha_verified figure includes primary '
        'childhood solely for the historical audit.\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from fast_pc_stable import pc_stable_edges as production_pc
    from original_hexaco_pc_bic import conditional_mi_pvalue_full_df as original_pvalue
    sources = [ROOT/'outputs/original_hexaco_scored_domains.csv',
               ROOT/'outputs/sapa_hexaco_scored_domains.csv']
    primary, sapa = (pd.read_csv(path) for path in sources)
    sapa['age_group'] = sapa.age_band.map(SAPA_MAP)
    assert len(primary) == 98917
    assert len(sapa) == 18616
    assert sapa[[d+'_item_count' for d in DOMAINS]].ge(3).all().all()
    rows, traces, stage_rows, test_checks = [], [], [], []
    for dataset, data, bands in [('Primary', primary, PRIMARY_BANDS), ('SAPA', sapa, SAPA_BANDS)]:
        for band in bands:
            group = data.loc[data.age_group.eq(band)]
            discrete = discretize(group)
            for alpha in ALPHAS:
                edges, trace, stages = independent_pc(discrete.to_numpy(), alpha)
                prod = production_pc(discrete+1, alpha)
                assert edges == prod, (dataset,band,alpha,edges,prod)
                rows.append(dict(dataset=dataset,age_group=band,n=len(group),alpha=alpha,
                                 edge_count=len(edges),edges=format_edges(edges),
                                 matches_production=True,childhood_exploratory=(band==PRIMARY_BANDS[0])))
                context = dict(dataset=dataset,age_group=band,alpha=alpha)
                traces.extend(dict(**context,**row) for row in trace)
                stage_rows.extend(dict(**context,**row) for row in stages)
                # Check one test at every reached conditioning depth using the
                # original pandas contingency calculation, including sparse tests.
                for level in sorted({t['level'] for t in trace}):
                    row = next(t for t in reversed(trace) if t['level']==level)
                    cond = tuple(row['conditioning'].split(',')) if row['conditioning'] else ()
                    old = original_pvalue(discrete+1,row['x'],row['y'],cond)
                    delta = abs(old-row['pvalue'])
                    assert delta < 1e-8, (dataset,band,alpha,delta)
                    test_checks.append(dict(**context,level=level,x=row['x'],y=row['y'],
                        conditioning=row['conditioning'],independent_pvalue=row['pvalue'],
                        original_pvalue=old,absolute_difference=delta))
                print(dataset, band, alpha, len(edges), format_edges(edges), flush=True)
    results = pd.DataFrame(rows)
    tests = pd.DataFrame(traces)
    stages = pd.DataFrame(stage_rows)
    results.to_csv(OUT/'categorical_alpha_results.csv',index=False)
    tests.to_csv(OUT/'all_ci_tests.csv',index=False)
    stages.to_csv(OUT/'stage_edge_counts.csv',index=False)
    pd.DataFrame(test_checks).to_csv(OUT/'original_statistic_checks.csv',index=False)
    primary_saved = pd.read_csv(ROOT/'outputs/four_figure_package_work/original_pc_sensitivity_compact.csv').fillna('')
    comparisons=[]
    for row in rows:
        if row['dataset']!='Primary':
            continue
        old=primary_saved.loc[primary_saved.specification.eq(f"equal_width_1_to_5; alpha={row['alpha']:g}") & primary_saved.age_group.eq(row['age_group'])].iloc[0]
        assert row['edges']==old.edges and row['edge_count']==old.edge_count
        comparisons.append(dict(**row,saved_count=int(old.edge_count),saved_edges=old.edges,match=True))
    pd.DataFrame(comparisons).to_csv(OUT/'historical_primary_comparison.csv',index=False)
    metadata = dict(language='Python',python=sys.version,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        no_resampling=True,alpha_values=list(ALPHAS),all_independent_skeletons_match_production=True,
        maximum_spotcheck_pvalue_error=max(x['absolute_difference'] for x in test_checks),
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        independent_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        historical_SAPA_alpha_table_found=False)
    (OUT/'validation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    text = ['Categorical alpha verification: fixed full samples, no resampling.',
            'Fresh independent G-squared/PC-stable skeleton fits agree with the production implementation in all 27 fits.',
            'All 15 primary alpha fits also agree exactly with the archived primary alpha table.',
            'The separate historical SAPA alpha figure/table was not found. The rerun verifies the reviewer-described nonmonotonic pattern, not pixel identity of that unavailable figure.',
            '',results[['dataset','age_group','n','alpha','edge_count','edges']].to_string(index=False),
            '', 'Interpretation: PC-stable freezes adjacency sets within each conditioning depth, but those sets still depend on the chosen alpha. A higher alpha can retain more edges at an early depth, allowing additional conditioning sets later; those later tests can remove more edges. Final edge counts therefore need not be monotonic in alpha.',
            'Concrete SAPA 19-39 example: at alpha .01, A-C is removed at conditioning depth 2 (given H,O; p=.038637), leaving neither A nor H with three eligible neighbors for the later H-A test. At alpha .05, A-C survives depth 2, so H-A given E,X,C is eligible at depth 3 (p=.063663) and is removed. The graph has 9 versus 11 edges after depth 2, then 2 versus 0 after depth 3. This directly reproduces and explains the counterintuitive final counts.',
            'The five-state conditional G-squared tests use the full nominal degrees of freedom, 16*5^k. With sparse contingency tables (especially SAPA and large conditioning sets), this asymptotic approximation can be poorly calibrated and favor edge deletion. Reproducing the computation does not validate those distributional assumptions or establish an age effect.',
            'The primary childhood group is retained only for this audit of the old figure; it remains exploratory and is excluded from the revised adolescence/adulthood main analysis.',
            'The SAPA childhood group is unavailable and is not plotted as a fitted zero-edge graph.',
            '',f"Python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}; scipy {scipy.__version__}."]
    (OUT/'audit_report.txt').write_text('\n'.join(text)+'\n')
    # A freshly regenerated figure can replace the unavailable archived alpha
    # figure. Actual public age bands are displayed for SAPA.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(10.8,4.3))
    for ax,(dataset,bands) in zip(axs,[('Primary',PRIMARY_BANDS),('SAPA',SAPA_BANDS)]):
        for alpha,marker,color in zip(ALPHAS,['o','s','^'],['#245c82','#8b4d28','#47835b']):
            sub=results.loc[results.dataset.eq(dataset)&results.alpha.eq(alpha)].set_index('age_group').loc[bands]
            ax.plot(range(len(bands)),sub.edge_count,marker=marker,color=color,label=f'α = {alpha:.2f}')
        ax.set_title(dataset)
        labels=['3–12*','13–19','20–39','40–64','65–100'] if dataset=='Primary' else ['≤18','19–39','40–59','≥60']
        ax.set_xticks(range(len(bands)),labels)
        ax.set_ylim(-.3,8.5)
        ax.set_yticks(range(0,9))
        ax.set_xlabel('Age group (years)')
        ax.set_ylabel('Number of skeleton edges')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle('Categorical PC: alpha sensitivity')
    fig.text(.02,.01,'* Childhood is shown only to verify the historical analysis; SAPA age categories are non-equivalent.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(OUT/'categorical_alpha_verified.png',dpi=250)
    fig.savefig(OUT/'categorical_alpha_verified.svg')
    plt.close(fig)
    write_publication_companion(results)


if __name__ == '__main__':
    main()
