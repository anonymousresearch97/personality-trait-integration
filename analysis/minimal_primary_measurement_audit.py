"""Minimal age-specific internal consistency and sample-description audit.

Use Python3.13 with pyreadstat. No network fitting or new modeling family.
The stored item responses reproduce the released domain means without
additional reverse-scoring. Altruism's four items are not domain items.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/minimal_revision_checks'
SOURCE = ROOT/'Lee & Ashton 2018  online sample data.sav'
CACHE = ROOT/'outputs/original_hexaco_scored_domains.csv'
DOMAINS = ['H','E','X','A','C','O']
SCORES = dict(zip(DOMAINS,['hones16','emoti16','extra16','agree16','consc16','openn16']))
LABELS = ['Childhood 3-12','Adolescence 13-19','Early adulthood 20-39',
          'Middle adulthood 40-64','Older adulthood 65-100']


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    raw, meta = pyreadstat.read_sav(str(SOURCE))
    raw_items = meta.column_names[2:102]
    assert len(raw_items)==100
    item_map = {d:[c for c in raw_items if c.startswith(d) and not c.startswith('Alt')] for d in DOMAINS}
    assert all(len(items)==16 for items in item_map.values())
    assert len(set(sum(item_map.values(),[])))==96
    errors = {}
    for domain, items in item_map.items():
        x = raw[items]
        assert x.notna().all().all() and x.ge(1).all().all() and x.le(5).all().all()
        error = float((x.mean(axis=1)-raw[SCORES[domain]]).abs().max())
        assert error==0
        errors[domain]=error
    missing_age = int(raw.user_age.isna().sum())
    outside = int((raw.user_age.notna() & ~raw.user_age.between(3,100)).sum())
    analytic = raw.loc[raw.user_age.between(3,100)].copy()
    analytic['age_group'] = pd.cut(analytic.user_age,[2,12,19,39,64,100],labels=LABELS)
    reconstructed = analytic[['user_age',*SCORES.values(),'age_group']].rename(
        columns={'user_age':'age',**{v:k for k,v in SCORES.items()}}).reset_index(drop=True)
    reconstructed.age_group = reconstructed.age_group.astype(str)
    cached = pd.read_csv(CACHE)
    pd.testing.assert_frame_equal(reconstructed,cached,check_exact=True)
    assert len(analytic)==98917 and missing_age==1373 and outside==28
    rows, descriptives, composition = [], [], []
    for age_group in LABELS:
        group = analytic.loc[analytic.age_group.eq(age_group)]
        age = group.user_age
        descriptives.append(dict(age_group=age_group,n=len(group),mean_age=age.mean(),sd_age=age.std(),
                                 median_age=age.median(),minimum_age=age.min(),maximum_age=age.max()))
        for code,n in group.sexnumeric.value_counts(dropna=False).items():
            composition.append(dict(age_group=age_group,sexnumeric_code=code,n=n,percent=100*n/len(group)))
        for domain,items in item_map.items():
            x = group[items].to_numpy(float)
            k = len(items)
            variance_sum = x.var(axis=0,ddof=1).sum()
            total_variance = x.sum(axis=1).var(ddof=1)
            alpha = k/(k-1)*(1-variance_sum/total_variance)
            covariance = np.cov(x,rowvar=False,ddof=1)
            alternative = k/(k-1)*(1-np.trace(covariance)/covariance.sum())
            assert np.isfinite(alpha) and abs(alpha-alternative)<1e-12
            score = group[SCORES[domain]]
            rows.append(dict(age_group=age_group,n=len(group),domain=domain,items=k,
                cronbach_alpha=alpha,domain_mean=score.mean(),domain_sd=score.std(),
                n_complete_items=len(group),alpha_covariance_check_abs_error=abs(alpha-alternative)))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'primary_age_reliability.csv',index=False)
    pd.DataFrame(descriptives).to_csv(OUT/'primary_age_descriptives.csv',index=False)
    pd.DataFrame(composition).to_csv(OUT/'primary_sex_code_composition.csv',index=False)
    analytic.loc[analytic.age_group.eq(LABELS[0]),'user_age'].value_counts().sort_index().rename_axis('age').reset_index(name='n').to_csv(OUT/'primary_childhood_age_counts.csv',index=False)
    table = result.pivot(index='age_group',columns='domain',values='cronbach_alpha').reindex(index=LABELS,columns=DOMAINS)
    note = '''Minimal primary measurement check — 10 November 2026

Computed raw-score Cronbach alpha separately by age and domain, using the
16 stored keyed items underlying each released domain mean. All 96 items
have complete responses in the file. Their unmodified means exactly match
all six released domain scores for every one of the100,318 source records.
No extra reverse-scoring or imputation was applied. Four interstitial
Altruism items were excluded. Filtering age3–100 exactly reproduces the
98,917-person analysis cache:1,373 missing ages and28 outside the range.

'''+table.to_string(float_format=lambda v:f'{v:.3f}')+'''

Alpha describes internal consistency of the stored item responses. It is
not evidence of unidimensionality, validity, invariant factor loadings,
equivalent response processes, or absence of differential measurement
error. No configural/metric invariance analysis was performed. Source-file
completion does not establish whether upstream imputation occurred.
These point estimates have no confidence intervals. The childhood N98
and use of an adult inventory remain limitations regardless of alpha.
Administration assistance is not established by this check.

Age summaries, sex-code counts (without invented code labels), domain
means/SDs and childhood single-year counts are saved alongside this note.
No network was rerun. All previous analyses and participant filters remain
unchanged. Primary reliability and SAPA item coverage address parts of
the reviewer requests; they do not resolve measurement comparability.
'''
    (OUT/'primary_reliability_note.txt').write_text(note)
    provenance = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        item_map=item_map,all_source_domain_reconstruction_errors=errors,
        analytic_cache_exactly_reproduced=True,source_n=len(raw),analytic_n=len(analytic),
        missing_age=missing_age,excluded_outside_age_range=outside,
        alpha_formula='k/(k-1)*(1-sum(item sample variances)/variance(item sum))',
        independent_covariance_formula_agrees=True,
        measurement_invariance_assessed=False)
    (OUT/'primary_measurement_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(table.to_string(float_format=lambda v:f'{v:.3f}'))
    print('Exact analytic sample and all domain scores reproduced; alpha formulas agree.')


if __name__=='__main__': main()
