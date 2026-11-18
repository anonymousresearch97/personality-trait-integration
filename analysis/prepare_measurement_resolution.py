"""Prepare verified keyed items for a fixed lavaan measurement audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/measurement_resolution'
TEMP=ROOT/'data_prepared'
SOURCE=ROOT/'Lee & Ashton 2018  online sample data.sav'
DOMAINS=['H','E','X','A','C','O']
SCORES=dict(zip(DOMAINS,['hones16','emoti16','extra16','agree16','consc16','openn16']))
LABELS=['Adolescence 13-19','Early adulthood 20-39','Middle adulthood 40-64','Older adulthood 65-100']


def main():
    OUT.mkdir(exist_ok=True,parents=True); TEMP.mkdir(exist_ok=True,parents=True)
    raw,meta=pyreadstat.read_sav(str(SOURCE))
    item_names=meta.column_names[2:102]
    records=[]
    for domain in DOMAINS:
        cols=[c for c in item_names if c.startswith(domain) and not c.startswith('Alt')]
        assert len(cols)==16
        assert np.array_equal(raw[cols].mean(axis=1),raw[SCORES[domain]])
        facets=list(dict.fromkeys(c.rstrip('0123456789') for c in cols))
        assert len(facets)==4
        for i,facet in enumerate(facets,1):
            facet_items=[c for c in cols if c.rstrip('0123456789')==facet]
            assert len(facet_items)==4
            for j,item in enumerate(facet_items,1):
                records.append(dict(domain=domain,facet=facet,facet_index=i,item=item,item_index=j,
                                    column_name=f'{domain}{i}{j}'))
    mapping=pd.DataFrame(records);mapping.to_csv(OUT/'primary_item_facet_map.csv',index=False)
    selected=raw.loc[raw.user_age.between(13,100)].copy()
    selected['age_group']=pd.cut(selected.user_age,[12,19,39,64,100],labels=LABELS).astype(str)
    cache=pd.read_csv(ROOT/'outputs/original_hexaco_scored_domains.csv')
    cache=cache.loc[cache.age.ge(13)].reset_index(drop=True)
    recon=selected[['user_age',*SCORES.values(),'age_group']].rename(columns={'user_age':'age',**{v:k for k,v in SCORES.items()}}).reset_index(drop=True)
    pd.testing.assert_frame_equal(recon,cache,check_exact=True)
    assert len(selected)==98819
    assert selected.groupby('age_group').size().reindex(LABELS).tolist()==[7738,51670,35496,3915]
    data=selected[mapping.item.tolist()].astype(int).rename(columns=dict(zip(mapping.item,mapping.column_name)))
    data['age_group']=selected.age_group
    data.to_csv(TEMP/'primary_items.csv',index=False)
    for domain in DOMAINS:
        d=mapping.loc[mapping.domain.eq(domain)]
        lines=[]
        for idx,g in d.groupby('facet_index',sort=True):
            lines.append(f'f{idx} =~ '+' + '.join(g.column_name))
        lines.append('D =~ f1 + f2 + f3 + f4')
        lines.extend(['f1 ~~ 0*f2 + 0*f3 + 0*f4','f2 ~~ 0*f3 + 0*f4','f3 ~~ 0*f4'])
        (OUT/f'primary_{domain}_model.lav').write_text('\n'.join(lines)+'\n')
    metadata=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  cache_sha256=hashlib.sha256((ROOT/'outputs/original_hexaco_scored_domains.csv').read_bytes()).hexdigest(),
                  data_n=len(data),age_groups=LABELS,group_ns=[7738,51670,35496,3915],
                  all96_items_reproduce_released_domains=True,nonchild_cache_exactly_reconstructed=True,
                  input_location=str(TEMP/'primary_items.csv'),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'primary_preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Prepared verified96items for98,819participants; no extra reversing or imputation.')


if __name__=='__main__':main()
