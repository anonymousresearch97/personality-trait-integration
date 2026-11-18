"""Independent checks of saved association contrasts and narrower-age summaries.

Checks all saved bootstrap-array summaries and selected freshly reconstructed
participant draws. This is validation of the existing exploratory analysis.
"""
from pathlib import Path
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/longitudinal_peak_wrapup/original_age_detail'
OUT = ROOT / 'outputs/submission_code_audit/associations'
NODES = list('HEXACO')
PAIRS = list(itertools.combinations(range(6), 2))
GROUPS = ['13–19', '20–39', '40–64', '65–100']
SEED = 2026101521


def independently_calculated(values):
    centered = values - values.mean(axis=0)
    residual_correlations = []
    for i, j in PAIRS:
        controls = [k for k in range(6) if k not in (i, j)]
        z = centered[:, controls]
        # Two separate least-squares residual series; no precision-matrix formula.
        bi = np.linalg.lstsq(z, centered[:, i], rcond=None)[0]
        bj = np.linalg.lstsq(z, centered[:, j], rcond=None)[0]
        ri = centered[:, i] - np.einsum('ij,j->i', z, bi)
        rj = centered[:, j] - np.einsum('ij,j->i', z, bj)
        residual_correlations.append(np.dot(ri, rj) / np.sqrt(np.dot(ri, ri)*np.dot(rj, rj)))
    corr = np.corrcoef(values, rowvar=False)
    edges = np.array(residual_correlations)
    sign, logdet = np.linalg.slogdet(corr)
    assert sign == 1
    metrics = np.array([np.abs(edges).mean(), np.abs(corr[np.triu_indices(6, 1)]).mean(), -.5*logdet])
    return edges, metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / 'outputs/original_hexaco_scored_domains.csv')
    raw = raw.loc[raw.age.ge(13)].copy()
    raw['group'] = pd.cut(raw.age, [12, 19, 39, 64, 100], labels=GROUPS)
    arrays = np.load(SOURCE / 'group_bootstrap_arrays.npz')
    ed, md = arrays['edge_draws'], arrays['metric_draws']
    fe, fm = arrays['full_edges'], arrays['full_metrics']
    assert ed.shape == (4, 2000, 15) and md.shape == (4, 2000, 3)
    assert all(np.isfinite(v).all() for v in [ed, md, fe, fm])
    errors = []
    for gi, group in enumerate(GROUPS):
        values = raw.loc[raw.group.eq(group), NODES].to_numpy()
        edges, metrics = independently_calculated(values)
        np.testing.assert_allclose(edges, fe[gi], atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(metrics, fm[gi], atol=1e-12, rtol=1e-11)
        for rep in [0, 1, 999, 1999]:
            rng = np.random.default_rng(np.random.SeedSequence([SEED, gi, rep]))
            selected = rng.integers(0, len(values), len(values))
            edges, metrics = independently_calculated(values[selected])
            np.testing.assert_allclose(edges, ed[gi, rep], atol=1e-12, rtol=1e-11)
            np.testing.assert_allclose(metrics, md[gi, rep], atol=1e-12, rtol=1e-11)
            errors.append(dict(group=group, replicate_zero_based=rep,
                               maximum_absolute_error=float(np.max(np.abs(edges-ed[gi, rep])))))
    np.testing.assert_allclose(md[:, :, 0], np.abs(ed).mean(axis=2), atol=1e-14)
    strength = pd.read_csv(SOURCE / 'original_group_strengths.csv')
    metric_names = ['mean_absolute_partial_r', 'mean_absolute_r', 'gaussian_total_correlation']
    for gi, group in enumerate(GROUPS):
        for mi, metric in enumerate(metric_names):
            row = strength.loc[strength.age_group.eq(group) & strength.metric.eq(metric)].iloc[0]
            np.testing.assert_allclose([row.estimate, row.bootstrap_p2_5, row.bootstrap_p97_5],
                                      [fm[gi, mi], *np.quantile(md[gi, :, mi], [.025, .975])], atol=1e-12)
    obs = np.concatenate([fe[i+1]-fe[i] for i in range(3)])
    delta = np.column_stack([ed[i+1, :, j]-ed[i, :, j] for i in range(3) for j in range(15)])
    se = delta.std(axis=0, ddof=1)
    assert (se > 0).all()
    maxima = np.max(np.abs((delta-obs)/se), axis=1)
    critical = np.percentile(maxima, 95)
    contrasts = pd.read_csv(SOURCE / 'adjacent_edge_differences_simultaneous.csv')
    np.testing.assert_allclose(contrasts.difference, obs, atol=1e-12)
    np.testing.assert_allclose(contrasts.bootstrap_se, se, atol=1e-12)
    np.testing.assert_allclose(contrasts.simultaneous_lower, obs-critical*se, atol=1e-12)
    np.testing.assert_allclose(contrasts.simultaneous_upper, obs+critical*se, atol=1e-12)
    p = np.array([(1+np.count_nonzero(maxima >= abs(o/s)))/2001 for o, s in zip(obs, se)])
    np.testing.assert_allclose(contrasts.adjusted_bootstrap_p, p, atol=1e-12)
    detected = (obs-critical*se > 0) | (obs+critical*se < 0)
    assert np.array_equal(contrasts.excludes_zero, detected)
    broad = pd.read_csv(SOURCE / 'broad_adult_hump_contrasts.csv')
    for mi, metric in enumerate(metric_names):
        adult = .5*(fm[1, mi]+fm[2, mi])
        b_adult = .5*(md[1, :, mi]+md[2, :, mi])
        vals = [b_adult-md[0, :, mi], b_adult-md[3, :, mi]]
        est = [adult-fm[0, mi], adult-fm[3, mi]]
        vals.append(np.minimum(vals[0], vals[1])); est.append(min(est))
        for k, row in enumerate(broad.loc[broad.metric.eq(metric)].itertuples()):
            np.testing.assert_allclose([row.estimate,row.bootstrap_p2_5,row.bootstrap_p97_5,row.fraction_positive],
                                      [est[k],*np.quantile(vals[k],[.025,.975]),np.mean(vals[k]>0)],atol=1e-12)
    decade = pd.read_csv(SOURCE / 'decade_matched_draws.csv')
    summary = pd.read_csv(SOURCE / 'decade_matched_summary.csv')
    assert len(decade) == 7000 and not decade.duplicated(['age_band','replicate']).any()
    for row in summary.itertuples():
        values = decade.loc[decade.age_band.eq(row.age_band), row.metric].to_numpy()
        assert len(values) == 1000
        np.testing.assert_allclose([row.mean,row.sd,row.p2_5,row.p97_5],
                                  [values.mean(), values.std(ddof=1),*np.quantile(values,[.025,.975])],atol=1e-12)
    pd.DataFrame(errors).to_csv(OUT/'fresh_bootstrap_draw_checks.csv',index=False)
    report = dict(status='passed', full_group_regression_residual_checks=4,
                  fresh_participant_bootstrap_draw_checks=len(errors),
                  saved_bootstrap_draw_vectors_checked=8000, signed_contrasts_checked=45,
                  simultaneous_intervals_excluding_zero=int(detected.sum()),
                  by_adjacent_contrast=detected.reshape(3,15).sum(axis=1).tolist(),
                  max_t_critical=float(critical), decade_draw_rows=7000,
                  all_decade_summary_rows_checked=len(summary),
                  max_error_fresh_draws=max(r['maximum_absolute_error'] for r in errors),
                  scope='Fresh checks use 16 bootstrap draws; all stored arrays/summaries are checked. No full new 2000-draw bootstrap.',
                  assumptions='Independent participants; approximate centered fixed-SE bootstrap simultaneous intervals; scalar/metric equivalence not established.',
                  source_hash=hashlib.sha256((SOURCE/'group_bootstrap_arrays.npz').read_bytes()).hexdigest())
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
