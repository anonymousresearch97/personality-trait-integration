"""Consolidate the fixed measurement audit; no model estimation or selection."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'measurement_resolution'
NAMES = dict(H='Honesty–Humility', E='Emotionality', X='Extraversion',
             A='Agreeableness', C='Conscientiousness', O='Openness to Experience')


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_fits, summary = [], []
    reliability = read(ROOT / 'outputs/minimal_revision_checks/primary_age_reliability.csv')
    for domain, name in NAMES.items():
        fits = read(OUT / f'primary_{domain}_fit_summary.csv')
        assert len(fits) == 6
        assert all(x['converged'] == 'TRUE' and x['admissible'] == 'TRUE' for x in fits)
        all_fits.extend(fits)
        baseline = [x for x in fits if x['setting'] == 'single_configural']
        config = next(x for x in fits if x['setting'] == 'multigroup_configural')
        metric = next(x for x in fits if x['setting'] == 'multigroup_metric')
        change = read(OUT / f'primary_{domain}_metric_comparison.csv')[0]
        failed = [x['age_group'] for x in baseline
                  if float(x['cfi.robust']) < .90 or float(x['rmsea.robust']) > .08
                  or float(x['srmr']) > .08]
        small_change = (float(change['delta_cfi_robust']) >= -.010
                        and float(change['delta_rmsea_robust']) <= .015
                        and float(change['delta_srmr']) <= .030)
        alpha = [float(x['cronbach_alpha']) for x in reliability
                 if x['domain'] == domain and not x['age_group'].startswith('Childhood')]
        row = dict(domain=domain, domain_name=name, alpha_min=min(alpha), alpha_max=max(alpha),
                   group_configural_cfi_min=min(float(x['cfi.robust']) for x in baseline),
                   group_configural_cfi_max=max(float(x['cfi.robust']) for x in baseline),
                   group_configural_rmsea_max=max(float(x['rmsea.robust']) for x in baseline),
                   group_configural_srmr_max=max(float(x['srmr']) for x in baseline),
                   every_age_configural_screen_pass=not failed,
                   groups_failing_configural_screen='; '.join(failed),
                   joint_configural_cfi=float(config['cfi.robust']),
                   joint_metric_cfi=float(metric['cfi.robust']),
                   delta_cfi_robust=float(change['delta_cfi_robust']),
                   delta_rmsea_robust=float(change['delta_rmsea_robust']),
                   delta_srmr=float(change['delta_srmr']),
                   metric_change_within_descriptive_screen=small_change,
                   interpretation=('Acceptable baseline under specified screen; small metric changes; exact equality rejected'
                                   if not failed else 'Baseline screen not met in every age group; small metric changes do not establish invariance'))
        summary.append(row)
    assert {r['domain'] for r in summary if r['every_age_configural_screen_pass']} == {'H', 'A'}
    assert all(r['metric_change_within_descriptive_screen'] for r in summary)
    write('primary_measurement_summary.csv', summary)
    write('primary_all_fit_indices.csv', all_fits)
    write('primary_main_age_reliability.csv', [r for r in reliability if not r['age_group'].startswith('Childhood')])
    write('primary_joint_model_group_ids.csv', [
        dict(numeric_group_id=1, age_group='Early adulthood 20-39', n=51670),
        dict(numeric_group_id=2, age_group='Middle adulthood 40-64', n=35496),
        dict(numeric_group_id=3, age_group='Adolescence 13-19', n=7738),
        dict(numeric_group_id=4, age_group='Older adulthood 65-100', n=3915)])
    for row in summary:
        print(f"{row['domain']}: CFI {row['group_configural_cfi_min']:.4f}–{row['group_configural_cfi_max']:.4f}; "
              f"delta {row['delta_cfi_robust']:.4f}; baseline pass {row['every_age_configural_screen_pass']}")


if __name__ == '__main__':
    main()
