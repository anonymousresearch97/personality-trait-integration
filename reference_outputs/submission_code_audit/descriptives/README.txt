Descriptive-statistics completion, 12 November 2026

Reviewer 1 minor 8 requested primary mean/SD/median/range/age distributions,
SAPA constituent age-band distributions, and sex/gender and domain means/SDs
by age group. All numerical tables for those requests are now supplied here.
These tables have not been inserted into a manuscript or Google document.

Primary main sample: ages 13-19 N=7,738; 20-39 N=51,670; 40-64 N=35,496;
65-100 N=3,915; total N=98,819. Childhood N=98 is in separate childhood_* files.
The source has 100,318 records: 1,373 missing age and 28 ages outside 3-100.
Source domain scores exactly equal means of the 16 stored keyed items per
domain in all 100,318 records. Those 96 items and six source scores are complete.
No additional reverse scoring or imputation was used. File completeness
cannot establish whether any imputation occurred before data release.

Primary sexnumeric labels are verified directly from SPSS SAV metadata:
1=men, 2=women. These replace the unlabeled codes in the older audit table.
SAPA retains source gender labels female/male. Percentages use every participant
in the specified age group as denominator; missing counts are explicit.

SAPA main sample: 18 and under N=4,550; 19-39 N=11,510; 40-59 N=2,253;
60 and over N=303; total N=18,616. Raw 1-6 responses were independently keyed,
averaged within each 40-item domain key, and mapped linearly to 1-5. At least
three answered keyed items in every domain and an observed public age band
were required. Raw reconstruction reproduced participant IDs, item counts,
and domain scores (maximum numerical discrepancy below 1e-14) in the cache.
SAPA means/SDs here are on that analyzed 1-5 scale. Item coverage is reported
separately; sparse item availability is not evidence of equivalent reliability.

Exact SAPA individual ages, mean age, median age, SD age, or observed age
range cannot be recovered from these public categories. No category midpoints
are substituted for ages. SAPA 18-and-under may include children; age 19 and
ages 60-64 are grouped differently than in the primary sample. The endpoint
bands are open-ended. The two sets of developmental categories are non-equivalent.

Contents:
primary_age_summary.csv: age means, sample SDs, medians, observed minima/maxima,
  quartiles and Ns for the four primary main groups and their combined sample.
primary_age_distribution.csv: single-year observed age frequencies by main group.
sapa_constituent_age_bands.csv: all eight constituent public age-band counts,
  with percentages within the pooled band and within the SAPA analytic sample.
domain_descriptives.csv: six domain means, sample SDs, medians, quartiles,
  observed ranges and missingness for each dataset/group and overall main sample.
domain_mean_sd_wide.csv: compact eight-row presentation of mean (SD) by group.
sex_gender_composition.csv: verified source labels, counts and percentages.
childhood_*.csv: all corresponding original childhood descriptions separately.
sapa_item_coverage.csv: answered-item counts across the four pooled/eight native bands.
sample_flow.csv and missingness_summary.csv: exclusions and missingness checks.
validation.json: versions, source hashes and reconstruction checks.

Language: Python (pandas/numpy/pyreadstat); no R or statistical network fitting
was needed for these descriptive tables. SDs are sample SDs (denominator N-1).

Data-deposition note: no explicit raw-data redistribution license was verified
from the local materials used in this audit. Package the code and aggregate
tables, and identify the original data sources; do not claim a public code
deposit, DOI, repository URL, or permission to redistribute participant-level
raw/cached data merely because this local package exists.
