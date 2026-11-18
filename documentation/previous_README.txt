PERSONALITY TRAIT ORGANIZATION ACROSS ADOLESCENCE AND ADULTHOOD
Cross-sectional analysis code and verification package — 12 November 2026

STATUS AND SCOPE
This is a local, shareable code archive, not a deposited public repository.
No repository URL, DOI, code-license grant, raw-data redistribution permission,
or completed journal submission is implied. The original study was not
preregistered. The revision analyses are exploratory; settings were fixed for
the stated runs, not preregistered. No tuning is performed by this runner.

Included: primary Lee & Ashton HEXACO-100 and SAPA IPIP-HEXACO scoring,
categorical PC/BIC, continuous Gaussian PC/BIC, the fixed sparse comparison,
age-group descriptions, association contrasts/narrower age bands, measurement
checks, skeleton figures, CPDAG auditing, and independent numerical audits.
Longitudinal participant data, results and analysis drivers are not included.
The file longitudinal_gaussian_bn.py supplies mathematical helper functions
used in the cross-sectional age-detail and CPDAG code; its name is historical.
Likewise reference_outputs/longitudinal_peak_wrapup/original_age_detail/
contains ONLY cross-sectional primary-data analyses.

This archive does not contain the older 98/3,000/3,915 categorical resampling
experiments, kernel/rank experiments, or every abandoned exploratory branch.
Those remain in the working project. Categorical supporting results here are
the full-sample fits. The manuscript-relevant standard and sparse Gaussian
equal-N per-draw results and edge-selection frequencies are included.

SOFTWARE
Python 3.13.9 was used for the current audits. Main analysis requirements are
pinned in requirements.txt: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3,
matplotlib 3.10.6, networkx 3.5, pyreadstat 1.3.5.
Independent CPDAG conversion additionally uses causal-learn 0.1.4.8; install
requirements-cpdag.txt for that optional stage. The source algorithm used to
check resolved PC orientations is a Python port of the pcalg retry procedure,
NOT a run of R pcalg. See NOTICE.txt and the orientation audit columns.
The optional primary measurement models were fitted with R 4.5.3 and lavaan
0.7-3; all network fitting, descriptions, and plotting use Python.

Install into a suitable Python environment:
  python -m pip install -r requirements.txt
For optional CPDAG audit:
  python -m pip install -r requirements-cpdag.txt
For optional measurement audit, install R/lavaan separately and put Rscript
on PATH. The saved R session-info files document the environment used.

DATA REQUIRED (OBTAIN SEPARATELY)
Place these files at the indicated paths relative to this folder:
  Lee & Ashton 2018  online sample data.sav
  data_external/sapa/sapa_data.tab
  data_external/sapa/superKey696.csv
The primary filename contains TWO spaces between 2018 and online.
Input SHA-256 hashes are recorded in reference_outputs/submission_code_audit/
descriptives/validation.json. Source descriptions are Lee & Ashton (2018)
online HEXACO-100 data and the Condon/Revelle SAPA 696-item public release.
Participant-level raw files, scored caches, item response tables, participant
memberships and fitted .rds objects are intentionally absent from the archive.
Do not substitute the different OpenPsychometrics 240-item primary dataset.

The primary SAV stores already keyed 1-5 item responses and released domain
means: 16 items per domain, with four Altruism items excluded. Do not reverse
these stored primary items a second time. SAPA responses are 1-6; apply the
signed key, average available responses, linearly transform to 1-5, and retain
respondents with >=3 answered keyed items in EVERY domain and a public age band.
The preparation stage uses the same scoring functions as the audited analyses.

LAYOUT
analysis/: curated source copies. PACKAGING_CHANGES.json records every copy's
original/package hash and portability/documentation edits. Statistical model
settings and fitting logic were not changed for packaging.
reference_outputs/: immutable saved aggregate and per-draw results. These
include the fresh audits and descriptive tables. They are not participant data.
outputs/: created on first run. Missing files are initialized from saved
references to provide comparisons; each fitting stage replaces its own output
files with freshly computed results. Initialization is not a new analysis.
.runtime/: local plotting cache and optional measurement working data.

USE THE CONTROLLED RUNNER, NOT THE HISTORICAL MODULE MAIN FUNCTIONS
Some historical modules have extra experiments or legacy longitudinal main
functions. They are retained as source provenance/helpers; the runner calls
only the cross-sectional stages below and never runs longitudinal analyses.
Run commands from this folder. The runner also sets its own working directory.
Run Python normally, never with -O or -OO: numerical guards use assert and
optimized execution would disable these checks.

1. Create scores from the separately supplied raw inputs:
   python run_reproduction.py prepare
2. Fit full-sample categorical primary/SAPA models (no resampling):
   python run_reproduction.py categorical
3. Fit full-sample continuous Gaussian models (no resampling):
   python run_reproduction.py gaussian-full
4. Recompute the 1,000 equal-N draws per group and method:
   python run_reproduction.py gaussian-equal-n --workers 4
5. Generate the verified descriptive tables and categorical alpha checks:
   python run_reproduction.py descriptives
   python run_reproduction.py alpha
6. Run independent numerical checks of scores, full fits, all saved summaries
   and frequencies, and selected fresh equal-N and association draws:
   python run_reproduction.py audit
   To independently refit ALL standard equal-N draws instead of selected draws:
   python run_reproduction.py audit --all-draws
7. Redraw skeletons and the previously audited strict appendix panels from
   saved/recomputed network tables (no new fits required):
   python run_reproduction.py figures

Additional fixed revision analyses:
   python run_reproduction.py sparse
   python run_reproduction.py associations
   python run_reproduction.py measurement
The sparse stage uses PC alpha .01 and Gaussian BIC penalty multiplier 2,
with the SAME participant selections as the standard Gaussian equal-N runs.
After freshly running the sparse stage, its independent validator can be run:
   python analysis/validate_sparse_gaussian_comparison.py
That validator checks provenance hashes as well as numerical results; archived
pre-portability script hashes intentionally differ from these documented copies.
The association stage recomputes 2,000 participant bootstrap draws per primary
age group and 1,000 narrower-band equal-N draws; it can take substantially
longer than a full-sample fit. The measurement stage fits the six domain-wise
R models sequentially and then checks SAPA item/facet coverage.

CPDAG POLICY: TWO DISTINCT AUDITED OPTIONS
  python run_reproduction.py cpdag-strict
  python run_reproduction.py figures --cpdag-policy strict
Strict reconstruction retains the original tested collider constraints and
explicitly leaves incompatible PC outputs as skeletons. It does not invent
an equivalence class where no consistent DAG extension exists.

Optional alternative already audited separately:
  python run_reproduction.py cpdag-resolved
  python run_reproduction.py figures --cpdag-policy resolved
This applies the documented deterministic retry/extension policy to the four
cross-sectional PC conflicts: primary ages 40-64 and 65-100, SAPA ages 19-39
and 60+. SAPA 19-39 uses the stated fallback allowing an additional collider.
Every skeleton and edge count is preserved, but the equivalence class can
depend on this extra orientation policy. It is a methodological choice, not
just a display change. The strict output and resolved output are separate.
The 12 compatible strict panels receive independent causal-learn DAG-to-CPDAG
checks; the other four strict PC panels remain unresolved skeletons. Under the
optional resolved policy, all 16 resulting CPDAGs receive independent checks.
No bootstrap or subsample orientation-selection frequencies are estimated.
Main-paper skeletons are identical under either orientation policy.

KEY FIXED SETTINGS AND INTERPRETIVE LIMITS
Primary full-group Ns: 7,738 / 51,670 / 35,496 / 3,915 for ages 13-19 / 20-39 /
40-64 / 65-100. Historical childhood N=98 is separate from the main sample.
SAPA Ns: 4,550 / 11,510 / 2,253 / 303 for <=18 / 19-39 / 40-59 / >=60.
SAPA bands are categorical, non-equivalent to the primary bands, and open-ended
at the ends; no exact or midpoint-substituted participant ages are assumed.

Standard Gaussian PC: partial-correlation Fisher z tests, alpha .05, stable
adjacency sets within each conditioning depth. Gaussian BIC: exact subset
dynamic programming; local Gaussian regression log likelihood minus
0.5*(number of parents + 2)*log(N), counting intercept and residual variance.
One maximizing DAG is saved; its directions are not demonstrated causes.
Primary equal N=3,000 and SAPA equal N=300: 1,000 seeded draws per group,
without replacement WITHIN each draw; draws can overlap. These are subsamples,
not ordinary bootstraps. Both methods use the same sampled participants.
Primary seed 20261015 and SAPA seed 20261020; the full seed rules and hashes
are in the relevant provenance and per-draw files.

Categorical scores use five fixed equal-width intervals on 1-5, cut points
1.8/2.6/3.4/4.2 (right closed). Categorical PC uses conditional G-squared/MI
with nominal df=16*5^k, alpha .05. Exact discrete BIC counts 4*5^parents local
parameters. Sparse contingency tables can undermine asymptotic calibration;
reproducible computations do not remove this statistical limitation.
The alpha audit checks .01/.05/.10 without participant resampling and explains
the nonmonotonic PC result through changing available conditioning sets.

Edges are conditional-dependence structure estimates under the relevant model,
not simply pairwise correlations. A DAG-skeleton absence does NOT imply
independence conditional on all other variables; conditioning on colliders
can create dependence. The separate full-order partial-correlation summaries
condition on all other measured domains and are not the same estimand.

No result here establishes a within-person developmental trajectory, causal
direction, full measurement invariance, or a universal early-adult maximum.
Code execution and numerical consistency checks are evidence of scoped
computational correctness, not proof that every modeling assumption holds.

DESCRIPTIVE HANDOFF
reference_outputs/submission_code_audit/descriptives/descriptive_tables.html
is directly readable. The associated CSVs include all requested age summaries,
single-year primary frequencies, SAPA constituent age bands, verified sex/gender
labels and missingness, and domain means/SDs. Childhood is separate.
Primary SAV metadata verifies 1=men and 2=women; no guessed labels are used.
The numerical tables are complete; inclusion in the manuscript remains an
editing step outside this archive. The editorial checklist must be assessed
against the actual journal requirements and final manuscript.

PACKAGE EXECUTION VERIFICATION
The package was copied to a separate folder and executed against the separately
supplied raw inputs. Preparation, categorical fits, Gaussian full/equal-N fits,
sparse fits, strict/resolved CPDAGs, all-draw core audit, descriptions and figure
exports passed. Fresh standard/sparse per-draw skeletons and counts all match
the immutable archived references (32,018 result rows). R input preparation
and syntax checks passed; all 36 original saved CFA objects were rechecked,
but the full optional R fitting stage was not rerun in the copied package.
See package_portability_validation.json for exact scope.
