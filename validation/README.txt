Student bundle verification, 17 November 2026

The ZIP was extracted to a new folder containing spaces. Commands were run
from an unrelated working directory, using only data inside the extracted
bundle. No symlinks to the working project were used.

Passed: input integrity and dimensions; all 8,000 standard equal-N samples
refitted with PC and BIC (16,000 fits); independent full-sample network and
association checks; sparse-reference validation with 17 independent fits;
all 36 R/lavaan CFA models refitted, with exact agreement in all archived fit
statistics/metric comparisons; saved-R-object audits before/after refitting.

A second copy had primary_items.csv deleted. prepare-measurement regenerated
an identical CSV, and the initial input integrity checks passed again.

Python 3.13.9; R 4.5.3 / lavaan 0.7-3. Dependencies are installed separately.
Machine-specific test-environment libraries are not included in this ZIP.
The modeling choices and numerical results have not been changed.
