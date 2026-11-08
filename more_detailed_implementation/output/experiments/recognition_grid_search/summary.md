# Recognition grid search

Tuning: Table1-6. Reference only: AllBalls. Zero radius padding; no CCA.

Stage 1 evaluated 1800 configurations. Stage 2 evaluated 5526 valid grouped combinations from 6 seeds over 2 passes. Repeated trials are retained.

Rank: correct numbers, correct types, oracle numbers, smaller baseline departure, lower trial ID. Best-type rank swaps the first two criteria. AllBalls never influences ranking.

| Configuration | Type (tuning) | Number (tuning) | Oracle number (tuning) | Number (reference) |
|---|---:|---:|---:|---:|
|baseline (98)|76/86|67/86|70/86|16/16|
|baseline_cloth (278)|77/86|69/86|72/86|15/16|
|best_type_grid (1510)|81/86|73/86|74/86|15/16|
|best_number (2153)|81/86|74/86|75/86|16/16|
|best_type (2153)|81/86|74/86|75/86|16/16|

These are exploratory tuning results, not independent test accuracy. The supplied images share the camera setup and some unchanged balls. Oracle-number accuracy supplies the true type, including the cue-ball shortcut; it is a diagnostic, not a deployable classifier. Unmatched candidates have N/A correctness and are excluded from recognition accuracy; missed balls remain in end-to-end recall's denominator.

Default experimental rules were checked against production for every cached segmentation. Production files and parameters were not changed.
