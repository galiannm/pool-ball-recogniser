# Row-based size support for competing cue claims

One row-only radius model is fitted from 16 AllBalls Hough detections, without labels: expectedRadius=max(1,20.274191+0.009677*originalImageRow). Robust Theil-Sen slope uses pairs >=10 rows apart, floored at0; intercept is the median residual. Every horizontal position uses the same expected radius at a given row.

The training row span is 143.250000..197.250000; rows outside that span use linear extrapolation, not interpolation. Hough radius bias/partial crops can bias the fit. Horizontal perspective variation and sphere height are ignored. The mean absolute training residual is 1.037298 pixels.

Size scoring activates only with >=2 original K2 White candidates (3 evaluation images). Then every candidate's cue score is multiplied by min(1,originalRadius/expectedRadius)^2, an apparent-area support heuristic. Numbered scores are unchanged. Applying this to all cue claims avoids merely handing cue to another soft non-White candidate. Oversized circles get no bonus. No thresholds were selected from this run.

| Method | Correct types | Correct identities | Unknown | Fixes/losses vs C | Fixes/losses vs +1 |
|---|---:|---:|---:|---:|---:|
| C_unique | 76/86 | 75/86 | 1 | 0/0 | 0/4 |
| white_pad1_unique | 79/86 | 79/86 | 1 | 4/0 | 0/0 |
| row_cue_unique | 79/86 | 78/86 | 4 | 3/0 | 1/2 |
| pad1_row_cue_unique | 79/86 | 79/86 | 1 | 4/0 | 0/0 |
| row_largest_cue | 79/86 | 78/86 | 4 | 3/0 | 1/2 |
| pad1_row_largest_cue | 80/86 | 80/86 | 2 | 5/0 | 1/0 |

| Table1 method | Original White candidate identities | All matched correct |
|---|---|---:|
| C_unique | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| white_pad1_unique | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| row_cue_unique | #6 actual Cue -> Cue; #7 actual 14 -> Unknown; | 11/14 |
| pad1_row_cue_unique | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| row_largest_cue | #6 actual Cue -> Cue; #7 actual 14 -> Unknown; | 11/14 |
| pad1_row_largest_cue | #6 actual Cue -> Cue; #7 actual 14 -> Unknown; | 11/14 |

The two largest-cue ablations test the stronger hypothesis directly: reserve cue for the original White candidate with greatest radius/expectedRadius and forbid every other cue claim. Ties keep the first candidate. The selected candidate is forced to cue; this can be wrong when its detection is inflated or a real cue was not originally White. This is compared separately from the soft area-support prior, with no selection from evaluation scores.

All methods share original K2, frozen AllBalls HSV histograms and best-effort unique assignment. Padding1 expands only originally White recognition ROIs, while size uses original Hough radii/rows. Detection geometry, count and matching are fixed. AllBalls results are training resubstitution, excluded from totals. The main pipeline is unchanged. These are evaluation-informed exploratory comparisons, not independent validation.

Outputs: row_model.yml, training_samples.csv, row_radius.csv, row_radius_model.png, configuration.txt, cue_size_scores.csv, predictions.csv, metrics.csv, changes.csv, candidate_diagnostics/<image>/candidate_N.png, final_results/<method>/<image>/, frozen_models.yml and synthetic_checks.txt.
