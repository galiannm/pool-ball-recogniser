# Colour-score weighting diagnostics

Fixed K2 and frozen AllBalls HSV histograms; recognition-only changes. All ten ablations are reported. Only original White candidates receive padding. Query minimum affects every candidate, while cue/type weights apply globally. Training histograms, light thresholds (V170/S165), erosion3, detections and coordinates stay fixed.

| Method | Correct types | Correct identities | Unknown | Fixes / losses vs unpadded | Fixes / losses vs +1 |
|---|---:|---:|---:|---:|---:|
| unpadded_control | 76/86 | 75/86 | 1 | 0 / 0 | 0 / 4 |
| pad1_control | 79/86 | 79/86 | 1 | 4 / 0 | 0 / 0 |
| pad4_control | 79/86 | 79/86 | 1 | 5 / 1 | 1 / 1 |
| pad1_min6 | 79/86 | 79/86 | 1 | 4 / 0 | 0 / 0 |
| pad1_cue075 | 79/86 | 79/86 | 1 | 4 / 0 | 0 / 0 |
| pad1_cue050 | 79/86 | 79/86 | 1 | 4 / 0 | 0 / 0 |
| pad1_width015 | 78/86 | 77/86 | 1 | 4 / 2 | 0 / 2 |
| pad1_min6_cue075 | 79/86 | 79/86 | 1 | 4 / 0 | 0 / 0 |
| pad1_min6_width015 | 78/86 | 77/86 | 1 | 4 / 2 | 0 / 2 |
| pad1_min6_cue075_width015 | 78/86 | 77/86 | 1 | 4 / 2 | 0 / 2 |

## Table1 original White candidates

| Method | Candidate identities | All matched correct |
|---|---|---:|
| unpadded_control | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| pad1_control | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| pad4_control | #6 actual Cue -> 12; #7 actual 14 -> 14; | 10/14 |
| pad1_min6 | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| pad1_cue075 | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| pad1_cue050 | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| pad1_width015 | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 8/14 |
| pad1_min6_cue075 | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 10/14 |
| pad1_min6_width015 | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 8/14 |
| pad1_min6_cue075_width015 | #6 actual Cue -> 14; #7 actual 14 -> Cue; | 8/14 |

Cue score = cueWeight * lightFraction * Gaussian compatibility with [.85,1]. Numbered score = frozen histogram similarity * Gaussian compatibility with solid [0,.30] or stripe [.30,.85]. Width .10/.15 controls compatibility outside those intervals. Histogram similarity and light fraction have different empirical distributions; neither their product nor the final scores are calibrated probabilities.

Lowering the minimum from12 to6 only admits tiny query samples; no masks/model bins or training samples change. Such evidence may be cloth, shadows, a neighbour or real body colour. Cue weights .75/.50 rescale cue support for every candidate; uniqueness allows at most one cue, so this does not automatically resolve which candidate deserves it.

cue_counterfactuals.csv forces each original White candidate to claim cue in turn, reruns the full assignment, and measures total objective loss. This is a diagnostic, never a prediction rule. cue_swaps.csv gives feasible two-candidate cue swaps; negative swapGain explains why the current objective prefers the existing allocation. evidence.csv decomposes every identity score and rank. Predictions are fixed before evaluation annotations are loaded.

This is an evaluation-informed investigation of observed Table1 failures, including an explicit parameter comparison. Training models use only AllBalls, but choosing a winning configuration using Table1–6 is test-set method/parameter selection. The reported figures are exploratory, not an independent validation or a recommended deployed setting. AllBalls is training resubstitution and excluded from totals. Main pipeline unchanged.

Outputs: configuration.csv, metrics.csv, predictions.csv, changes.csv, evidence.csv, cue_counterfactuals.csv, cue_swaps.csv, frozen_models.yml, candidate_diagnostics/<image>/candidate_N.png, final_results/<method>/<image>/.
