# Direct HSV pixel proportions versus K2 type evidence

HSV profiles contain no K-means type/trigger stage. Light/dark/coloured fractions are counted directly using the existing thresholds, without tuning. Frozen AllBalls HSV histogram comparison supplies colour evidence.

| Method | Correct types | Correct identities | Correct cues | Unknown | Fixes / losses vs paired K2 |
|---|---:|---:|---:|---:|---:|
| K2_local | 76/86 | 72/86 | 6/6 | 4 | — |
| HSV_local | 78/86 | 74/86 | 6/6 | 3 | 3 / 1 |
| K2_unique | 76/86 | 75/86 | 3/6 | 1 | — |
| HSV_unique | 76/86 | 74/86 | 3/6 | 1 | 1 / 2 |
| K2_pad1_unique | 79/86 | 79/86 | 5/6 | 1 | — |
| HSV_pad1_unique | 79/86 | 78/86 | 5/6 | 1 | 1 / 2 |
| K2_pad1_largest_cue | 80/86 | 80/86 | 6/6 | 2 | — |
| HSV_pad1_largest_cue | 80/86 | 79/86 | 6/6 | 2 | 1 / 2 |

Light: V>=170 and S<=165. Dark: V<=90 outside light. Coloured: all remaining pixels. Masks partition the complete extraction disk; type fractions have no erosion. White>=85% light; solid>=70% nonlight; otherwise stripe when both occur. The independent histogram body mask still uses its frozen erosion3/light exclusion/min12 pixels. Histogram scores are 1-Bhattacharyya distance; backprojection images are not the classification score.

Each frontend's original White labels control its own padding and ambiguous-cue trigger. Thus replacing type evidence may also change which crops expand and whether a cue is reserved. Original circles, coordinates and detections remain fixed. The row model is copied unchanged from the previous AllBalls-only fit.

Unique scores use the same soft type intervals/width.10 and Hungarian assignment; largest-cue variants reserve cue for the greatest original radius/expected(row) among original White candidates only when >=2 exist. No threshold sweep, new histogram training or production integration. Table1–6 have been inspected; these are exploratory comparisons. AllBalls is training resubstitution, excluded from totals.

Outputs: final_results/<method>/<image>/; candidate_comparisons/<image>/candidate_N.png; pixel_masks/<image>/candidate_N.png; mask_counts.csv; predictions.csv; metrics.csv; changes.csv; configuration.txt; frozen_models.yml; frozen_row_model.yml; synthetic_checks.txt.
