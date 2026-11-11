# A/B/C/D identity heuristics

Shared K=2 features and frozen AllBalls HSV histograms. No K6/CCA search settings used. A is the existing classifier; B uses independent colour/type compatibility; C assigns identities jointly with missing identities and unlimited Unknowns; D adds weak source-radius consistency.

| Method | Correct types | Correct numbers | Unknown predictions | Extra duplicate identity claims |
|---|---:|---:|---:|---:|
| A: K2 + HSV | 76/86 | 72/86 | 4 | 10 |
| B: independent scores | 73/86 | 72/86 | 1 | 13 |
| C: unique identities | 72/86 | 69/86 | 8 | 0 |
| D: uniqueness + size | 72/86 | 69/86 | 8 | 0 |

| Comparison | Numbers corrected | Previously correct numbers lost |
|---|---:|---:|
| A: K2 + HSV -> B: independent scores | 1 | 1 |
| B: independent scores -> C: unique identities | 3 | 6 |
| C: unique identities -> D: uniqueness + size | 0 | 0 |
| A: K2 + HSV -> C: unique identities | 2 | 5 |
| A: K2 + HSV -> D: uniqueness + size | 2 | 5 |

Table1–6 totals exclude AllBalls, which is shown separately as training resubstitution. All stages retain all Hough candidates, including Unknown identities; detection remains 86 TP, 2 FP and 4 FN. Identity uniqueness is not physical circle deduplication. No candidate is removed just for overlapping another. False detections can still steal identities, and global assignment can turn duplicates into different incorrect numbers.

Type evidence uses Gaussian distance to the original K2 decision intervals, width .10; numbered scores multiply type and histogram support. Cue uses its light fraction as colour evidence. B chooses the best identity above .35. C/D allow only identities within .15 of that best and above .35. Hungarian assignment maximises total score with one column per identity and separate Unknown columns. These are fixed heuristic scores, not calibrated probabilities. Normalised relative supports in evidence.csv are diagnostic only.

Size comes from the inverse homography Jacobian in source pixels/mm and the known 43mm diameter. The major projected semiaxis approximates sphere radius; the plane ellipse is also shown in grey. A single median AllBalls detected/expected radius ratio, 0.972030, scales the prediction. No evaluation locations or radii fit this model. D multiplies each candidate's identity scores by .75+.25*exp(-.5*(log(radius/expected)/log(1.5))^2), leaving Unknown at .35. Invalid geometry is neutral. The planar model is approximate because balls extend above the cloth and camera intrinsics are unknown.

All weights were declared before this run, with no evaluation-based adjustment. The six images have been inspected in earlier development, so this is not pristine unseen-data validation. Choosing a stage after inspecting its scores is method selection, not a new independent test. The main pipeline is unchanged.

Outputs: configuration.txt; frozen_models.yml; training_geometry.csv; geometry_calibration.txt; metrics.csv; predictions.csv; evidence.csv (all 16 scores); type_scores.csv; geometry.csv; duplicate_claims.csv; changes.csv; confusion.csv; final_results/<stage>/<image>/; per-image geometry, candidate comparisons with score bars, and whole-table comparisons.
