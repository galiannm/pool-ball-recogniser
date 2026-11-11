# Best-effort A/B/C/D identity heuristics

Shared K=2 features and frozen AllBalls HSV histograms. No K6/CCA search settings used. A is the existing classifier; B uses independent colour/type compatibility; C assigns identities jointly with missing identities and unlimited Unknowns; D adds weak source-radius consistency.

| Method | Correct types | Correct numbers | Unknown predictions | Extra duplicate identity claims |
|---|---:|---:|---:|---:|
| A: K2 + HSV | 76/86 | 72/86 | 4 | 10 |
| B: independent scores | 74/86 | 73/86 | 0 | 13 |
| C: unique identities | 76/86 | 75/86 | 1 | 0 |
| D: uniqueness + size | 76/86 | 75/86 | 1 | 0 |

| Comparison | Numbers corrected | Previously correct numbers lost |
|---|---:|---:|
| A: K2 + HSV -> B: independent scores | 2 | 1 |
| B: independent scores -> C: unique identities | 5 | 3 |
| C: unique identities -> D: uniqueness + size | 0 | 0 |
| A: K2 + HSV -> C: unique identities | 6 | 3 |
| A: K2 + HSV -> D: uniqueness + size | 6 | 3 |

Only the abstention policy changes relative to the original experiment. B gives the best supported guess; C/D permit every supported identity, Unknown=0, no .15 shortlist. Values <=1e-9 are numerical zero. A candidate still needs usable body-colour evidence or a baseline White classification. Uniqueness can require Unknown even with usable evidence, including when there are more than 16 candidates.

| Stage | Conservative correct | Best-effort correct | Conservative Unknown | Best-effort Unknown | Corrections | Regressions | Low-support guesses |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: K2 + HSV | 72 | 72 | 4 | 4 | 0 | 0 | 1 |
| B: independent scores | 72 | 73 | 1 | 0 | 1 | 0 | 1 |
| C: unique identities | 69 | 75 | 8 | 1 | 6 | 0 | 7 |
| D: uniqueness + size | 69 | 75 | 8 | 1 | 6 | 0 | 7 |

Controls were recomputed from the same cached candidates/scores, not refitted. confidence.csv flags selected scores <=.35 or alternatives farther than .15 from local best (D uses its weighted score for the .35 check). Amber rings in confidence_overlay.png and ? in candidate comparisons mark these guesses without replacing their identity. Flags are heuristics, not calibrated probabilities. policy_comparison.csv gives every conservative/current decision.

Table1–6 totals exclude AllBalls, which is shown separately as training resubstitution. All stages retain all Hough candidates, including Unknown identities; detection remains 86 TP, 2 FP and 4 FN. Identity uniqueness is not physical circle deduplication. No candidate is removed just for overlapping another. False detections can still steal identities, and global assignment can turn duplicates into different incorrect numbers.

Type evidence uses Gaussian distance to the original K2 decision intervals, width .10; numbered scores multiply type and histogram support. Cue uses its light fraction as colour evidence. B chooses the best supported identity above numerical zero. C/D consider every supported identity, with Unknown=0. Hungarian assignment maximises total score with one column per identity and separate Unknown columns. These are fixed heuristic scores, not calibrated probabilities. Normalised relative supports in evidence.csv are diagnostic only.

Size comes from the inverse homography Jacobian in source pixels/mm and the known 43mm diameter. The major projected semiaxis approximates sphere radius; the plane ellipse is also shown in grey. A single median AllBalls detected/expected radius ratio, 0.972030, scales the prediction. No evaluation locations or radii fit this model. D multiplies each candidate's identity scores by .75+.25*exp(-.5*(log(radius/expected)/log(1.5))^2), leaving the policy's Unknown score unchanged. Invalid geometry is neutral. The planar model is approximate because balls extend above the cloth and camera intrinsics are unknown.

This policy is an evaluation-informed follow-up after inspecting the conservative results. Colour/type/geometry scoring weights stay fixed; no additional adjustments were made during this run. The six images have been inspected in earlier development, so this is not pristine unseen-data validation. Choosing a stage after inspecting its scores is method selection, not a new independent test. The main pipeline is unchanged.

Outputs: configuration.txt; frozen_models.yml; training_geometry.csv; geometry_calibration.txt; metrics.csv; predictions.csv; evidence.csv (all 16 scores); type_scores.csv; geometry.csv; duplicate_claims.csv; changes.csv; confusion.csv; final_results/<stage>/<image>/; per-image geometry, candidate comparisons with score bars, and whole-table comparisons.
