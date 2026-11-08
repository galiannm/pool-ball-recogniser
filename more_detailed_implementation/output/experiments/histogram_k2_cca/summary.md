# Frozen HSV histograms with original K=2 CCA

All profiles use exactly the same saved AllBalls colour models, query masks, scores, Hough circles and coordinates. Only type classification changes. No histograms were retrained and no parameters were searched or selected in this run.

| Method | Correct types | Correct numbers | Number with true type supplied |
|---|---:|---:|---:|
| K=2 + HSV histogram | 76/86 | 72/86 | 79/86 |
| K=2 + original CCA + HSV histogram | 73/86 | 69/86 | 79/86 |

| Comparison (Table1–6) | Numbers corrected | Previously correct numbers lost |
|---|---:|---:|
| K=2 + HSV histogram -> K=2 + original CCA + HSV histogram | 0 | 3 |

Counts pool spatially matched Table1–6 detections only. AllBalls is reported separately as training resubstitution. Detection totals are 86 TP, 2 FP, 4 FN and do not change. Unknown number predictions remain errors on matched balls. Number recall includes missed annotations; number precision includes false detections in its denominator. The true-type oracle is a diagnostic colour ceiling, not a deployed prediction.

This comparison calls the existing production CCA functions with K=2 directly, enabling the original cue, stripe and solid rules. No K=6 settings or grid-search CCA winner are applied. Colour histograms were trained on AllBalls only. Existing CCA thresholds are held fixed, not retuned from this run. This does not independently establish the provenance of every historical baseline threshold, and Table1–6 have been inspected in earlier development. The main application is unchanged.

Outputs: metrics.csv, predictions.csv, changes.csv (pairwise comparisons), colour_scores.csv (one shared set), cca_features.csv, confusion.csv, frozen_models.yml, configuration.txt, final_results/<method>/<image>/ and per-image candidate/table comparisons.
