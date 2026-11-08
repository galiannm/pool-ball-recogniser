# Frozen HSV histograms with K=6 and CCA

All profiles use exactly the same saved AllBalls colour models, query masks, scores, Hough circles and coordinates. Only type classification changes. No histograms were retrained and no parameters were searched or selected in this run.

| Method | Correct types | Correct numbers | Number with true type supplied |
|---|---:|---:|---:|
| K=2 + HSV histogram | 76/86 | 72/86 | 79/86 |
| K=6 + HSV histogram | 81/86 | 77/86 | 79/86 |
| K=6 + solid CCA + HSV histogram | 82/86 | 78/86 | 79/86 |

| Comparison (Table1–6) | Numbers corrected | Previously correct numbers lost |
|---|---:|---:|
| K=2 + HSV histogram -> K=6 + HSV histogram | 5 | 0 |
| K=6 + HSV histogram -> K=6 + solid CCA + HSV histogram | 1 | 0 |
| K=2 + HSV histogram -> K=6 + solid CCA + HSV histogram | 6 | 0 |

Counts pool spatially matched Table1–6 detections only. AllBalls is reported separately as training resubstitution. Detection totals are 86 TP, 2 FP, 4 FN and do not change. Unknown number predictions remain errors on matched balls. Number recall includes missed annotations; number precision includes false detections in its denominator. The true-type oracle is a diagnostic colour ceiling, not a deployed prediction.

The trained colour histograms use AllBalls only. The K=6 and solid-only CCA settings came from earlier searches using Table1–6; this combined evaluation is therefore exploratory, not independent test accuracy. Histogram light-mask thresholds stay V>=170,S<=165 for all methods; K=6 type/CCA uses its own V>=170,S<=125 cluster definition. Keeping these masks distinct preserves the frozen colour experiment and isolates the type-classifier contribution. The main application is unchanged.

Outputs: metrics.csv, predictions.csv, changes.csv (three pairwise comparisons), colour_scores.csv (one shared set), cca_features.csv, confusion.csv, frozen_models.yml, configuration.txt, final_results/<method>/<image>/ and per-image candidate/table comparisons.
