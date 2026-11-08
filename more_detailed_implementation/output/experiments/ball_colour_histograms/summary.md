# Ball-colour histogram experiment

Models are trained exclusively on AllBalls. Three fixed representations are evaluated; no configuration is selected or adjusted using Table1–6. AllBalls metrics are training resubstitution, not a held-out estimate. The main application is unchanged.

| Method | Correct types | Correct numbers | Number with true type supplied |
|---|---:|---:|---:|
| current_k2 | 76/86 | 68/86 | 71/86 |
| hls_histogram | 76/86 | 71/86 | 78/86 |
| hs_histogram | 76/86 | 72/86 | 78/86 |
| hsv_histogram | 76/86 | 72/86 | 79/86 |

| Histogram method | Corrected numbers versus current | Previously correct numbers lost |
|---|---:|---:|
| hs_histogram | 4 | 0 |
| hsv_histogram | 4 | 0 |
| hls_histogram | 4 | 1 |

Type is deliberately held at the current K=2 classifier for all methods, isolating the colour-number stage. A cue prediction maps to 0; solids use colours 1–8; stripes use colours 1–7 plus 8. A striped prediction whose best model is black returns Unknown. Oracle numbers supply the annotation type only for diagnosis.

Training uses the same eroded-circle, non-white pixel mask as inference. It does not use hand-painted masks, test identities or test-derived colour thresholds. Each solid/stripe example contributes equally to its shared colour model; black has one example. Low-chroma hue is canonicalised to zero. Smoothing prevents hard bin boundaries from being the only source of mismatch. There is no fitted confidence or margin rejection rule.

Scores are 1 minus Bhattacharyya distance between L1-normalised histograms. Backprojection looks up the trained histogram mass per pixel using float images. PNG maps divide by each model's own peak for visibility; their brightness cannot be compared as class probabilities.

Outputs: training/models.yml and labelled training masks; metrics.csv; predictions.csv; colour_scores.csv (all eight scores); changes.csv (fixes and regressions); confusion.csv; per-image candidate comparisons; final_results/<method>/<image>/ (the same PNGs and CSV as the application).

Detection counts and localisation are unchanged. Conditional number accuracy divides by matched detections; number recall divides by all annotations; number precision divides by all detections. Confusion matrices cover spatially matched detections only. Earlier project experiments inspected Table1–6, so they are not pristine unseen images, although none of their labels fit these models. A single training view per ball limits lighting and pose variation. Residual cloth, neighbouring balls, white-patch masking and the unchanged type classifier remain potential failure sources.
