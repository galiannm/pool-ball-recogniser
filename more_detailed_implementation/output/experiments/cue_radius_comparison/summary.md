# Does the row adjustment change cue selection?

All methods share K2, frozen AllBalls HSV histograms, +1 padding on original White candidates, and best-effort unique assignment. Only cue reservation differs. Models and geometry are frozen.

| Method | Correct identities | Accuracy | Correct cues | Unknown | Fixes / losses vs padding |
|---|---:|---:|---:|---:|---:|
| padding_only | 79/86 | 91.860465% | 5/6 | 1 | 0 / 0 |
| largest_raw_radius | 80/86 | 93.023256% | 6/6 | 2 | 1 / 0 |
| largest_adjusted_radius | 80/86 | 93.023256% | 6/6 | 2 | 1 / 0 |

Both selectors activate only with>=2 original White candidates (3 evaluation images). Raw selection maximises original radius; adjusted selection maximises radius/expected(row). Ties retain the first candidate. Each forces the selected candidate to cue and forbids other cue claims.

| Image | Raw selected candidate | Adjusted selected candidate | Comparison |
|---|---:|---:|---|
| Table1 | 6 | 6 | same |
| Table3 | 2 | 2 | same |
| Table6 | 1 | 1 | same |

Raw versus adjusted: 0 different cue choices and 0 different final identities across Table1–6. The radius line adds no observed prediction benefit here. This supports preferring the simpler raw-radius rule on these examples, but does not establish perspective-invariance on new scenes. The narrow AllBalls row range cannot strongly calibrate depth; the existing model is used unchanged. Neither size rule can guarantee that the full cue is larger than a nearer white patch or an inflated false circle.

AllBalls is reported separately as training resubstitution. Ground truth is used only after predictions, but these images informed earlier development; the comparison is exploratory, not independent validation. Outputs: metrics.csv, predictions.csv, changes.csv, cue_candidates.csv, configuration.txt, frozen models, synthetic_checks.txt, candidate_comparisons/<image>/candidate_N.png, final_results/<method>/<image>/. Main application and report unchanged.
