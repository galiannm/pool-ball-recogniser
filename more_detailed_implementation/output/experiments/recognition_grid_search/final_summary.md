# Final-result recognition comparisons

All profiles use the same detections and rendering as the application. No search was rerun, and no experimental recognition rules were changed.

| Profile | Correct type (Table1–6) | Correct number (Table1–6) |
|---|---:|---:|
| Current application | 76/86 | 68/86 |
| K=6 tuned (2153) | 81/86 | 74/86 |
| K=6 + CCA (20242) | 82/86 | 75/86 |

There are 2 false detections and 4 missed annotations. These are excluded from conditional recognition accuracy, not silently corrected.

- `final_results/<image>/`: tuned K=6 winner 2153, without CCA.
- `final_results_cca/<image>/`: K=6 plus solid-only CCA winner 20242.
- `final_results_current/<image>/`: snapshot of the current application.
- `final_comparisons/<image>/`: whole-table comparison and one labelled crop per changed candidate. Green crop text means correct number, red incorrect, grey unmatched.
- `final_changes.csv`: all candidates, including corrections, regressions and unchanged predictions.
- `final_counts.csv`: per-image counts; AllBalls is listed separately.
- `final_configuration.txt`: frozen settings and the black-threshold distinction.

Each final-result folder contains the same four PNGs and balls.csv as the application. They show predictions only; annotation labels are confined to the comparison outputs.

The current application includes S <= 60 for black. The historical K=6 experiment uses S < 60. Consequently Table2 candidate 16 (black 8) is correct in the current application but remains yellow 1 in these K=6 snapshots. This is shown as a regression.

The winners were selected using Table1–6 during earlier experiments. These remain exploratory results, not independent test estimates. The main application settings are unchanged.
