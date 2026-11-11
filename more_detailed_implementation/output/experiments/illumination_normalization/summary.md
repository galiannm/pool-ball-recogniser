# Lighting normalisation: current final pipeline

Fixed colour-branch experiments; detections and K=2 appearance evidence are unchanged.
AllBalls supplies reference models and is excluded from the pooled Table1-6 totals.

| Method | Correct colours | Colour accuracy | Correct identities | Matched identity accuracy | End-to-end | Numbered false detections |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 73/80 | 91.25% | 80/90 | 93.02% | 88.89% | 1 |
| white_balance | 74/80 | 92.50% | 79/90 | 91.86% | 87.78% | 2 |
| clahe_lab_l | 69/80 | 86.25% | 75/90 | 87.21% | 83.33% | 2 |

Detection stays at 86 TP, 2 FP and 4 FN for all methods. Initial types stay at 76/86; this is a controlled colour-only change, not a full-image preprocessing comparison. All three methods retain 15/15 reference colours and 16/16 reference identities on AllBalls.

| Image | Baseline IDs | White balance IDs | CLAHE IDs |
|---|---:|---:|---:|
| Table1 | 11 | 11 | 7 |
| Table2 | 16 | 15 | 15 |
| Table3 | 15 | 15 | 15 |
| Table4 | 14 | 14 | 14 |
| Table5 | 13 | 13 | 13 |
| Table6 | 11 | 11 | 11 |

## Interpretation

- White balance fixes one raw colour prediction (Table5, true purple stripe 12: black to purple) without losing any previously correct colour predictions. Its final identity was already correct, so the colour correction adds no final correct identity.
- White balance loses one final identity on Table2: green stripe 14 becomes Unknown, while an existing false detection changes from Unknown to 14. The best-colour label remains green in both candidates; changed score strengths affect the global assignment.
- CLAHE loses four correct standalone colour predictions and five correct identities, with no corrections of either metric. Four identity regressions occur on Table1 and one on Table2. The Table2 false-detection competition also occurs with CLAHE.
- Neither method repairs the blue stripe 10 errors on Tables5 or 6.
- Keep the uncorrected baseline for this dataset. These fixed experiments do not rule out calibrated white balance or differently designed lighting correction on new data.

## Reproducibility

Build/run instructions and exact parameters: [illuminationNormalization.md](../../../experiments/illuminationNormalization.md) relative to this output directory. Production model and per-image baseline identity parity checks passed. References were rebuilt for each method from the same raw AllBalls detections. White coverage, cue selection, and raw/padded geometry were fixed. No ground truth enters the image correction.

Source fingerprints are recorded in `source_hashes.json`. No production source, root build configuration, or report files were edited.
