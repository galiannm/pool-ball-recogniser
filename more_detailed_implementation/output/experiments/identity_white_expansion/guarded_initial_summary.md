# D: selective expansion of white detections

A/B/C retain best-effort K2+HSV inference. D replaces size weighting with colour growth around originally White candidates. Frozen colour models and all detection geometry are unchanged.

| Stage | Correct types | Correct numbers | Unknown | Extra duplicate claims |
|---|---:|---:|---:|---:|
| A: K2 + HSV | 76/86 | 72/86 | 4 | 10 |
| B: independent scores | 74/86 | 73/86 | 0 | 13 |
| C: unique identities | 76/86 | 75/86 | 1 | 0 |
| D: white-crop expansion | 76/86 | 75/86 | 1 | 0 |

Across Table1–6, 9 original White candidates were probed; 3 passed the expansion rule. Relative to C, D corrected 0 numbers and lost 0.

Probe +1..4 pixels. Admit coloured connected extensions reaching the original boundary, excluding cloth-hue pixels and other detected disks. Preserve the entire original disk. A component needs >=3 boundary pixels and >=max(12,2% original area) new pixels. Use the first padding confirmed by the next consecutive padding with the same valid non-cue type and body colour (choose the confirming padding). Otherwise keep original evidence. D recomputes K2 type/histogram evidence only for accepted expansions and reruns global assignment; non-expanded balls can change indirectly through uniqueness.

There is no size weighting, per-image exception or ground-truth padding selection. Cloth hue is estimated independently per image, without annotations. Green-ball pixels can resemble cloth; neighbour exclusions can hide real body colour and undetected neighbours can leak into growth. The few-pixel search may not reach the actual coloured body. These are limits of this specific rule.

This is an evaluation-informed follow-up to observed cue/stripe errors. Parameters were fixed before this run and not adjusted from its results. AllBalls is reported separately as training resubstitution. Detection counts, radii and coordinates stay unchanged; a larger recognition mask does not claim improved localisation. The main application and previous experiment outputs are unchanged.

Outputs: white_expansion.csv (all triggered paddings), applied_expansions.csv, cloth_models.csv, d_evidence.csv, predictions.csv, changes.csv, metrics.csv, frozen_models.yml, configuration.txt; white_expansion/<image>/candidate_N.png (five padding crops/masks); <image>/candidate_N.png (A–D comparison); final_results/<stage>/<image>/ (standard four PNGs and balls.csv).
