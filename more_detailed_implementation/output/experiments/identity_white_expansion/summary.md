# D: selective expansion of white detections

A/B/C retain best-effort K2+HSV inference. D replaces size weighting with colour growth around originally White candidates. Frozen colour models and all detection geometry are unchanged.

| Stage | Correct types | Correct numbers | Unknown | Extra duplicate claims |
|---|---:|---:|---:|---:|
| A: K2 + HSV | 76/86 | 72/86 | 4 | 10 |
| B: independent scores | 74/86 | 73/86 | 0 | 13 |
| C: unique identities | 76/86 | 75/86 | 1 | 0 |
| D: guarded white expansion | 76/86 | 75/86 | 1 | 0 |
| D: raw white padding +1 | 79/86 | 79/86 | 1 | 0 |
| D: raw white padding +2 | 77/86 | 77/86 | 1 | 0 |
| D: raw white padding +3 | 79/86 | 79/86 | 1 | 0 |
| D: raw white padding +4 | 79/86 | 79/86 | 1 | 0 |

Across Table1–6, 9 original White candidates were probed; 3 passed the guarded expansion rule. Relative to C, guarded D corrected 0 numbers and lost 0.

Of the accepted guarded changes, 3 were genuine cue balls whose local type was changed to non-cue. This is a guard failure even when global assignment hides the change.

Raw +1..4 variants use full circular padding on every originally White candidate, without added-pixel gates or stable-pair selection. All four are reported, not selected from their evaluation scores. These ablations were added after the initial guarded run; guarded_initial_summary.md preserves the earlier report when one existed in this output folder.

| D variant vs C | Corrected numbers | Previously correct numbers lost |
|---|---:|---:|
| D: guarded white expansion | 0 | 0 |
| D: raw white padding +1 | 4 | 0 |
| D: raw white padding +2 | 2 | 0 |
| D: raw white padding +3 | 4 | 0 |
| D: raw white padding +4 | 5 | 1 |

Guarded D: probe +1..4 pixels. Admit coloured connected extensions reaching the original boundary, excluding cloth-hue pixels and other detected disks. Preserve the entire original disk. A component needs >=3 boundary pixels and >=max(12,2% original area) new pixels. Use the first padding confirmed by the next consecutive padding with the same valid non-cue type and body colour (choose the confirming padding). Otherwise keep original evidence. D recomputes K2 type/histogram evidence only for accepted expansions and reruns global assignment; non-expanded balls can change indirectly through uniqueness.

There is no size weighting, per-image exception or ground-truth padding selection. Cloth hue is estimated independently per image, without annotations. Green-ball pixels can resemble cloth; neighbour exclusions can hide real body colour and undetected neighbours can leak into growth. The few-pixel search may not reach the actual coloured body. These are limits of this specific rule.

This is an evaluation-informed follow-up to observed cue/stripe errors. Parameters were fixed before this run and not adjusted from its results. AllBalls is reported separately as training resubstitution. Detection counts, radii and coordinates stay unchanged; a larger recognition mask does not claim improved localisation. The main application and previous experiment outputs are unchanged.

Outputs: white_expansion.csv (guarded paddings), raw_padding.csv (all raw paddings), applied_expansions.csv, cloth_models.csv, d_evidence.csv, predictions.csv, changes.csv, metrics.csv, frozen_models.yml, configuration.txt; white_expansion/<image>/candidate_N.png (guarded/raw padding crops and masks); <image>/candidate_N.png (A–D comparison); final_results/<stage>/<image>/ (standard four PNGs and balls.csv).
