# Analysis of saved results

Reviewed 3 November 2026; combined Hough + watershed experiment updated 4 November 2026. The original analysis reads saved outputs. The combined Hough + watershed section was updated after porting and rerunning that experiment; selected application outputs were unchanged. Paths below are relative to the repository root.

## Which results are comparable?

There are three result sets:

1. `output/final-results/`: the saved selected application and the newly run reconstructed historical comparisons.
2. `output/experiments/`: the original, more extensive experiment implementations and their saved comparisons.
3. `output/myApplication/` and `output/validation/`: older application/validation outputs.

The current saved selected run has 80 correct identities; the older `output/myApplication/metrics.csv` has 76 despite using the same method label. A label alone does not identify an identical implementation or run. Use the current result files together, rather than combining metrics or images from different directories. No cause for that historical discrepancy is established here.

The source comments explicitly describe several historical comparisons as reconstructions rather than exact reproductions. The former additive guided-Hough reconstruction has now been replaced by the original cluster-local Hough + watershed algorithm in the single-file application. Fresh before/after-watershed metrics reproduce the original pooled counts. Older ExperimentGuidedHoughResults images remain historical artifacts and are not the current experiment.

Recognition scores below exclude AllBalls and normally use the 86 spatially matched Table1–6 detections. Type accuracy means White/Solid/Striped correctness; identity accuracy means the exact numbered ball or cue. These are different tasks. AllBalls is used for training and its 16/16 result is training resubstitution, not independent validation.

## Selected application

Source: `output/final-results/metrics.csv` and `evaluation_matches.csv`.

| Image | Ground truth | TP | FP | FN | Correct identities / TP | Mean localisation error |
|---|---:|---:|---:|---:|---:|---:|
| Table1 | 16 | 14 | 1 | 2 | 11/14 | 7.68 |
| Table2 | 16 | 16 | 1 | 0 | 16/16 | 3.58 |
| Table3 | 16 | 15 | 0 | 1 | 15/15 | 3.12 |
| Table4 | 15 | 14 | 0 | 1 | 14/14 | 3.89 |
| Table5 | 14 | 14 | 0 | 0 | 13/14 | 5.83 |
| Table6 | 13 | 13 | 0 | 0 | 11/13 | 5.76 |
| Total | 90 | 86 | 2 | 4 | 80/86 | 4.91 |

Pooled detection precision is 86/88 = **97.73%**, recall is 86/90 = **95.56%**, and F1 is 172/178 = **96.63%**. These are computed from total counts, not by averaging image percentages.

Conditional identity accuracy is 80/86 = **93.02%**. End-to-end correct-identity recall is 80/90 = **88.89%**. Correct identities divided by all detections is 80/88 = **90.91%**. The latter two measures include different penalties: missed balls and false detections respectively. All six cue balls are correctly identified. There are two Unknown predictions: one genuine ball and one false detection.

The main weaknesses are concentrated: Table1 contributes two of four misses and three of six recognition errors. Tables2–4 identify every matched ball correctly, although Table2 contains a false detection and Tables3–4 each miss a ball. Tables5–6 show that recognition errors remain even when detection counts are perfect.

### Every remaining error

- Table1: balls6 and7 are missed; one unmatched circle is assigned6. Ball14 is Unknown. Balls9 and5 exchange identities.
- Table2: one extra circle is Unknown. All16 genuine balls are found and correctly identified.
- Table3: ball10 is missed.
- Table4: ball10 is missed.
- Table5: ball10 is predicted4.
- Table6: ball3 is predicted11, and ball10 is predicted4.

Ball10's repeated failures are useful report examples: it is missed twice and assigned4 twice. Predictions10→4 combine blue/purple colour confusion with striped/solid confusion; they are not merely a wrong number within the correct type. Prediction3→11 preserves red but confuses solid with striped. The Table1 9/5 exchange changes both colour and type. These describe the observed errors; establishing their pixel-level cause requires inspecting the corresponding masks and crops.

The Table1 false detection receiving6 illustrates a limitation of uniqueness: a candidate counted as a false detection can still consume a valid identity. Uniqueness constrains labels, not whether circles are real.

Mean localisation error is 4.91 table units, approximately millimetres at the chosen image scale. Median error is 3.75 and maximum is14.81. Seven matches exceed10 and three exceed14; the matching tolerance is15. In particular, Table1's swapped5/9 predictions have errors14.63 and14.81. These are weakly localised crops as well as recognition errors. Their proximity to the threshold means annotation uncertainty and matching tolerance deserve discussion. This does not justify changing the threshold to improve the score.

## New reconstructed comparisons

Sources: `historical_detection_experiments.csv`, `historical_recognition_experiments.csv`, and the corresponding functions in `myApplication.cpp`.

### Detection

| Method | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Selected Hough pipeline | 86 | 2 | 4 | 97.73% | 95.56% | 96.63% |
| Reconstructed watershed | 86 | 2 | 4 | 97.73% | 95.56% | 96.63% |
| Cluster-local Hough, before watershed | 86 | 16 | 4 | 84.31% | 95.56% | 89.58% |
| Cluster-local Hough + watershed | 83 | 11 | 7 | 88.30% | 92.22% | 90.22% |

Watershed produces no accepted detection changes. Visually separating a foreground region is not equivalent to accepting a new ball centre.

The current combined experiment estimates local cloth colour, selects foreground clusters, retains baseline circles outside routed cluster interiors and searches inside them with permissive Hough settings. Edge/arc/foreground checks precede marker-controlled watershed. The watershed stage rejects proposals whose supported basin area is inconsistent with their circle area. It does not consult annotations.

Before watershed the run gives 86TP/16FP/4FN. After watershed it gives83TP/11FP/7FN: five false detections are removed, but three genuine matches are lost. F1 improves from89.58% to90.22%, still below96.63% baseline. Table3 improves from15TP/0FP/1FN at baseline to16TP/4FP/0FN before watershed and16TP/3FP/0FN after watershed. This local success does not produce a pooled recall gain.

Current rows are named `GuidedHoughPreWatershed` and `GuidedHoughWatershed` in `historical_detection_experiments.csv`. Corresponding final-results directories contain circles; the combined directory also exports foreground, routing masks, cluster windows, basin diagnostics and proposal-level acceptance reasons. Current annotation configuration is `BallLocationsNew.csv`, verified byte-identical to `BallLocations.csv` for this update. The selected pipeline and its39 final PNG/CSV/model/metric artifacts were unchanged.

### Recognition

| Method | Correct types /86 | Correct identities /86 |
|---|---:|---:|
| Selected complete pipeline | 80 | 80 |
| K=2 only | 76 | Not evaluated |
| K=3 only | 73 | Not evaluated |
| K=4 only | 73 | Not evaluated |
| K=5 only | 73 | Not evaluated |
| K=6 only | 74 | Not evaluated |
| K=2 plus reconstructed CCA | 73 | Not evaluated |
| Manual HLS masks | 43 | Not evaluated |
| Independent K-means hue classifier | 75 | 68 |
| Expected-radius cue alternative | 76 | 75 |

Increasing K alone is not an improvement: the highest type count in this comparison is the K=2 baseline. More clusters can separate highlights and shading as well as useful regions, and the unchanged type thresholds need not suit the resulting partitions.

CCA loses three correct types relative to K=2. Spatial component rules can be affected by disconnected highlights, visible stripe orientation, occlusion and thresholded shadows. The result supports rejecting this particular CCA addition, not claiming that CCA is universally ineffective.

Manual HLS achieves43/86 =50% type accuracy, well below K=2's88.37%. These fixed thresholds provide a weak baseline on these images; this is not evidence against all uses of HLS.

The independent hue classifier achieves68/86 =79.07% identity accuracy, compared with93.02% for the complete selected system. That comparison changes histogram scoring and global heuristics together, so the12-ball gain cannot all be credited to histograms. The original histogram experiment below isolates that contribution more fairly.

The expected-radius alternative gives75/86 identities, five fewer than the selected raw-radius system. It chooses the white candidate closest to an approximate geometrically expected radius, not the same rule as the older row-adjusted radius-ratio experiment. Do not present their results as contradictory reproductions of one method.

`ExperimentHistogramBackprojectionResults` contains diagnostic images, not a separate row of classification metrics. The classifier's colour score is histogram similarity (1 minus Bhattacharyya distance); backprojection maps are not themselves the final recognition score.

## Original experiment programme

The following figures come from the original `output/experiments/` CSVs and summaries. They explain the development decisions but must retain their own controls and provenance.

### Detection alternatives

- **Hough search:** `output/grid_search.csv` contains480 configurations evaluated on AllBalls. A top configuration matches the selected dp1.5/minDist18/edge75/vote35/radius15–30 settings. Raw Hough gives16TP and1FP there; pocket filtering removes that extra training-image detection. A perfect post-filter AllBalls score does not establish generalisation to crowded tables.
- **Cloth backprojection:**36 settings preserve AllBalls recall. The frozen chosen filter rejects no Table1–6 candidates, retaining86TP/2FP/4FN. It adds computation without measured benefit. An inner white patch is not cloth-like, so a cloth rejection rule is poorly matched to that false-detection mechanism.
- **Circular edge support:** measurement only. False candidates score33.3% and45.8%; the weakest genuine ball scores41.7%. A support-only threshold sufficient to remove both false candidates would also remove at least one genuine detection. Removing just the weaker false candidate is separable in these examples, but selecting such a threshold from the same images would require new validation.
- **Original watershed:** Hough-only, Hough-seeded and hybrid all give86TP/2FP/4FN. Hough-seeded equality is expected because its circles are intentionally unchanged. Hybrid proposals are rejected as duplicates, crop-edge peaks, or implausible basins.
- **Original guided detector, initial version:**85TP/34FP/5FN, F1=81.34%; adding its watershed check gives82TP/24FP/8FN, F1=83.67%.
- **Original guided detector, revised version:**86TP/16FP/4FN, F1=89.58%; watershed checking gives83TP/11FP/7FN, F1=90.22%. The revised guided detector recovers three previously missed matches but loses three others. Recovering Table4's10 is a real local success, not a net recall gain. The watershed check removes some false candidates but also genuine balls. Both variants remain worse than96.63% baseline F1.

### Colour and type evidence

- **Padding every ball:** the initial radius-padding experiment has67/86 correct identities at zero padding without cloth removal, decreasing to60,49,43,42 for +1,+2,+3,+4. Cloth removal improves these to66,64,55,50, but none beats its own zero-padding cloth baseline69. Expanding every crop changes proportions and admits surrounding pixels. This is different from later selective padding of originally White candidates.
- **Recognition threshold search:** baseline67/86 rises to74/86 with configuration2153, using K=6 and tuned thresholds. It was explicitly ranked on Table1–6. It is a development result, not independent test accuracy, and is not equivalent to simply changing K in the new reconstruction.
- **CCA search:** on that tuned K=6 base, unrestricted original CCA reduces74 to72. The searched solid-rule configuration reaches75. This is a one-ball gain after a large exploratory search, with a substantial selection-bias concern.
- **Histogram colour representations:** on fixed K=2 types, the source's own current control has68/86 correct identities. HLS gives71; HS and HSV give72. HSV fixes four identities without losses against that control. Supplying true types gives79/86 for HSV, indicating that type errors remain important. The68 control differs from older67-count baselines and must not silently replace them in an ablation chain.
- **K=2 plus original CCA plus HSV:**72→69 identities, zero fixes and three regressions.
- **Tuned K=6 plus HSV:**72→77; adding the selected solid CCA rule reaches78. These use settings informed by Table1–6; they are not independent validation of the tuned settings.
- **Direct HSV pixel proportions:** independent local classification gives74 versus K=2's72. However, after uniqueness, padding and cue reservation, HSV gives79 versus K=2's80. A locally better frontend does not necessarily produce a better complete pipeline because it also changes padding triggers and global assignment evidence.

### Uniqueness, padding and cue selection

| Original controlled comparison | Correct identities /86 | Interpretation |
|---|---:|---|
| K=2 + HSV, local baseline | 72 | Duplicate labels allowed |
| Independent soft scores | 73 | Small net gain |
| Best-effort unique assignment | 75 | Six fixes, three losses versus72 baseline |
| Add weak geometric size weight | 75 | No observed benefit |
| Add guarded white expansion | 75 | No observed benefit |
| Selective raw padding +1 | 79 | Four fixes, no losses versus75 control |
| Selective raw padding +2 | 77 | Smaller gain |
| Selective raw padding +3 | 79 | Same total as+1 |
| Selective raw padding +4 | 79 | Five fixes and one loss versus75 control |
| Padding + largest raw cue reservation | 80 | One further correction; all six cues correct |
| Padding + largest row-adjusted cue reservation | 80 | Same cue choices and final identities as raw |

Initially conservative uniqueness achieved only69/86 and produced eight Unknowns. Best-effort eligibility raised that to75/86 and reduced Unknowns to one, with six fixes and no losses relative to conservative assignment. This supports the less restrictive policy on these examples, while leaving scores as uncalibrated heuristics.

Guarded expansion changed three genuine cues' local types without improving final identities. Global assignment masked those local failures, illustrating why intermediate masks and decisions matter as well as the final total.

The useful padding result is specifically **+1 on originally White recognition crops**, with detection geometry fixed. It is not evidence that enlarging every detected ball improves localisation. Identity changes can propagate to unpadded candidates through the unique assignment.

The row-radius line was trained over rows143.25–197.25. Using it elsewhere extrapolates beyond a narrow training span. Its raw and adjusted hard cue selectors made the same choices on all three tables where they activated. Raw selection therefore provides the same observed gain with fewer assumptions; it can still fail if perspective makes a genuine rear cue smaller than a nearer white patch, or if another circle's radius is inflated.

### Score diagnostics and chamfer refinement

- Lowering minimum body pixels12→6 or scaling cue scores to0.75/0.50 leaves the padded control at79/86. Broadening type compatibility width0.10→0.15 reduces it to77. These trials do not support adding further weighting complexity.
- Chamfer minimum-cost refinement stays at75/86, equal to its unpadded unique control. The outer-supported variant reaches77; simple white padding reaches79 in the same experiment.
- Outer-supported chamfer fixes the Table1 cue/14 confusion locally, but its documented changes include one-pixel centre shifts, mask discretisation and crossing the12-body-pixel cutoff. It does not demonstrate consistent recovery of the full physical ball boundary.
- Detection counts and exported Hough positions remain unchanged in the chamfer recognition experiment. Separate refined-centre diagnostics show both local improvements and regressions; they should not be described as a proven localisation improvement.

## Conclusions and limits

The saved evidence supports retaining the selected Hough detector with K=2, AllBalls HSV histogram evidence, best-effort unique assignment, selective +1 White-crop padding and raw-radius ambiguous-cue reservation. It does not support adding the tested watershed, permissive guided search, original K=2 CCA, generic size weighting or chamfer rules to that pipeline.

This is a judgement about the tested configurations and complexity tradeoffs, not a claim that those methods could never work. A one-ball difference here is only1.16 percentage points of conditional identity accuracy. Six images from the same camera, with some repeated ball arrangements, provide limited evidence about robustness in new conditions.

AllBalls-only model fitting does not make Table1–6 pristine test data after repeated inspection and method selection. The recognition and CCA grid searches explicitly tune on them. Present the final numbers as performance on the supplied development/evaluation images and disclose the selection history.

Ground-truth centres, the15-unit matching tolerance, possible annotation revisions, and transforming an elevated sphere's image centre using a cloth-plane homography all affect localisation interpretation. Do not change annotations merely to improve a score; any verified corrections should be documented and consistently applied across methods.

The report should show the current final images and metrics together, state both93.02% conditional identity accuracy and88.89% end-to-end recall, illustrate Table1's weakly localised swap, the repeated ball10 failures, and the recovered-but-costly Table3 combined-experiment detection. Perfect AllBalls performance belongs in the training discussion, not as evidence of unseen-image accuracy.
