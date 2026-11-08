# Cloth histogram backprojection verification

Screened 36 configurations on AllBalls only. Recall-safe: 36/36. Frozen config: 9, enabled=1, H bins=18, S bins=16, similarity threshold=0.800000, maximum cloth fraction=0.900000.

Selection preserves every originally matched AllBalls candidate, then uses a predeclared preference: higher similarity threshold, higher allowed cloth fraction, fewer H bins, fewer S bins, earlier ID. If no valid recall-safe model exists, verification is disabled. AllBalls has no post-pocket false detections in this run, so this is recall screening, not learned precision optimisation. selection.csv is frozen before Table1-6 images/labels are evaluated; only that setting is evaluated there.

| Method | Split | TP | FP | FN | Precision | Recall | F1 | Mean error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|A_hough|evaluation|86|2|4|0.977273|0.955556|0.966292|4.914978|
|A_hough|screening|16|1|0|0.941176|1.000000|0.969697|2.762781|
|B_pocket|evaluation|86|2|4|0.977273|0.955556|0.966292|4.914978|
|B_pocket|screening|16|0|0|1.000000|1.000000|1.000000|2.762781|
|C_backprojection|evaluation|86|2|4|0.977273|0.955556|0.966292|4.914978|
|C_backprojection|screening|16|0|0|1.000000|1.000000|1.000000|2.762781|

Rejected candidates:

None.

Table1-6: 0 baseline matched candidates rejected, 0 baseline false detections rejected.

Cloth models are re-estimated per image. Sampling: original table mask eroded by an 11x11 ellipse, exclude all raw Hough circles with a four-pixel radius margin and known 25-table-pixel pocket regions. At least 1000 sampling pixels are required. No H/S eligibility threshold or GT exclusions are used.

H/S ranges: [0,180), [0,256). Histogram min/max normalised to [0,255]; 8-bit backprojection then converted to float by dividing by 255. No per-map renormalisation. Values are relative histogram similarities, not probabilities. Scoring uses the original circle intersected with the table/image, never the sampling mask. Less than 80% circle coverage causes conservative retention. High-similarity pixels use score >= threshold; rejection uses clothFraction > maximum. Invalid/insufficient models retain all candidates. No edge criterion was added in this first experiment.

The filter never moves, adds or changes a circle radius. Main localisation/recognition code is unchanged. Table1-6 were excluded from this filter's tuning but informed previous project experiments, so these results are a frozen-setting comparison, not pristine independent test evidence.
