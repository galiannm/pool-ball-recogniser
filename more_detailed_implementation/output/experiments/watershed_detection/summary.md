# Watershed detection experiment

Fixed prototype; no grid search. Ground truth is used only for evaluation and diagnostics, never masks or seeds.

| Method | Split | TP | FP | FN | Precision | Recall | F1 | Mean error | Recovered/lost | Correct type/number |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|hough|evaluation|86|2|4|0.977273|0.955556|0.966292|4.914978|0/0|82/75|
|hough|reference|16|0|0|1.000000|1.000000|1.000000|2.762781|0/0|16/16|
|hough_seeded|evaluation|86|2|4|0.977273|0.955556|0.966292|4.914978|0/0|82/75|
|hough_seeded|reference|16|0|0|1.000000|1.000000|1.000000|2.762781|0/0|16/16|
|hybrid|evaluation|86|2|4|0.977273|0.955556|0.966292|4.914978|0/0|82/75|
|hybrid|reference|16|0|0|1.000000|1.000000|1.000000|2.762781|0/0|16/16|

Hough-seeded watershed is a segmentation control: its circles and standard extraction masks are deliberately unchanged, so detection and recognition scores should equal Hough. Hybrid can append accepted peak-centred candidates but does not relocate or delete Hough circles. Recognition is frozen to configuration 2153 plus the selected solid-only CCA rule. Watershed labels are diagnostic and are not substituted for circular recognition masks.

Baseline misses and pocket-filter status are in ground_truth_audit.csv. Parameters are in parameters.md and watershedDetection.h. Basin rejection reasons are in proposals.csv. A local window requires at least two nearby filtered Hough detections; wholly missed or isolated balls outside those windows cannot be recovered. Green balls can still merge into the cloth mask. Watershed cannot reconstruct an invisible boundary or guarantee an unbiased centre for an occluded ball. No pixel-level annotations are available, so segmentation quality is inspected visually. Recognition parameters were previously tuned on Table1-6, so recognition scores are not independent test estimates.

Baseline misses:

- Table1.jpeg: ball 6 (inside a crowded ROI); already missed before pocket filtering
- Table1.jpeg: ball 7 (inside a crowded ROI); already missed before pocket filtering
- Table3.jpeg: ball 10 (inside a crowded ROI); already missed before pocket filtering
- Table4.jpeg: ball 10 (inside a crowded ROI); already missed before pocket filtering

Peak outcomes across all seven images:

- basin_area: 1
- basin_circularity: 3
- hough_duplicate: 40
- near_crop_edge: 8
