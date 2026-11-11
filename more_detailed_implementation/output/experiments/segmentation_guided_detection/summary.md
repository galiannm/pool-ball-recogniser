# Segmentation-guided Hough experiment

Foreground and cluster windows are constructed before Hough. Ground truth is used only after detection. This is one fixed configuration, not a grid search.

| Method | Split | TP | FP | FN | Precision | Recall | F1 | Mean error | Recovered/lost | Correct type/number |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|hough|evaluation|86|2|4|0.977273|0.955556|0.966292|4.914978|0/0|82/75|
|hough|reference|16|0|0|1.000000|1.000000|1.000000|2.762781|0/0|16/16|
|segmentation_guided|evaluation|86|16|4|0.843137|0.955556|0.895833|5.292716|3/3|78/71|
|segmentation_guided|reference|16|4|0|0.800000|1.000000|0.888889|3.650814|0/0|16/15|
|segmentation_guided_ws|evaluation|83|11|7|0.882979|0.922222|0.902174|5.239971|3/6|75/69|
|segmentation_guided_ws|reference|14|3|2|0.823529|0.875000|0.848485|3.323080|0/2|14/14|

Changed annotation matches:

- segmentation_guided_ws / AllBalls.jpeg / ball 2: lost
- segmentation_guided_ws / AllBalls.jpeg / ball 12: lost
- segmentation_guided / Table1.jpeg / ball 7: recovered
- segmentation_guided / Table1.jpeg / ball 9: lost
- segmentation_guided_ws / Table1.jpeg / ball 7: recovered
- segmentation_guided_ws / Table1.jpeg / ball 9: lost
- segmentation_guided_ws / Table2.jpeg / ball 8: lost
- segmentation_guided_ws / Table2.jpeg / ball 10: lost
- segmentation_guided / Table3.jpeg / ball 10: recovered
- segmentation_guided_ws / Table3.jpeg / ball 10: recovered
- segmentation_guided / Table4.jpeg / ball 7: lost
- segmentation_guided / Table4.jpeg / ball 10: recovered
- segmentation_guided_ws / Table4.jpeg / ball 7: lost
- segmentation_guided_ws / Table4.jpeg / ball 10: recovered
- segmentation_guided_ws / Table5.jpeg / ball 2: lost
- segmentation_guided / Table6.jpeg / ball 10: lost
- segmentation_guided_ws / Table6.jpeg / ball 10: lost

Standard Hough is evaluated on the unmodified table-masked image, then its cluster-interior centres are routed to local permissive searches. Boundary overlap is retained to avoid routing gaps. Local candidates require radial Canny-edge support and a visible circular arc. Duplicate suppression uses 0.8 times the smaller radius, allowing overlapping balls with distinct centres. If a local region accepts no candidate, its available standard detections are retained as an explicit fallback.

The optional watershed pass uses accepted circles as markers and rejects local candidates whose basin foreground area falls outside [0.15,2.0] expected disk areas. It does not invent centres or alter radii. Standard/fallback detections remain untouched. Comparing guided and guided_ws measures this check's contribution.

Recognition stays fixed to configuration 2153 plus selected solid-only CCA, with circular extraction masks. Recognition scores on Table1-6 are exploratory because its thresholds were previously tuned there. No pixel-level segmentation accuracy is claimed. Main application files are unchanged. parameters.csv, regions.csv and proposals.csv record settings, routes, arc evidence and watershed checks.
