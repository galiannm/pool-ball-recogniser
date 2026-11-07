# Missing final-results image exports

All paths below are relative to `output/final-results/`. They are proposed destinations, not existing results.
The report never falls back to another output folder. Re-run `python3 "report v3/check_assets.py"` after exporting.

Do not overwrite a selected result with a control. Export a configuration identifier, matching metric row, and candidate indices for every comparison.

- `ExperimentCuePaddingResults/Table1/padded_numbered.png`
  Export selective +1 White-crop padding plus assignment, without largest-radius cue reservation.
- `ExperimentCuePaddingResults/Table1/unpadded_numbered.png`
  Export soft scores plus unique assignment, without selective padding or cue reservation.
- `ExperimentKMeansK2Results/AllBalls/predicted_types.png`
  Run the specified type-only method on AllBalls and export its labelled overlay. Training performance must stay separate from Tables1–6.
- `ExperimentManualHLSResults/AllBalls/predicted_types.png`
  Run the specified type-only method on AllBalls and export its labelled overlay. Training performance must stay separate from Tables1–6.
- `ExperimentSoftTypeResults/Table1/hard_types.png`
  Export the hard-type histogram identity control with detection fixed, before unique assignment.
- `ExperimentSoftTypeResults/Table1/soft_types.png`
  Export independent soft-score identities with the same detections, before unique assignment.
- `mainBallExtractionResults/AllBalls/ball_1_mask.png`
  Export the crop-aligned circular extraction mask before recognition padding.
- `mainBallExtractionResults/AllBalls/ball_1_original.png`
  Export the original-colour circular candidate crop before selective recognition padding, using the same candidate index as recognition_evidence.csv.
- `mainBallExtractionResults/Table1/ball_6_original.png`
  Export the original-colour circular candidate crop before selective recognition padding, using the same candidate index as recognition_evidence.csv.
- `mainBallExtractionResults/Table1/ball_7_original.png`
  Export the original-colour circular candidate crop before selective recognition padding, using the same candidate index as recognition_evidence.csv.
- `mainColourRecognitionResults/Table1/ball_9_histogram.png`
  Plot/export the measured HSV descriptor with labelled axes or projections; do not substitute a backprojection map.
- `mainDetectionResults/AllBalls/initial_manual_circles.png`
  Export the actual initial manual Hough configuration; record its parameters. Do not relabel the selected configuration.
