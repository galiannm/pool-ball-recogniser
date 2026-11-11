# Illumination experiments around the final pipeline

`illuminationNormalization.cpp` includes the current parent `myApplication.cpp`
without its standalone entry point. It does not use the historical modular
recognition implementation or run the parent application's menu/experiments.

Build and run from the assignment root:

```sh
cmake -S more_detailed_implementation -B more_detailed_implementation/build \
  -DOpenCV_DIR=/Users/manon/opencv-4.12.0-install/lib/cmake/opencv4
cmake --build more_detailed_implementation/build --target IlluminationExperiments -j 4
./more_detailed_implementation/build/IlluminationExperiments
```

The executable sets its working directory to the parent project before reading
`Media/` and `data/`. Its optional argument overrides the output directory. Default:
`more_detailed_implementation/output/experiments/illumination_normalization/`.
Use an appropriate OpenCV/toolchain configuration on other machines.

## Fixed alternatives

- **Baseline:** no image correction.
- **White balance:** use table pixels whose maximum BGR channel is 80-250 and
  whose `(max-min)/max` is at most 0.25. At least 100 pixels are required; otherwise
  retain the uncorrected image and record that fallback. Channel gains equal the
  average channel mean divided by each channel mean, clamped to 0.5-2.0. Saturate
  corrected values to the 8-bit range. This is a neutral-pixel grey-world estimate,
  not calibration from a known white target or automatic cue identification.
- **CLAHE:** convert the complete raw image to Lab, apply CLAHE with clip limit
  2.0 and an 8x8 tile grid to L only, then convert to BGR.

Settings were fixed before the run; no parameter search selected a winning variant.
There is no combined white-balance/CLAHE variant.

## Controlled comparison

Detect circles and estimate K=2 type evidence on raw images once. Use those same
circles, masks, original type evidence, original-radius cue rule, and Hungarian
assignment settings for all methods. Only histogram input pixels change, including
the HSV-based body-pixel selection. Apply each method consistently to AllBalls and
queries; rebuild its eight reference histograms from the same CSV-labelled raw
reference circles. AllBalls is reported separately as reference resubstitution.

For originally white candidates, reanalyse K=2 on the raw +1-pixel crop and score
colour on the corresponding corrected crop. The small identity coordinator in
this experiment preserves that split; calling the parent's `rawWhitePadding()` on
a corrected image would also change type evidence and confound the comparison.

The baseline model is checked for exact equality with the production model. Baseline
identities are also compared exactly against `runIdentityRecognition()` on every
image. Pooled controls must reproduce 86 TP, 2 FP, 4 FN, 73 correct colours and
80 correct identities; a mismatch fails the run instead of silently changing controls.
Ground truth labels are used for AllBalls reference construction and for evaluation
and figure captions. They do not select neutral pixels, corrections, or query identities.

## Outputs

- `metrics.csv`: per-image rows and pooled Table1-6 counts, matched and end-to-end
  accuracy, unknowns, numbered false detections, corrections and regressions.
- `predictions.csv`: all detected candidates and their evaluation outcomes. Missed
  annotations are represented in the metric counts, not as prediction rows.
- `changes.csv`: changed matched predictions relative to the raw control.
- `corrections.csv`: estimated gains, sample counts and fallback status.
- `*_model.yml`: each method's rebuilt reference histograms.
- `<method>/<image>/`: the same four final images and prediction CSV as production.
- `comparisons/<image>/`: labelled table comparisons and enlarged changed crops,
  including blue stripe 10 when present, whether or not its prediction changes.

The changed-crop grid shows corrected colours for inspection; the production output
views intentionally draw all predictions on the original photograph. A black-ball
or cue appearance change is not sufficient evidence of better recognition.

These are development-set experiments on the supplied camera and lighting. A
negative result does not establish that calibrated white balance, other CLAHE
parameters, or other illumination methods would fail in new conditions.
