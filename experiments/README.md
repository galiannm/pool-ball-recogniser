# Radius padding experiment

Run these commands from the repository root:

```sh
cmake -S . -B build-local
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments
```

The default image is `Media/AllBalls.jpeg`. To process another image or the
configured image set:

```sh
./build-local/BallExperiments Media/Table1.jpeg
./build-local/BallExperiments --all
```

An optional second argument overrides the output directory. Detection uses the
existing table geometry and Hough parameters, so custom images must use the
same camera/table setup.

Results go to `output/experiments/radius_padding/<image-stem>/`:

- `ball_N_comparison.png` contains one candidate's ten final masked crops:
  padding 0–4 across columns, padding only on top and cloth filtering below.
  All variants use the same display scale. Images live directly in this directory,
  without individual ball folders.
- Each image names the expected type and number in its header. Each crop shows
  separate type and number predictions, coloured green (`OK`) or red (`WRONG`).
  Candidates without a ground-truth match show grey `N/A` indicators.
- `results.csv` records pixel counts, predicted type/number, cluster percentages,
  expected type/number and correctness flags. Unmatched correctness fields are blank.
  Predicted numbers of `-1` mean unknown.

Ground truth is loaded from `config::groundTruthPath`. Candidates are matched once
in transformed table coordinates using the existing one-to-one evaluation and
`config::matchTolerance`; this same match is used for all padding variants.

Each image is detected once. Padding enlarges both the crop and circular mask;
unfiltered padding 0 reproduces the existing extraction. Classification uses the
same K-means and type/number functions as `MyApplication`, with its fixed seed.
No GUI windows are opened.

`RadiusPaddingOptions` in `radiusPadding.h` contains the filtering settings.
The filter estimates green cloth colour in a six-pixel ring outside the padded
circle, then removes similar-colour components connected to the circular mask
boundary. It skips filtering when fewer than 20 green samples are available.
The HSV thresholds are experimental: green balls can lose real ball pixels,
and shadows or reflections can leave cloth behind. Inspect masks before drawing
conclusions. The CSV's kept pixel count is the denominator for cluster percentages.

Repeated runs overwrite results with the same image stem.
Use different output directories to preserve comparisons between filter settings.
Generated results are ignored by Git; this experiment does not change the main
application's behaviour.

# Recognition grid search

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --recognition-grid
```

An optional second argument changes the output directory. The search ranges,
default thresholds, shortlist size and pass count live in `recognitionSearchConfig.h`.
The command preserves the radius-padding experiment and the main application.

The search detects candidates once per image and caches K-means features for
K=2–6 with cloth filtering off/on, always using zero padding. Before searching,
parameterised default rules must agree with production classifications and
percentages on every cached segmentation. Type and number thresholds share
the same definition of a light cluster.

Stage one exhaustively compares 1,800 K/filter/type-threshold configurations.
The shortlist includes the top three distinct K/filter pairs under number-first
and type-first rankings, plus both original K=2 baselines, deduplicating seeds.
Stage two performs two passes of grouped Cartesian searches over warm-colour,
cool-colour and black/body-selection thresholds. Hue intervals must remain
ordered and the existing unknown-colour gaps remain positive. This is a staged
search, not a global exhaustive search over every parameter at once.

Ranking uses correct numbers, correct types, oracle numbers (true type supplied),
smaller departure from the original settings, then earlier configuration ID.
The best-type ranking swaps the first two criteria. Table1–6 supply tuning
scores. AllBalls is evaluated separately for finalists and never affects selection.
These are exploratory tuning scores, not independent test accuracy.

Outputs in `output/experiments/recognition_grid_search/`:

- `summary.md`: run counts, principal scores and interpretation limits.
- `type_leaderboard.csv` / `colour_leaderboard.csv`: all trials, parameters,
  ranking scores, stage and parent ID. Repeated configurations remain in the audit trail.
- `k_comparison.csv`: original thresholds and best stage-one thresholds per K/filter pair.
- `shortlist.csv` / `finalists.csv`: seed-to-final IDs and full finalist parameters.
- `metrics.csv`: pooled/per-image type, number and oracle accuracy plus end-to-end number recall.
- `predictions.csv`: finalist predictions for each candidate, with ground truth and correctness.
- `confusion.csv`: type, number and oracle-number matrices, including unknown predictions and zero cells.
- `detection_counts.csv` / `mask_counts.csv` / `cluster_features.csv`: matching counts,
  retained pixels and cluster features for reproducibility.
- `<image-stem>/ball_N_comparison.png`: one image per candidate showing the baseline
  and distinct finalists. Correctness is green/red, or grey for unmatched candidates.
  Candidate indices are not ground-truth ball numbers.

Failed/empty segmentations count as incorrect for matched balls. Unmatched
detections have blank correctness fields and are excluded from recognition
accuracy. End-to-end number recall includes missed ground-truth balls in its
denominator. Oracle scores use the true type, including the cue-ball shortcut,
and are a diagnostic rather than a deployable classifier.

To refresh the report's local result tables and example figure after running
both experiments:

```sh
python3 experiments/writeRecognitionReport.py
make -C report
```

The LaTeX discussion is in `report/sections/recognition_experiments.tex`. Its
interpretation and run-specific numerical statements need review after changes
to the search; the snapshot script refreshes tables and the figure only.

# CCA grid search

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --cca-grid
```

The optional second argument overrides `output/experiments/cca_grid_search/`.
Ranges and a frozen copy of recognition configuration 2153 are in
`ccaSearchConfig.h`. CCA runs on K=6 with that profile's light definition,
zero padding and no cloth filtering; the previous recognition output files
are not required. K=2/no-CCA and K=2/production-CCA are additional controls.
Experimental default features and hybrid types must exactly match production
on every original K=2 candidate before the search proceeds.

The search compares four component-area cutoffs, three erosion kernels and
all eight enabled-rule subsets. For each of the 96 seeds, two grouped passes
tune the enabled cue, stripe and solid rules. The default grid scores 33,124
trials including four controls; repeated parameter vectors are retained.
This is a staged search rather than a global exhaustive threshold search.

Ranking prioritises correct numbers, then correct types. Ties prefer fewer
number regressions, fewer type regressions, fewer type changes, fewer enabled
rules, smaller parameter departure and earlier ID. All regression counts
compare against the tuned K=6 base, including those shown for K=2 controls.
The best-type ranking swaps the first two criteria. Table1–6 supply tuning
scores; AllBalls is a reference check only.

Outputs include `summary.md`, `leaderboard.csv`, `ablation.csv`, `seeds.csv`,
`finalists.csv`, `metrics.csv`, `predictions.csv`, `confusion.csv`,
`detection_counts.csv`, `features.csv`, and `number_ceiling.csv`.
The latter supplies true types to the fixed K=6 number rules: because a
correct number implies a correct type under those rules, its score bounds
the number improvement possible from type-only correction on these candidates.

One `<image-stem>/ball_N_comparison.png` per candidate shows distinct
controls/finalists, final masked crops, light/non-light masks and their boundary
bands, feature values, the rule that fired and correctness indicators.
Green/red pixels in the feature masks mean light/non-light boundary pixels;
green/red prediction text means correct/incorrect. Unmatched correctness is N/A.

Small-component filtering changes component statistics only, not the raw
light percentages or boundary coverage. Cue, stripe and solid overrides are
tested in that order; otherwise the colour-only type is retained. A cue rule
that fails does not veto the baseline's white prediction.

To refresh the detailed CCA report snapshots and PDF:

```sh
python3 experiments/writeCCAReport.py
make -C report
```

The discussion is in `report/sections/cca_experiment.tex`. Its run-specific
prose must be reviewed if the grid or results change. The main application's
recognition settings and CCA functions are unchanged by this experiment.

# Watershed detection comparison

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --watershed
```

An optional output-directory argument overrides
`output/experiments/watershed_detection/`. This is a fixed prototype, not a grid
search. Options are in `watershedDetection.h`; values are recorded in `parameters.csv`
and the procedure in `parameters.md`. It processes the configured image set
with the existing camera/table geometry.

The three variants are Hough alone, Hough-seeded watershed, and hybrid watershed
with additional distance-transform markers. Nearby Hough groups define local
windows. A local Lab cloth estimate supplies a provisional foreground, keeping
known Hough cores, closing small gaps and filling holes. The hybrid considers
distance maxima that are sufficiently independent of existing markers, then
checks their watershed basins for area, shape, foreground support and pockets.
Existing Hough circles are never moved or removed. Proposed centres use peaks,
not basin centroids; proposed radii use the local median Hough radius.

Hough-seeded watershed is a segmentation control, so its detection scores are
unchanged by design. All recognition uses standard circular masks and the
fixed K=6/configuration-2153 classifier plus the selected solid-only CCA rule.
Watershed masks are diagnostic only. Ground-truth annotations enter evaluation
and overlays, never segmentation, markers or candidate acceptance. The public
segmentation function accepts no ground-truth arguments.

Outputs:

- `summary.md` and `metrics.csv`: pooled and per-image detection/recognition scores.
- `ground_truth_audit.csv`: all annotations, raw/filtered matching, nearest baseline
  error and whether the point falls inside a processed window.
- `regions.csv`: crop coordinates, Hough seeds, expected size, cloth statistics and status.
- `proposals.csv`: peak positions, basin checks and explicit rejection reasons.
- `predictions.csv`: final centres, matched labels and frozen recognition predictions.
- `changes.csv`: recovered/lost annotation identities for each method.
- `<image-stem>/detection_comparison.png`: all three detection variants.
- `<image-stem>/region_N_diagnostics.png`: original region, foreground, distance
  transform, both marker maps, both watershed label maps, and peak outcomes.

The current prototype adds no detections on the real images. Do not interpret
the existence of watershed labels as successful missed-ball recovery. Its
distance peaks are unreliable in the merged silhouettes of the actual crowded
groups. No pixel-level annotations are available to quantify mask accuracy.

To refresh the report snapshots and PDF:

```sh
python3 experiments/writeWatershedReport.py
make -C report
```

The detailed discussion is in `report/sections/watershed_experiment.tex`.
The snapshot script updates tables and figures, not the run-specific prose.

# Segmentation-first Hough routing

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --segmentation-hough
```

An optional output directory overrides `output/experiments/segmentation_guided_detection/`.
Options are in `segmentationGuidedDetection.h`. Foreground and cluster windows are
created **before the first Hough call**, without Hough seed disks or annotations.
Known table geometry and the existing radius limits provide a size guide.
The main application remains unchanged.

The comparison includes:

1. Original standard Hough.
2. Standard Hough routed outside component-based cluster interiors, with permissive
   local Hough and radial arc validation inside. A boundary-overlap margin avoids gaps.
3. The same guided detections with an additional watershed basin-area check.

Standard Hough is calculated on the unmodified table-masked image and routed by
centre, avoiding artificial image edges at cluster boundaries. Local Hough uses
accumulator threshold 22, minimum centre distance 12, and modestly widened size
bounds. It requires radial Canny/Sobel support over at least 30% of the circumference
and a 75-degree supported arc. Duplicate rejection uses 0.8 times the smaller
radius, permitting genuinely overlapping neighbours. If a local region accepts
nothing, available standard circles in that region are retained as an explicit fallback.

Watershed uses the accepted circles as markers and tests local candidates' basin
foreground area. It does not invent centres or adjust radii; retained standard and
fallback circles are not rejected by it. Recognition stays fixed to configuration
2153 plus the selected solid-only CCA rule, using standard circular masks.

Outputs include `summary.md`, `metrics.csv`, `parameters.csv`, `regions.csv`,
`proposals.csv`, `predictions.csv`, `changes.csv`, and `coverage.csv`. Each image
has a foreground/routing sheet, a full three-method detection comparison and one
diagnostic sheet per cluster window. Recovered annotations also have evaluation-only
close-ups such as `Table4/recovered_ball_10_comparison.png`. The annotation centres
select these display crops only after all detections finish.

The current guided prototype recovers ball 10 in Table3 and Table4, but its extra
false detections and lost baseline matches make its overall performance worse.
Read the aggregate scores as well as the recovered-ball pictures. The initial
11-pixel boundary-inset engineering run was preserved under
`output/experiments/segmentation_guided_initial_boundary/`; current defaults use
a 41-pixel inset to reduce rail/shadow connections before component labelling.
This was not an exhaustive parameter search or held-out generalisation test.

Refresh the report's result tables and Table4 figure with:

```sh
python3 experiments/writeSegmentationGuidedReport.py
make -C report
```

The detailed discussion is in `report/sections/segmentation_guided_experiment.tex`.
The snapshot script does not regenerate its run-specific interpretation.

# Cloth histogram backprojection verification

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --backprojection
```

An optional second argument overrides `output/experiments/cloth_backprojection/`.
Reusable experiment-only functions are in `clothBackprojection.h/.cpp`; they
accept images/masks/candidates and never load annotations. The runner is
`backprojectionExperiment.cpp`. The main application and recognition remain unchanged.

The runner screens the 36 H/S-bin and threshold configurations on AllBalls,
requiring every originally matched candidate to remain. Selection uses a fixed
conservative order: highest similarity threshold, highest allowed cloth fraction,
fewest H bins, fewest S bins, earlier ID. If no valid recall-safe configuration
exists, filtering is disabled. `selection.csv` is closed before Table1–6 images
or labels are evaluated. Only that frozen setting is evaluated on Table1–6.
AllBalls has no post-pocket false detections, so this is recall screening rather
than a learned optimum for precision. Previously inspected project images are
not claimed to be pristine independent test data.

Sampling excludes all raw Hough candidates (four-pixel radius margin), the known
pocket neighbourhoods and an 11x11 inset of the table boundary. At least 1000
samples are needed. H/S ranges are [0,180), [0,256). The histogram is normalised
to [0,255], backprojected with 8-bit HSV, then divided by 255 into a float score.
There is no subsequent map renormalisation. Scores are similarities, not probabilities.

Scoring uses the original circle intersected with the table/image—not the
sampling mask. At least 80% circle coverage is required for rejection. Pixels
with score >= the similarity threshold are cloth-like; a candidate is rejected
only when their proportion is strictly greater than the maximum cloth fraction.
Insufficient/flat models and inadequate circle coverage retain candidates.
Mean score is recorded but is not an extra rejection rule. There is no edge
criterion in this first experiment. Retained circles keep their order, coordinates
and radii exactly.

Outputs:

- `grid_search.csv` / `screening_scores.csv`: 36 AllBalls configurations and all
  candidate-level screening scores. IDs identify this experiment only.
- `selection.csv`: frozen configuration, fixed mask settings and selection policy.
- `evaluation.csv`: A (table mask + Hough), B (+ pockets), C (+ verification).
- `candidate_scores.csv`: original B candidate index, raw candidate index,
  ground-truth audit label, coverage, mean score, cloth fraction and retention reason.
- `models.csv`, `rejections.csv`, `predictions.csv`, `changes.csv`: model status,
  removed matched/unmatched candidates and final annotation matches.
- `screening/`: similarity and histogram images for the four AllBalls H/S models.
- `<image-stem>/`: sampling mask, saved histogram YAML, histogram heatmap
  (H rows/S columns), similarity maps, accepted/rejected overlay and comparison sheets.

Ground-truth labels appear only in screening/evaluation files and overlays.
In `verification.png`, green means retained and red rejected, independently of
ground-truth correctness. `detection_comparison.png` separately uses green for
matched detections, red for unmatched detections and blue for missed annotations.

The current conservative setting preserves all true detections but also both
remaining false detections. No evaluation threshold was changed to force a gain.
Refresh the report snapshots with:

```sh
python3 experiments/writeBackprojectionReport.py
make -C report
```

The detailed discussion is in `report/sections/backprojection_experiment.tex`.

# Circular edge support diagnostic

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --edge-support
```

An optional second argument overrides `output/experiments/circular_edge_support/`.
This measures every post-pocket Hough candidate across AllBalls and Table1–6.
It does not reject, move or resize circles, or change the application/report.
`circularEdgeSupport.h/.cpp` provides annotation-free measurement functions;
`circularEdgeExperiment.cpp` handles evaluation and visualisation.

Fixed settings: 72 boundary directions, 2.5-pixel search tolerance, Canny
37.5/75 after grayscale Gaussian smoothing (5x5, sigma 1.5), and minimum
absolute radial gradient alignment 0.6. Edge maps use the original image to
avoid artificial table-mask boundary edges; supporting pixels and sampled
boundary positions must be inside the table. Opposite gradient polarities
both count. The search tests proximity and gradient direction, so an edge
merely near the circumference does not automatically support it.

The score is supported directions / valid directions; coverage records valid
directions / 72. Arc lengths include wrap-around at zero degrees. The raw arc
requires uninterrupted support; the bridged arc fills isolated one-sample gaps
and leaves the support fraction unchanged. These are geometric measurements,
not probabilities or a trained rejection rule. Settings are not tuned on labels.

Outputs:

- `summary.md`: false detections, overlap with genuine-detection scores, and
  the weakest genuine detections across Table1–6 (excluding AllBalls).
- `candidate_scores.csv` and `scores_ranked.csv`: candidate ID, annotation
  match, original circle, valid coverage, nearby-edge fraction, aligned edge
  fraction, raw longest arc and bridged longest arc.
- `angular_samples.csv`: every sampled boundary position, validity, nearby
  edge flag, aligned support flag, selected edge pixel and gradient alignment.
- `settings.csv` and `baseline_metrics.csv`: reproducible settings and the
  unchanged detector's per-image counts and localisation metrics.
- `Table1_false_candidate_15.png`, `Table2_false_candidate_17.png`: convenient
  close-ups of both current false detections.
- `<image-stem>/candidate_N.png`: one comparison per candidate with original
  crop, coloured circumference and Canny evidence. Green means aligned support,
  orange a nearby edge without sufficient alignment, red no nearby edge, grey
  an invalid sample; cyan dots identify the selected supporting edge pixels.
- `<image-stem>/overview.png`, `canny.png`: overview labels contain candidate
  IDs and support percentages. Overview green/red mean matched/unmatched
  against annotations, independently of the boundary-support colours.

Current results: Table1's false candidate scores 33.3%, Table2's false candidate
(the offset circle around ball 10) scores 45.8%. The weakest genuine detection,
Table1's crowded orange ball 5, scores 41.7%. A support-only cutoff could remove
the first false detection without losing an existing true detection on these
images, but rejecting the second would also reject at least that genuine ball.
This is exploratory score inspection on already seen images, not independent
validation of a chosen threshold. No rejection threshold is applied.

# Final images for the tuned recognition configurations

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --recognition-final
```

The optional output argument defaults to
`output/experiments/recognition_grid_search/`. This exports the already selected
recognition winner 2153 and CCA winner 20242 without rerunning either search.
It does not modify the main classifier or its settings. All seven images use
the current detector, with exactly the same candidate order and coordinates
for every recognition profile.

- `final_results/<image-stem>/`: tuned K=6, no CCA.
- `final_results_cca/<image-stem>/`: tuned K=6 plus the selected solid-only CCA.
- `final_results_current/<image-stem>/`: current production classifier snapshot.
- `final_comparisons/<image-stem>/table_comparison.png`: three overhead views.
- `final_comparisons/<image-stem>/candidate_N.png`: original crop with the
  three predictions and the annotation identity, saved for changed candidates.
  Green text means correct number, red incorrect, grey unmatched detection.
- `final_changes.csv`: every candidate's current, K=6 and K=6+CCA predictions,
  with number corrections, regressions and other changes explicitly identified.
- `final_counts.csv`, `final_summary.md`, `final_configuration.txt`: per-image
  counts, pooled comparison and settings/provenance.

Each final-results directory has the same four PNGs and `balls.csv` schema as
the application. Application and experiment use `io/finalResults.h/.cpp` for
identical rendering. Ground truth is loaded after predictions are exported
and is used solely for the comparison outputs and counts.

The historical experiment uses strict `S < 60` for black; the application now
uses `S <= 60`. The exporter preserves both rules rather than retroactively
changing the experiment. In particular, Table2 candidate 16 remains yellow 1
in the K=6 exports but is black 8 in the current application, so it appears as
a regression. The original winner was selected on Table1–6; its scores remain
exploratory rather than independent test accuracy.

# AllBalls recognition slider viewer

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --recognition-tuner
```

Run from the repository root in a desktop terminal. The optional output argument
defaults to `output/experiments/recognition_tuner/`. This opens one window with
the sliders, a prediction overlay cropped to the balls, and masks for the selected
candidate. The compact shared window keeps sliders visible on macOS instead of
letting larger preview windows cover a separate controls window. Saved previews
retain the full original-image overlay and full-resolution selected-ball grid.
AllBalls is the only image that can be loaded in this viewer. Detection and
extraction stay fixed while recognition changes; K-means is cached separately
for each visited K, so changing thresholds does not rerun clustering.

Controls:

| Slider | Meaning |
| --- | --- |
| K | Number of BGR clusters, 2–8 |
| Light V min | Minimum HSV brightness for a light cluster |
| Light S max | Maximum HSV saturation for a light cluster |
| Cue light % | Minimum light coverage for cue classification |
| Solid nonlight % | Minimum non-light coverage for solid classification |
| Black S max | Saturation cutoff for the selected body cluster |
| Black V max | Maximum brightness for black classification |
| Black S inclusive | 0 means S < cutoff; 1 means S <= cutoff |
| Blue-purple H | Hue boundary between blue and purple, 101–164 |
| Candidate | One-based detection index for the enlarged crop and masks |

These are HSV controls, not HLS: V is brightness, not HLS lightness. Masks show
whole K-means clusters selected by their representative HSV, not independent
pixel-level thresholding. The body mask identifies the largest eligible non-light
cluster. Black-like masks show all clusters meeting S/V cutoffs; the number rule
still uses only the selected body cluster and requires a solid type for black.
The reporting-only `darkMaxV` is fixed and not exposed as a misleading number
classification slider. CCA, padding and cloth filtering are disabled.

- **T**: tuned K=6 preset (configuration 2153, original strict black cutoff).
- **B**: current K=2 baseline preset, including the inclusive black cutoff.
- **S**: save a new `snapshot_0001/`, `snapshot_0002/`, etc.; existing snapshots
  are preserved. Each contains `settings.yml` with every recognition parameter,
  `overview.png`, `selected_ball.png`, `predictions.csv`, `clusters.csv`, and the
  usual `final_results/AllBalls/` images and `balls.csv`.
- **Q** or **Esc**: quit. Closing the viewer window also exits.

The viewer starts with the tuned K=6 preset. Cyan highlights the selected
candidate; green/red overlays indicate correct/incorrect number predictions
against AllBalls annotations; unmatched detections are grey. The header shows
conditional type/number accuracy and matched detections versus annotation count.
Annotations do not influence predictions. No settings are written to the main
application and no snapshot is saved until S is pressed.

Use the masks and cluster CSV to inspect why a setting changes a prediction.
Perfect AllBalls accuracy alone cannot distinguish several configurations or
establish generalisation. The T preset comes from the earlier exploratory search
on Table1–6; starting from it does not make those earlier results independent.
Future manual adjustments should use AllBalls only and be frozen before any
further evaluation of the six project images.

# Learned ball-colour histograms and backprojection

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --colour-histograms
```

The optional output argument defaults to
`output/experiments/ball_colour_histograms/`. Reusable training/inference code
is in `ballColourHistograms.h/.cpp`; the runner is `ballHistogramExperiment.cpp`.
The main application and existing experiments are unchanged.

The runner matches AllBalls detections to its labelled annotations and uses
balls 1–15 to train eight body-colour models. Solid and stripe examples share
a model: 1/9 yellow, 2/10 blue, 3/11 red, 4/12 purple, 5/13 orange, 6/14 green,
7/15 maroon; 8 supplies the black model. Cue is handled by the existing type
classifier and does not train a body-colour histogram. Missing training identities
or insufficient body pixels abort training rather than using evaluation images.

Three fixed representations are compared, without a parameter search or automatic
winner selection: HS (18 hue × 8 saturation bins), HSV (18×8×8), and HLS (18×8×8,
stored internally in H,S,L order). Hue spans [0,180); saturation and tone span
[0,256). Histograms use individual pixels, rather than K-means average colours.
HLS is included as a genuine lightness/saturation representation in this experiment.

All spaces use the same body mask: erode the circular crop with a 3×3 ellipse,
then remove pixels satisfying HSV V>=170 and S<=165. At least 12 body pixels
are required. For BGR chroma (max channel minus min channel) <15, hue is set to
zero because near-neutral hue is unstable. There is no cloth filter, padding,
hand-painted training mask or class-specific threshold. Shadows and some adjacent
balls can therefore remain in the training/query pixels.

Each histogram is smoothed with [1,2,1]/4 along its axes, wrapping hue and
replicating saturation/tone endpoints, then L1-normalised. Shared colour models
average individually normalised examples, giving each ball equal weight despite
different crop sizes. The models are written and closed in `training/models.yml`
before Table1–6 images or their labels are used.

Inference compares the query histogram against all eight frozen models using
1 minus Bhattacharyya distance. The highest score selects body colour; there is
no fitted confidence/margin rejection threshold. The current K=2 type classifier
is held fixed for every method to isolate number recognition. White maps to cue,
solid maps to colours 1–8, stripe maps to colours 1–7 plus 8; stripe+black returns
Unknown. The baseline includes the application's inclusive black cutoff.

Backprojection uses float feature images so sub-unit histogram masses are not
rounded away. Its maps locate colours consistent with each learned histogram.
The displayed winning-model map is scaled by that model histogram's peak, purely
for visibility; map brightness is not the classification score or a calibrated
class probability. Rankings come from whole-body histogram comparison.

Outputs:

- `configuration.txt`: fixed settings and scoring protocol.
- `training/models.yml`: all trained histograms and example counts.
- `training/examples.csv`, `training/ball_N.png`: training identities, source
  circles, body-pixel counts, original crops and masks (cue marked unused).
- `training/*_histograms.png`: model visualisations, with H rows, S columns and
  tone slices arranged left-to-right; each colour is peak-scaled for display.
- `metrics.csv`: per-image and pooled evaluation metrics, with AllBalls explicitly
  labelled training resubstitution and excluded from pooled Table1–6 results.
- `predictions.csv`, `colour_scores.csv`, `changes.csv`, `confusion.csv`: candidate
  predictions, all eight model scores, fixes/regressions and number confusion.
- `<image-stem>/candidate_N.png`: original crop, body mask and three winning-model
  backprojection maps, with predicted identities and a separate annotation label.
- `final_results/<method>/<image-stem>/`: the same original, rectified, numbered
  and diagram PNGs plus `balls.csv` as the application, using predictions only.
- `summary.md`: method comparison, denominators and limitations.

Current Table1–6 number scores: baseline 68/86; HS 72/86; HSV 72/86; HLS 71/86.
All methods retain the same 76/86 correct types and detection counts (86 TP,
2 FP, 4 FN). AllBalls scores are 16/16, measured on the training crops themselves.
Oracle-number scores supply the true type solely to diagnose remaining colour
errors; they are not deployed predictions. The experiment does not combine the
learned colour models with tuned K=6/CCA type decisions.

No evaluation labels train these histograms or select thresholds. The six images
have nevertheless been inspected in earlier project work, so they should not be
described as pristine unseen data. Scores from these three predefined methods
must not be used to retrospectively claim a newly selected independent test result.

# Frozen HSV colour models with K=6 and solid-only CCA

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --histogram-cca
```

The optional output argument defaults to `output/experiments/histogram_cca/`.
Run `--colour-histograms` first if its model file does not exist. This experiment
loads the exact saved HSV model from
`output/experiments/ball_colour_histograms/training/models.yml`; it does not
retrain, rerun a search, or change the main application. It validates the eight
colour models, dimensions, example counts and normalisation, then snapshots
the source file as `frozen_models.yml` before evaluation.

The three fixed profiles are:

| Profile | Type classifier | Number classifier |
| --- | --- | --- |
| `k2_hsv` | Current K=2 rules | Frozen AllBalls HSV histogram |
| `k6_hsv` | Tuned K=6 type rules from 2153 | Same histogram |
| `k6_cca_hsv` | Same K=6 rules plus solid-only CCA 20242 | Same histogram |

All profiles share identical detections, crops, histogram query masks, colour
scores and transformed centres. Scores are computed only once per candidate.
The histogram body mask retains its original pixel-level V>=170,S<=165 light
exclusion and 3×3 erosion, while K=6 type/CCA uses the separate cluster-level
V>=170,S<=125 light definition. These deliberately remain distinct so the colour
models and query extraction match the preceding histogram experiment.

K=6 uses cue light>=85%, solid non-light>=80%. CCA uses eight-connectivity,
a 2% ball-area component cutoff and a 5×5 ellipse boundary band. Only the solid
override is enabled: non-light>=70%, light-boundary coverage<=20%, exactly one
eligible non-light component; otherwise retain the K=6 base type. Cue and stripe
overrides are disabled. Histogram numbering uses no hand-written hue boundary
or black saturation/brightness cutoff.

Outputs:

- `summary.md`, `metrics.csv`: per-image and pooled counts/accuracies, oracle
  number scores, detection counts and number recall/precision. AllBalls is labelled
  training resubstitution and excluded from pooled Table1–6 results.
- `predictions.csv`: all three profile predictions, shared colour and score,
  margin, body-pixel count and annotation correctness.
- `changes.csv`: every candidate under K2→K6, K6→CCA and K2→CCA comparisons,
  separately identifying type and number corrections/regressions.
- `colour_scores.csv`: one shared set of all eight colour scores per candidate.
- `cca_features.csv`: K=6/CCA types, decision reason, light/non-light proportions,
  connected-component measurements and light-boundary coverage.
- `confusion.csv`: type and number matrices for spatially matched detections.
- `final_results/<profile>/<image-stem>/`: the usual four PNGs and `balls.csv`.
- `<image-stem>/candidate_N.png`: three prediction crops plus K=6 light,
  non-light and boundary masks. Green/red text indicates correct/incorrect number;
  grey means unmatched detection. The annotation label is separate from inference.
- `<image-stem>/table_comparison.png`: the three overhead prediction views.
- `configuration.txt`, `frozen_models.yml`: settings and original model snapshot.

Current Table1–6 results:

| Profile | Correct types | Correct numbers |
| --- | ---: | ---: |
| K=2 + HSV histogram | 76/86 | 72/86 |
| K=6 + HSV histogram | 81/86 | 77/86 |
| K=6 + solid-only CCA + HSV histogram | 82/86 | 78/86 |

K2→K6 corrects five numbers without regressions. CCA corrects one further number:
Table1 candidate 10, true orange ball 5, changes from striped 13 to solid 5.
All three obtain 16/16 on AllBalls. The shared true-type number diagnostic is
79/86, so type-only changes can recover at most one further number at these fixed
colour predictions. Detection remains 86 TP, 2 FP and 4 FN; combined number recall
over all 90 annotations is 78/90. The earlier tuned rule-based K=6+CCA classifier
scored 75/86, but is not another profile in this isolated comparison.

Colour models were trained only on AllBalls. K=6/CCA settings were selected using
Table1–6 in earlier experiments, so these combined results remain exploratory.
They are not independent test accuracy, and no settings were adjusted in this run.

# K=2 histograms with the original CCA rules

```sh
./build-local/BallExperiments --histogram-k2-cca
```

The optional output argument defaults to `output/experiments/histogram_k2_cca/`.
This is a separate two-profile comparison: `k2_hsv` versus
`k2_original_cca_hsv`. It reuses the frozen AllBalls HSV models and the shared
histogram/CCA exporter, but calls the original production
`analyseConnectedComponents()` and `classifyBallTypeWithCCA()` directly with K=2.
The K=6 settings and tuned CCA winner are not applied.

Original CCA enables all cue, stripe and solid overrides, retaining their
existing thresholds and fallback order. Light is V>=170,S<=165; components use
eight-connectivity and a 2% area cutoff; the boundary uses a 5×5 ellipse.
Cue requires light>=85%, largest light>=80%, light boundary>=75% and no eligible
non-light component. Stripe requires largest light>=20%, light boundary>=20%
and non-light>=10%. Solid requires non-light>=70%, light boundary<=12% and one
eligible non-light component. No thresholds are adjusted from this evaluation.

The output schema matches `--histogram-cca`, with two profiles, one pairwise
comparison, and `k2Type` instead of `k6Type` in `cca_features.csv`. Final results,
per-ball mask/prediction comparisons, frozen model snapshot, metrics, scores,
confusion matrices and corrections/regressions are all saved.

On Table1–6, original CCA reduces correct types from 76/86 to 73/86 and correct
numbers from 72/86 to 69/86. It corrects no numbers and loses three previously
correct ones. AllBalls remains 16/16 for both profiles. Detection and histogram
scores are identical. This comparison therefore supports retaining K=2+HSV
histograms without the original CCA overrides. The main application is unchanged.
These are fixed existing rules, not a newly tuned CCA configuration; that does
not independently certify the provenance of every historical baseline threshold.

# A/B/C/D identity and geometric-size heuristics

```sh
cmake --build build-local --target BallExperiments -j 4
./build-local/BallExperiments --identity-heuristics
```

The optional output argument defaults to `output/experiments/identity_heuristics/`.
All four related variants are implemented together in `identityHeuristics.cpp`;
the header only exposes the runner. It loads the existing AllBalls HSV histogram
file (run `--colour-histograms` first if absent), snapshots it, and uses K=2.
No K=6 settings or tuned CCA parameters are applied. The main pipeline is unchanged.

| Stage | Inference |
| --- | --- |
| A | Existing hard K=2 type + winning frozen HSV body-colour histogram |
| B | Independent best identity from soft type/colour compatibility scores |
| C | Same scores, with unique identity assignment and Unknown options |
| D | Same assignment, with a weak geometric size multiplier |

The original K=2 light-fraction intervals motivate the type scores: cue [.85,1],
solid [0,.30], stripe [.30,.85]. Compatibility is exp(-.5*(distance/.10)^2),
where distance is zero inside the interval. Numbered identity scores multiply
type compatibility by the corresponding colour-histogram similarity. Cue uses
light fraction times cue compatibility. These scores are not calibrated
probabilities. `relativeSupport` divides by the sum over the 16 identities only
for inspection; it is not used by the assignment.

B takes the best score if strictly above the predeclared Unknown score .35.
C/D use a rectangular Hungarian assignment with columns for identities 0–15 and
one additional Unknown column per candidate. Any number of identities can be
absent, including cue. Multiple candidates may be Unknown. Only alternatives
within an absolute .15 of that candidate's local best, and above .35, are allowed.
Unknown has a 1e-9 tie preference. Overlapping circles are not removed: uniqueness
of identities is distinct from physical circle deduplication. No identity score
uses evaluation annotations.

Geometry uses the inverse homography Jacobian, expressed in source pixels per
physical table millimetre. A 43mm circle projects locally to an ellipse; its major
semiaxis approximates a sphere's source-image radius. The minor axis/angle remain
available for diagnostics. A single median scale correction is estimated from
AllBalls Hough radii, without annotations or evaluation-image radius fitting.
The endpoint convention of the existing table transform is preserved.

D uses the multiplier .75+.25*exp(-.5*(log(detected/expected)/log(1.5))^2).
This can reduce identity scores by at most 25%; it does not modify coordinates,
radii or the Unknown score. Invalid geometry has neutral multiplier 1. The model
is approximate because balls extend above the cloth plane and camera intrinsics
are unknown. The initial multiplier range over these images is about .950–1.000;
both true and false circles can have plausible sizes. There is no hard size filter.

Outputs:

- `final_results/<stage>/<image-stem>/`: usual four PNGs and `balls.csv` for A–D.
- `<image-stem>/candidate_N.png`: four predictions, expected/detected geometry,
  body mask and all 16 identity-score bars. B/C/D markers identify selected scores.
  Green/red text means correct/wrong number; grey means unmatched detection.
- `<image-stem>/table_comparison.png`: four overhead prediction images.
- `<image-stem>/geometry.png`: detected orange circles, expected cyan circles,
  projected plane ellipses in grey and candidate IDs.
- `evidence.csv`: all identity scores, colour/type factors, normalised relative
  support, shortlist eligibility and D's weighted scores.
- `type_scores.csv`, `geometry.csv`: measured light fractions, type scores,
  geometric radii/angles, size ratios, compatibility and weighting factors.
- `predictions.csv`, `metrics.csv`, `changes.csv`, `confusion.csv`: all predictions,
  per-image/pooled metrics, incremental and baseline comparisons, and matched-number
  confusion matrices. Unknowns remain errors on matched annotations.
- `duplicate_claims.csv`: repeated identities in each image/stage; counts in
  `metrics.csv` are extra claims beyond one, not numbers of physical duplicate balls.
- `configuration.txt`, `frozen_models.yml`, `training_geometry.csv`,
  `geometry_calibration.txt`, `summary.md`: reproducibility and result summary.

Initial Table1–6 results (parameters were fixed before this run):

| Stage | Correct types | Correct numbers | Extra duplicate claims |
| --- | ---: | ---: | ---: |
| A | 76/86 | 72/86 | 10 |
| B | 73/86 | 72/86 | 13 |
| C | 72/86 | 69/86 | 0 |
| D | 72/86 | 69/86 | 0 |

All stages obtain 16/16 on AllBalls. Across the six images, A→B fixes one number
and loses one. B→C fixes three and loses six; A→C fixes two and loses five.
D changes no assignment relative to C. For example, Table6's real cue has local
cue score .947, while a light striped 12 has cue score .979; uniqueness awards
cue to the wrong candidate and leaves the real cue Unknown. This illustrates why
a correct uniqueness constraint cannot repair unreliable evidence by itself.

These particular heuristics do not improve the baseline. All stages keep every
Hough detection (86 TP, 2 FP, 4 FN); an Unknown identity is not a removed detection.
No weights were adjusted after examining the results. The six images have been
inspected in earlier development; this is not pristine unseen-data validation,
and selecting a method after seeing these scores would itself be method selection.

## Best-effort identity assignment

```sh
./build-local/BallExperiments --identity-best-effort
```

This uses the same `identityHeuristics.cpp` implementation with a separate policy
and output directory, `output/experiments/identity_best_effort/` (optional output
argument supported). The original conservative results are preserved.

Only abstention/shortlisting changes:

- A remains exactly the K=2+HSV baseline.
- B returns the strongest supported identity without requiring score >.35.
- C/D allow every identity with nonzero support; the .15 alternative-score
  shortlist is removed. Unknown score drops from .35 to zero. A 1e-9 numerical
  tie preference keeps genuinely unsupported choices Unknown.
- Candidates must still have usable body-colour evidence or a baseline White
  classification; lack of image evidence is not replaced by a fabricated colour.
- Uniqueness is retained. More than 16 candidates, or too few supported identity
  columns, can still require Unknown. Missing identities remain permitted.

The histogram models, K=2 segmentation, type-compatibility width and geometric
calibration/weighting do not change. Conservative controls are recomputed from
the same cached evidence for direct comparison; this is not a new parameter search.

`confidence.csv` flags selected scores at/below the old .35 threshold or farther
than .15 from the local best (D uses its size-weighted score for the threshold).
Those labels remain predictions, with `?` on candidate sheets and amber rings in
`final_results/<stage>/<image>/confidence_overlay.png`. Flags describe old-rule
violations, not calibrated probabilities. All stages also retain the standard
four PNGs and `balls.csv`; no flags or settings are applied to the main application.
`policy_comparison.csv` records conservative/current identities and each correction
or regression. Existing `changes.csv` still compares stages within this new run.

Current results across Table1–6:

| Stage | Conservative correct numbers | Best-effort correct numbers | Conservative Unknown | Best-effort Unknown |
| --- | ---: | ---: | ---: | ---: |
| A | 72/86 | 72/86 | 4 | 4 |
| B | 72/86 | 73/86 | 1 | 0 |
| C | 69/86 | 75/86 | 8 | 1 |
| D | 69/86 | 75/86 | 8 | 1 |

Relative to the conservative policy, C/D correct six numbers and lose none.
Relative to the plain A baseline, they correct six and lose three (net +3).
The sole remaining C/D Unknown is Table2 candidate 17, an unmatched false
detection; all matched balls receive a guess. Seven C/D predictions carry
low-support flags. D still changes no assignments relative to C. These gains
are observed on previously inspected evaluation images; they do not establish
independent generalisation or calibrated confidence. The policy change follows
inspection of the conservative results and is therefore evaluation-informed.
No further score weights
were adjusted from these results.

## White-crop expansion instead of size weighting

```sh
./build-local/BallExperiments --identity-white-expansion
```

This keeps A/B/C from the best-effort run and replaces D's radius weighting
with recognition-crop expansion, still in `identityHeuristics.cpp`. The default
output is `output/experiments/identity_white_expansion/` (optional override).
Only candidates classified **White by the original K=2 classifier** are expanded.
The target is a Hough circle following a striped ball's white patch. The actual
stored Hough centre/radius, detection counts and transformed positions never change.
The frozen AllBalls HSV models and best-effort assignment settings also stay fixed.

Two forms are reported:

- `D_raw_pad1` through `D_raw_pad4`: expand each originally White crop and its full
  circular mask by exactly 1–4 source pixels, recompute K=2 type/query-colour scores,
  and rerun unique assignment. Non-white candidates keep their original evidence.
  All four fixed padding variants are exported; no per-image padding is selected.
- `D_white_expansion`: guarded growth admitting only coloured connected extensions,
  with cloth-hue and neighbouring-circle exclusions. It requires two consecutive
  paddings agreeing on a non-cue type/body colour and uses the first confirming
  padding; otherwise it keeps the original crop. Exact thresholds are recorded
  in `configuration.txt`. This guard was evaluated first; its thresholds were not
  retuned when it failed. The direct-padding ablations were then added to distinguish
  that guard from the underlying padding idea. An existing guarded report is kept
  as `guarded_initial_summary.md`.

Guarded sampling estimates cloth hue per image from the inset table excluding raw
Hough disks. Coloured extensions require S>=80,V>=40, exclusion of light clusters
and cloth-like hue, boundary contact and a minimum new area. This can remove dark
purple or genuine green-body evidence while still admitting cue-ball shadows.
Three genuine cues passed the guard and became locally striped/green; none of the
false-white examples passed. Their final global assignments stayed unchanged.

Results on Table1–6:

| Variant | Correct numbers | Corrections vs C | Regressions vs C |
| --- | ---: | ---: | ---: |
| C, no expansion | 75/86 | — | — |
| Guarded D | 75/86 | 0 | 0 |
| Raw +1 | 79/86 | 4 | 0 |
| Raw +2 | 77/86 | 2 | 0 |
| Raw +3 | 79/86 | 4 | 0 |
| Raw +4 | 79/86 | 5 | 1 |

Raw +1/+3 fix the false cue claims for striped 12 in Table3 and Table6, and restore
the two corresponding real cue identities. +4 additionally fixes Table1 striped
14 but causes another Table1 candidate (true striped 12) to receive cue. The raw
variants illustrate that extra colour evidence and global assignment interact;
local type need not become striped for the global identity to improve. Padding
also includes cloth and shadows, so performance is not monotonic in radius.
All variants retain 16/16 training-resubstitution numbers on AllBalls, 86 TP/2 FP/
4 FN on Table1–6, and one Unknown identity in the unique-assignment results.

Outputs include the usual `final_results/<stage>/<image>/`, `metrics.csv`,
`predictions.csv`, `changes.csv` (each D vs A/C), and `summary.md`. Additional files:

- `white_expansion.csv`: every guarded padding, support measurements and selection.
- `raw_padding.csv`: local measurements for all four raw paddings.
- `applied_expansions.csv`: accepted guarded updates, with evaluation labels added
  only after inference for diagnosis.
- `cloth_models.csv`: per-image hue/sample counts for the guarded method.
- `d_evidence.csv`: original and updated scores for all identities and D variants.
- `white_expansion/<image>/candidate_N.png`: guarded crops/masks and raw crops for
  every triggered candidate, in one sheet per candidate.
- `<image>/candidate_N.png`, `table_comparison.png`: all stage predictions together.

This is an evaluation-informed follow-up to observed errors, not independent test
validation. Raw variants were added after inspecting the guarded failure, without
choosing a winner from the four or changing their fixed padding values. The main
application and previous experiment outputs remain unchanged.

## Colour-score weighting diagnostics

```sh
./build-local/BallExperiments --colour-score-diagnostics
```

Outputs default to `output/experiments/colour_score_diagnostics/`. This compares
ten explicit diagnostic settings in `identityHeuristics.cpp`: unpadded, +1 and +4
controls; +1 with query minimum body pixels 12 -> 6; cue-score multipliers .75 and
.50; Gaussian type width .10 -> .15; and three combinations of those changes.
Only originally White candidates receive padding. The minimum query sample
requirement and score weights apply to all candidates. Frozen training histograms,
white thresholds and detection geometry stay fixed. These comparisons investigate
an observed evaluation error; they are parameter experiments, not independent
validation or settings selected on AllBalls.

The unpadded/+1/+4 controls exactly reproduce the earlier 75/79/79 correct
identities. Lowering the pixel minimum and reducing cue weight do not change any
identities relative to +1. Increasing type width reduces 79 to 77, including two
Table1 regressions. None resolves the Table1 cue/14 conflict.

The Table1 diagnostic exposes why. With +1, the real cue (#6) has cue support .863
but also false green-stripe support .363 from a cloth rim in its crop. Ball14 (#7)
has cue support .940 and only eight eligible body pixels. Allowing those eight
pixels admits green support .101, but the assignment still prefers false cue plus
false green (total1.303) over the correct pair (total.963). Uniformly reducing cue
weight does not remove the misleading green alternative. Larger padding further
contaminates the real cue crop with cloth, eventually allowing another striped
candidate to claim cue. These are heuristic support scores, not probabilities.

`evidence.csv` decomposes every identity score into colour and type terms.
`cue_counterfactuals.csv` forces each originally White candidate to claim cue in
turn, reruns assignment, and reports objective loss; this never changes inference.
`cue_swaps.csv` reports feasible two-candidate swap costs. `changes.csv` compares
every variant against both unpadded and +1 controls. Per-candidate diagnostic
sheets show crop, histogram body mask and top five identity scores. Standard
`final_results/<method>/<image>/` exports are provided for every setting.

## Chamfer boundary-refinement experiment

```sh
./build-local/BallExperiments --chamfer-refinement
```

The independent `chamferRefinement.cpp/.h` experiment defaults to
`output/experiments/chamfer_refinement/`. It uses the same K2/frozen-HSV/best-effort
unique recognizer as the colour diagnostics. Only original White candidates are
refined; original stored Hough geometry and detection counts remain fixed.

Candidate crop centres shift by at most two pixels per axis and radii grow by
0..8 pixels. A circle template has 72 samples; its score is the mean distance to
a nearby edge agreeing with the radial gradient direction, capped at three
pixels. Coverage, angular-sector support and neighbour-interior gates reject
poorly supported proposals. Exact settings are saved in `configuration.txt`.
One profile takes the minimum-cost supported refinement; another prefers the
largest supported radius within .15 pixels of the lowest mean cost. All settings
were declared before this run, with no per-image exceptions.

| Method | Correct identities | Corrections / regressions vs C |
| --- | ---: | ---: |
| Unrefined unique control | 75/86 | 0 / 0 |
| Fixed +1 control | 79/86 | 4 / 0 |
| Minimum-cost chamfer | 75/86 | 0 / 0 |
| Outermost supported chamfer | 77/86 | 2 / 0 |

The outer-supported variant fixes both Table1 identities: real cue (#6) and
striped14 (#7), while it does not reproduce the Table3/6 padding fixes. The real
cue crop moves down one pixel with unchanged radius; ball14 moves down one pixel
and grows one pixel. This is a small boundary/ROI correction, rather than proof
that a full outer ball boundary has been recovered. AllBalls retains16/16 across
the four profiles. Overall +1 still scores higher; the two methods fix different
examples and are not combined in this experiment.

Outputs include the standard `final_results`, `predictions.csv`, `metrics.csv`,
`changes.csv`, per-image edge/distance images and four-panel candidate overlays
(yellow original, cyan crop). `template_scores.csv` logs all circle proposals;
`boundaries.csv` separates crop geometry from stored detection geometry.
`refined_centre_metrics.csv` evaluates hypothetical centre changes separately.
Built-in synthetic checks cover a shifted/undersized circle, flat background,
neighbour suppression and clipped image boundaries. These results follow observed
test errors and remain exploratory. Neither experiment changes the application.

## Row-based radius support for competing cue claims

```sh
./build-local/BallExperiments --row-cue-size
```

This final isolated experiment lives in `identityHeuristics.cpp` and defaults to
`output/experiments/row_cue_size/`. A robust line is fitted once from the 16
pocket-filtered AllBalls Hough circles, without annotations. Expected radius is
`max(1,20.274190 + 0.00967744 * originalImageRow)`. Every horizontal location uses
the same expectation at a given row. The slope is the median pairwise radius
change per row for samples at least ten rows apart; negative slopes are clamped
to zero. The intercept is the median residual.

AllBalls covers rows143.25–197.25, so predictions outside this narrow band are
linear extrapolation. Training mean absolute radius residual is1.04pixels.
Circle underestimation and horizontal perspective variation can affect the fit;
the row approximation does not establish that true sizes are identical across a
row. The model and all inference settings are frozen before loading test labels.

Size rules activate only when an image has at least two candidates originally
classified White by K2. This activates on Table1, Table3 and Table6 in this run.
The soft variant multiplies every candidate's cue score by
`min(1,originalRadius/expectedRadius)^2`, leaving numbered scores unchanged. The
stronger variant directly reserves cue for the original White candidate with
the largest depth-adjusted radius and excludes all other cue claims. This is an
explicit forced heuristic, not a probability. Each is tested with original crops
and with +1 padding on original White crops. Size always uses original Hough
geometry; padding does not increase a candidate's size score.

| Method | Correct identities | Fixes / regressions vs +1 |
| --- | ---: | ---: |
| Unpadded unique control | 75/86 | 0 / 4 |
| +1 unique control | 79/86 | 0 / 0 |
| Soft row cue support | 78/86 | 1 / 2 |
| +1 and soft row support | 79/86 | 0 / 0 |
| Reserve largest depth-adjusted cue | 78/86 | 1 / 2 |
| +1 and reserve largest cue | 80/86 | 1 / 0 |

The last variant correctly identifies all six genuine cues and preserves every
previously correct +1 identity. In Table1, the real cue changes14->cue, while
striped14 changescue->Unknown because its eight body pixels remain below the
unchanged histogram minimum12. Thus this resolves the cue identity but does not
recognise the striped partner. It produces two Unknowns overall (one genuine
ball and the existing unmatched Table2 false detection). AllBalls remains16/16;
detection stays86TP/2FP/4FN. Results are exploratory comparisons on previously
inspected evaluation images, with no production integration or additional search.

Outputs include `row_model.yml`, `training_samples.csv`, a per-row lookup
`row_radius.csv`, and `row_radius_model.png` highlighting the training row span.
`cue_size_scores.csv` records original radius, expected radius, activation,
extrapolation, applied multiplier and original/weighted cue scores. Standard
metrics/predictions/changes, one comparison sheet per relevant candidate, and
`final_results/<method>/<image>/` are provided. Synthetic checks cover outlier-
robust fitting, same-row fallback, the positive-depth prior, factor monotonicity
and a cue conflict with preserved stripe alternatives.

## Direct HSV pixel recognition

```sh
./build-local/BallExperiments --hsv-pixel-recognition
```

This independent experiment in `hsvPixelRecognition.cpp/.h` replaces K2 type
evidence with direct per-pixel HSV counts. It uses the existing thresholds without
searching: light V>=170/S<=165, dark V<=90 outside light, and coloured pixels as
the remainder. Masks are disjoint and exhaust the circular ROI. Type proportions
use the full mask; the frozen histogram query still uses its separate erosion3
and minimum12 body pixels. Cue requires85% light; solid requires70% nonlight;
otherwise a mixed crop is striped.

The experiment compares K2 and direct HSV at each stage, with the same AllBalls
histograms and assignment backend. Each frontend's own original White labels
control its padding and ambiguous-cue triggers; the HSV frontend does not rely
on K2 labels. The optional size rule loads the previous AllBalls row model
unchanged. No training images, thresholds or detection geometry are changed.

| Stage | K2 correct identities | Direct HSV correct identities |
| --- | ---: | ---: |
| Local type + histogram colour | 72/86 | 74/86 |
| Unique assignment | 75/86 | 74/86 |
| +1 padding and unique assignment | 79/86 | 78/86 |
| +1 padding and largest depth-adjusted cue | 80/86 | 79/86 |

Direct HSV fixes Table6's red solid (11->3), but turns Table1's maroon stripe
(15) into solid7. Its light fraction drops from41.66% under clustered evidence
to25.10% under pixel thresholds, crossing the solid boundary. In the global
variants, this also allows another Table1 candidate (actual12) to claim15.
Thus the final HSV variant corrects one identity and loses two relative to its
paired K2 control. AllBalls remains16/16 for all eight methods. This is a fixed-
threshold exploratory comparison; no further tuning or production integration
is performed.

Default outputs are in `output/experiments/hsv_pixel_recognition/`, including
`summary.md`, automatic metrics/predictions/changes, `mask_counts.csv`, frozen
models, and standard `final_results/<method>/<image>/`. Each ball has one
`candidate_comparisons/<image>/candidate_N.png` sheet and one
`pixel_masks/<image>/candidate_N.png` sheet for original/padded crops and all
three pixel masks. Synthetic checks verify partitioning, exact fractions, mask
exclusions and empty/white cases. K2 controls exactly reproduce previous outputs.

## Report figures for the row/radius model

```sh
MPLCONFIGDIR=/tmp/pool-matplotlib python3 experiments/plotRowRadius.py
```

This reads the saved AllBalls training samples from `--row-cue-size`, rather
than fitting to evaluation data. Matplotlib/Python are used only for offline
report graphics; application and recognition implementations remain C++/OpenCV.
Optional `--data-directory`, `--output-directory` and `--image` arguments are
available. Defaults create three figures in `report/assets/row_radius/`, each
as a 300dpi PNG and vector PDF:

- `row_radius_samples`: full image with the training-row band, plus a zoom with
  circle outlines, centre crosses, detection IDs and horizontal row guides.
- `row_radius_fit`: radius-versus-row samples and fitted line, beside the model
  over the full image height; dashed segments explicitly mark extrapolation.
- `row_radius_estimation`: all71 eligible pairwise slopes and their median, plus
  the16 individual intercept estimates and their median.

The sample and fit figures provide the clearest main-report explanation. The
estimator figure is useful when space permits more mathematical detail. Captions
are supplied in `captions.md`; `pair_slopes.csv` and `intercept_samples.csv`
preserve the plotted calculations. Figures contain no code/library names.
Plots reconstruct the model from six-decimal saved samples and verify agreement
with saved expected radii within1e-4pixels. Report pages are not modified.
