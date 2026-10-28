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
