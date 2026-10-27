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
