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
