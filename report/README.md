# Pool ball detection report

`main.tex` is an editable draft based on the current source and saved results.
Review the author details and draft text before submission.

The introduction, current baseline results, final recommendation and conclusion
reflect the completed experiment comparisons. `sections/final_configuration.tex`
summarises the recommended original detector + tuned K=6 + solid-only CCA.
The report explicitly records that the main application still runs the original
K=2 recognition baseline; this report update does not integrate experimental code.

The histogram-only candidate-verification experiment is in
`sections/backprojection_experiment.tex`. Its outputs are under
`../output/experiments/cloth_backprojection/`. After running
`BallExperiments --backprojection`, refresh tables and the diagnostic figure with
`python3 experiments/writeBackprojectionReport.py`. The selected conservative
setting is frozen from AllBalls recall screening before Table1--6 evaluation;
the report does not claim precision optimisation on a reference image with no
post-pocket false detections.

Build from this directory:

```sh
make
```

Or from the project root:

```sh
make -C report
```

The PDF is `build/main.pdf`. The build requires a LaTeX installation providing
`latexmk`, `pdflatex`, and the standard packages listed in `main.tex`.

Without latexmk, run this command twice from `report/`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The figures in `assets/` and tables in `tables/` are local snapshots, so the
report can compile without access to the rest of the project. They do not
automatically update when experiments are rerun.

Result sources:

- `../output/validation/validation_results.csv`
- `../output/validation/recognition/Table1/recognition_results.csv` through Table6
- `../output/experiments/radius_padding/*/results.csv`
- `../output/experiments/recognition_grid_search/` (metrics, K comparison,
  finalist parameters, confusion matrices and per-candidate figures)
- `../output/experiments/cca_grid_search/` (original and tuned recognition controls)
- `../output/experiments/segmentation_guided_detection/metrics.csv` (current rerun
  detector baseline and the final detector alternatives)

The headline detection table (`tables/detection_current.tex`) uses the `hough`
rows of the last source: 86 TP, 2 FP, 4 FN, F1 96.63%. The older validation CSV
records different historical false-positive counts and is no longer the source
of the report's current headline detection result. The cause of that discrepancy
has not been established; it is not attributed to a specific implementation change.

The detailed exploratory experiment discussion is in
`sections/recognition_experiments.tex`, included from `main.tex`.
Refresh its generated tables and example figure from the project root with
`python3 experiments/writeRecognitionReport.py` after running both experiments.
The surrounding prose contains run-specific counts and interpretations and
must be reviewed separately when the search changes.

The final CCA experiment is described in `sections/cca_experiment.tex`.
Its source results are in `../output/experiments/cca_grid_search/`.
Refresh its tables and example figure with `python3 experiments/writeCCAReport.py`
from the project root after running `BallExperiments --cca-grid`. The prose
again contains run-specific interpretations and is not regenerated automatically.

The fixed watershed detection comparison is in `sections/watershed_experiment.tex`.
Its results are in `../output/experiments/watershed_detection/`; run
`python3 experiments/writeWatershedReport.py` from the repository root to update
its tables and diagnostic figures after `BallExperiments --watershed`.
The section uses the same rerun detector baseline as the current headline table;
the older saved validation CSV is identified separately as historical.

The subsequent segmentation-first/two-Hough experiment is described in
`sections/segmentation_guided_experiment.tex`. Its outputs are under
`../output/experiments/segmentation_guided_detection/`. Run
`python3 experiments/writeSegmentationGuidedReport.py` after
`BallExperiments --segmentation-hough` to refresh its tables and Table4 recovery
figure. Recovered annotations and aggregate regressions are both reported.

The baseline table counts were checked against the recorded correctness flags.
Empty trailing CSV fields were ignored. The added recognition experiments are
separate from the unchanged main application. Table1--6 are explicitly treated
as tuning images for this later search, rather than independent test data.
