# Pool ball detection report

`main.tex` is an editable draft based on the current source and saved results.
Replace the author/student-number placeholder and review the text before submission.

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

The baseline table counts were checked against the recorded correctness flags.
Empty trailing CSV fields were ignored. The added recognition experiments are
separate from the unchanged main application. Table1--6 are explicitly treated
as tuning images for this later search, rather than independent test data.
