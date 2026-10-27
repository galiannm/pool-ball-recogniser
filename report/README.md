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

The table counts were checked against the recorded correctness flags. Empty
trailing CSV fields were ignored; no application source or experiment outputs
were changed while creating this report.
