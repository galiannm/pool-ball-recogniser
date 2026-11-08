# Report v5

Eight-page report for the current `myApplication.cpp` pipeline. The existing
`assets/pipeline.png` diagram is retained unchanged. Body text is 12 point with
normal line spacing and 25 mm A4 margins. All eight pages, including references,
are part of the submitted PDF.

Build from the project root:

```sh
make -C 'report v5'
```

The build audits figures and current result totals, runs LaTeX twice, then checks
that the PDF has eight pages and no overfull boxes. Open `main.pdf` for submission.
No application code is modified or application pipeline rerun by this build.

## Evidence and illustrations

- Final measurements and labelled views: `../final_output/`.
- CCA comparison: `../final_output/experiments/k2_cca/`.
- Simplified guided Hough/watershed detection:
  `../final_output/experiments/guided_hough_watershed/`.
- Recognition with all three detection variants:
  `../final_output/experiments/guided_recognition/recognition_comparison.csv`.
- `assets/candidate_evidence.csv` and `assets/colour_scores.csv` record diagnostic
  measurements on the saved final circles using the current application methods.
- `assets/group_counts.csv` contains the per-identity chart counts.
- `image_manifest.json` records SHA-256 hashes for every figure read by the report.

`export_evidence.cpp` includes the current application without enabling its standalone
entry point. Compile it with C++17 and the same OpenCV flags, then run its executable
**from the project root** to refresh the report-only diagnostics. It reads existing
final prediction CSVs and writes only to `report v5/assets/`. It does not rewrite
application results or source. An executable was built in `/tmp` during authoring;
no additional executable is required to build the submitted report.

`build_figures.py` composes those diagnostic crops and existing outputs into PDF
figures. It requires NumPy, Matplotlib and Pillow. Run it with a Python environment
containing those packages, from any directory. On the authoring machine:

```sh
MPLCONFIGDIR=/tmp/report-v5-mpl /usr/local/bin/python3 'report v5/build_figures.py'
```

Process montages, plots and enlarged crops are labelled as such. They are not new
method variants. The report does not use histogram backprojection or historical
ablation numbers. Group colour/type evidence is distinguished from final identity
performance; final identity comparisons include false numbered outputs.

## Layout

Pages 1-4 describe the method and experimental designs. Pages 5-8 give evaluation,
per-image outcomes, success/failure illustrations, controlled experiments, group
counts, robustness and improvements. The final page includes references and an
assistance acknowledgement.

The optional `gallery.tex` and figure-status `missing_figures.*` files are not
included in `main.tex`. All figures needed by this report are present; the audit
fails if one is missing. New results require updating the prose, figures and audit
expectations together, rather than silently substituting an old run.