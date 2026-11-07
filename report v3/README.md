# Report v3 — eight-page submission draft

This version applies the agreed shortening and restructuring to the report v4 narrative. It uses 12 pt text and 25 mm margins, with four pages of theory/application and four pages of results/analysis, including references. Report v4 and application sources are unchanged.

Build from the repository root with:

```sh
make -C 'report v3'
```

Open `main.pdf`. The build checks referenced image availability, selected metric totals, the eight-page limit and layout overflow. Requires Python 3, pdflatex and pdfinfo.

## Structure

1. Masking, smoothing, Hough parameters, pocket filtering and extraction.
2. K-means type evidence and reference HSV histograms, including bin choices and similarity formula.
3. Soft type compatibility, unique assignment, cue correction and homography.
4. Alternative type/colour and segmentation approaches, with diagnostic images.
5. Evaluation definitions and per-image/pooled final performance.
6. Detection and recognition experiment comparisons.
7. All six final labelled views, enlarged failure details and analysis.
8. Robustness, limitations, proposed improvements and references.

There are no missing-image placeholders, library calls or application-code references in the report. The old gallery and missing-export files remain as unused development material; the PDF does not include them. The asset checker now audits only the figures actually referenced by `main.tex` and does not regenerate those old appendices.

## Evidence and figure provenance

All figure sources are under `output/final-results/`. Several figures use LaTeX clipping to show enlarged regions of existing images; they are presentation crops, not newly generated experiment outputs. The circular extraction illustration uses the inverse table homography to locate Table3 candidate 8 in the saved masked photograph. Cue examples similarly use the saved Table1 photograph. Original experimental images and numerical outputs are unchanged.

Final metrics come from `metrics.csv`, `evaluation_matches.csv` and `recognition_evidence.csv`. Recognition development comparisons follow `results_analysis.md`; reconstructed type comparisons remain distinct from the original fixed-detection recognition comparisons. The updated cluster-local Hough/watershed results already present in v3 are retained: 86 TP/16 FP/4 FN before watershed and 83 TP/11 FP/7 FN after it. They replace the older additive guided-search comparison in v4.

The AllBalls grid-search row comes from `output/grid_search.csv` and remains a training/parameter-selection result. AllBalls is excluded from the six-image development totals. The report discloses repeated development-set selection and does not claim independent test performance.
