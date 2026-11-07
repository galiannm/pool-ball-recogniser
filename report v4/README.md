# Report v3

Narrative report following the supplied new draft, with evaluation introduced early and validation/decision explanations integrated into detection, recognition and visualisation.

Build with `make -C 'report v3'`. Open `main.pdf` to review. No application runs or application source edits are performed by this build.

## Image provenance

Every imported image is referenced directly beneath `../output/final-results/`; no image is copied, cropped, manufactured or borrowed from report v2, Media or older experiments. The conceptual pipeline is drawn directly in LaTeX and is not an experimental image. Available panels include preprocessing, pocket filtering, watershed, guided Hough, HLS/K-means/CCA, histogram diagnostics and all seven final labelled/diagram pairs.

Missing exports appear as visible boxes in the PDF and are documented in `missing_figures.md`. Proposed paths make it possible to add code exports and rebuild without changing report references. Do not fill a missing historical control by renaming the selected output. Export the actual configuration and associated measurements.

`check_assets.py` regenerates the gallery, image manifest (with SHA-256 hashes), and missing-image lists. It checks image paths remain under final-results and verifies the selected six-image metric totals. The `.fls` recorder file also records the graphics actually read by LaTeX.

## Measurement provenance

- Final run: `output/final-results/metrics.csv`, `evaluation_matches.csv`, `recognition_evidence.csv` and final PNG exports.
- Reconstructed experiments: `historical_detection_experiments.csv` and `historical_recognition_experiments.csv` in the same directory.
- Earlier controlled recognition comparisons: explicitly marked historical, as summarised in `output/final-results/results_analysis.md`. These summaries are not newly rerun controls, and several intermediate images are missing.
- Hough search numbers: selected row of `output/grid_search.csv`, explicitly identified as an external numerical source. Its selected AllBalls result is 16 TP/1 FP/0 FN; images still come exclusively from final-results.
- Working annotations: `data/ground_truth/BallLocations.csv`; supplied original: `BallLocation_original.csv`. Annotation changes and repeated development-set selection are disclosed.

The report removes the unsupported Gaussian-versus-linear claim and distinguishes reconstructed expected-radius selection (75/86) from original row-adjusted radius selection (80/86). It does not claim an unseen test evaluation.

This is an expanded draft with visible missing exports and an appendix, not a fixed eight-page submission. The original `report`, `report v2`, `ai_report`, application sources, and final-results exports are left unchanged.
