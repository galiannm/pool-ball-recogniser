# Report v6

A separate version focused on the final pipeline and the decisions that produced it.
**Report v5 is unchanged.**
The report retains 12-point text, normal spacing, 25 mm margins and the eight-page
upper limit. It keeps the original pipeline diagram and the repository link.
The AI-use statement is included at the end of the main report within those eight
pages; the separate `ai_report` retains the detailed prompts and usage account.

The current manuscript is a full prose rewrite from the author's development
notes, including the evaluation and failure analysis. The previous compressed
descriptions have been replaced with paragraphs describing the attempts,
observations and reasons for the next step. All final table images remain unchanged.

The latest revision follows the author's development notes: objective and stages,
manual tuning to grid search, HLS to K-means, representative colour to histograms,
hard to soft decisions, unique assignment, and two distinct cue corrections.
The rejected detection trials are discussed as actual experiments. Their
measurements were rechecked from the existing methods by a report-only exporter.
CCA remains outside the final pipeline; it is now discussed as future work,
alongside lighting correction and guided Hough/watershed. Historical diagnostics
remain on disk. The author's subsequent edits to the structure and wording are retained.

Checked against the supplied report brief: pages 1–4 cover processing theory and
pages 5–8 results and analysis. There is no separate title page. The method uses
technique descriptions and parameters without source filenames or library calls.
The AI-use disclosure and repository link remain at the end. Metrics use the
working annotations in `data/ground_truth/BallLocationsNew.csv`.

Changes from v5:

- Gives the design progression from HLS pixels to K=2 regions, representative hue
  to HSV histograms, and hard decisions to soft scores and unique assignment.
- Uses saved controlled comparisons, with AllBalls excluded from pooled results.
  The original manual Hough trial is described without inventing its parameters
  or presenting the retrospective fixed baseline as that trial.
- Briefly reports the earlier CCA and guided-Hough limitations when discussing
  future work on lighting, spatial masks and crowded-ball segmentation. The
  detailed rejected-trial montage remains a diagnostic asset.
- Combines missing-colour and neighbour-contamination failures into one padding
  trade-off, retaining the three failure crops. A compact per-identity count plot
  illustrates group performance. The closing text explicitly limits the evidence
  to six development images from one fixed table view, followed by next steps and
  a short conclusion.
- Explains exactly how AllBalls crops produce eight fixed HSV body-colour
  references, with equal weighting of normalised solid/stripe examples.
- Uses actual AllBalls crops for the mask, clustering and body-pixel illustration.
- Explains why white pixels are excluded, what histogram distributions retain,
  and the limits of colour matching.
- Explains soft type support and maximum-score Hungarian assignment using a real
  Table2 conflict: the purple stripe takes its second choice, identity 12, while
  the purple solid retains identity 4.
- Explains continuous type compatibility in prose and an equation. The type-support
  plot and purple-crop panel were removed from the PDF to make room for the
  development narrative. Their verified diagnostic assets remain available locally.
- Illustrates table masking, greyscale smoothing, raw circle detection and pocket
  filtering separately in the detection montage.
- Draws schematic hue histograms for solid 4, stripe 12 and their equal-weight
  purple reference. Their invented bar heights illustrate averaging; they are
  not measurements of the training images. The purple peaks lie between blue
  (240 degrees) and magenta (300 degrees). Matching uses full HSV histograms.
  Below, two examples use only native application exports: ROI crops,
  `identity/hard_hue/balls.csv` for the earlier colour (from the predicted number),
  and `recognition/Table*/ball_colours.csv` for the best colour and similarity.
  The next-best score is the exported similarity minus the exported margin;
  its colour is not exported and is therefore not labelled. Table2 ball 4 changes
  from blue to purple, and Table1 ball 10 from purple to blue. The old heatmap,
  measured reference bars and report-only top-three score comparison are removed.
  There are eight colour corrections and no losses
  across the 80 matched numbered balls (65 to 73 correct).
- The type-development figure shows the actual cue,
  solid 1 and stripe 9 through HLS classes, K-means, light masks and coverage bars
  leading to the initial type. Gold denotes light and navy non-light; measured
  HLS light fractions make its failure on the cue and stripe visible.
- Separates crop padding (75 to 79 correct identities) from cue reservation (79
  to 80), and labels the Table3 example with its actual before/after predictions.
- Shows the actual `table_numbered.png` and `table_diagram.png` outputs for every
  Table1–6, including the application's larger labels and magenta/yellow outlines.
- Splits Figure 8 into three two-table bands (`final_views_1_2.pdf`,
  `final_views_3_4.pdf`, `final_views_5_6.pdf`) so parts can fit on page 6.
  Continued floats retain one figure number, with an invisible first caption
  reserving that number when only the final band has a visible caption.
- Refreshes the detection montage from the latest saved raw/filtered circle images.
- Places TP/FP/FN, correct identities/annotations and centre error above each
  final pair, replacing the separate per-image results table.
- Uses the requested course, CP-Algorithms, scikit-learn and MathWorks sources. See
  [references.md](references.md) for scope, links and exact sections.

Build from the project root:

```sh
make -C 'report v6'
```

This builds **report v6/report_v6.pdf** and an identical **main.pdf**, after checking
page count and layout. A separate `.build` job avoids colliding with an editor's
automatic `main.pdf` compilation; final copies are replaced atomically. It never
runs or changes the production pipeline. The audit reads existing final results
and report diagnostics, plus `final_output/experiments/past_designs` for the
controlled design comparisons. The earlier unchanged watershed detection trial
is checked against `more_detailed_implementation/output/experiments/watershed_detection/metrics.csv`.
The guided/local-watershed figure uses report-only diagnostics. The type figure
uses actual AllBalls crops and masks verified against the measured cluster areas.
Lighting experiment outputs are not used.

`build_figures.py` builds only v6 figures and the measured assignment CSV.
With NumPy/Matplotlib/Pillow in the Python environment, run:

```sh
MPLCONFIGDIR=/tmp/report-v6-mpl /usr/local/bin/python3 'report v6/build_figures.py'
```

`export_reference_examples.cpp` reuses existing application methods, reads saved
AllBalls circles and matches their labels, then writes only v6 diagnostics,
including historical HLS pixel classes, K-means light masks and measured coverage
for the cue, solid and stripe. It
checks all eight reference example counts and verifies that the two yellow crops
average to the actual yellow model. It also exports the two yellow descriptors
and their reference to `assets/yellow_histograms.csv` as retained diagnostics.
Build it with C++17 and the same OpenCV/SDK
flags as the application, without `POOL_BALL_STANDALONE`, then run it from the
project root. The exporter is not required for a normal report rebuild.

`export_development_examples.cpp` reuses the existing methods in `myApplication.cpp`
and `experimenent.cpp`, but never calls their batch output runners. It writes only
v6 diagnostics, `development_comparisons.csv`, `cue_decisions.csv` and
`cca_changes.csv`. The audit verifies its 76/73 K-means/CCA type counts, 75/79/80
identity counts, and 86/2/4, 89/8/1 and 88/6/2 detector counts. Compile with the same C++17
and OpenCV flags as the reference exporter, without `POOL_BALL_STANDALONE`.
The figure builder and audit use its saved outputs, so rerunning it is optional.

`export_colour_comparison.cpp` reads saved circles and rechecks the representative-
colour baseline and full histogram comparison on the same 80 numbered balls. It
exports the actual training histograms for all eight colours, the crop/body views
for corrected cases, and `colour_method_evidence.csv`. The audit verifies these
against the earlier saved similarities and the 65/73 totals. These are retained
diagnostics; the current reference figure does not use this exporter's values.
Its reference histograms are explicitly schematic and its comparison scores come
from native application exports. The exporter retains
a local parameter-validation fallback because the current application working
copy calls `validate` without defining it; it does not modify that working copy.
Build it with the same C++17/OpenCV flags as the other report exporters.

Older numbers in the notes were not copied unchecked. The current AllBalls
diagnostic contains 17 raw circles and 16 after filtering, not three raw false
positives. Earlier standalone watershed trials and the later guided-watershed
variant are distinguished. The unverified linear-penalty comparison was omitted.

Pages 1–4 follow the development notes through detection, rejected segmentation,
recognition, identity assignment, cue corrections and visualisation. Page 5 defines
the metrics and compares successive designs. Page 6 shows
all rectified-image/diagram pairs, page 7
analyses failures, and page 8 explains the rejected trials, group results,
robustness and references.

`image_manifest.json` records figure-source hashes. The example in
`assets/assignment_example.csv` uses measured Table2 candidates 6 and 11. The composite
`assets/final_views.pdf` directly combines the saved `table_numbered.png` and
`table_diagram.png` files from `final_output/final/Table1` through `Table6`.
The report does not redraw the labels, circle outlines or ball colours.
Magenta marks correct identities; yellow marks identity errors and false
detections, using ground truth only for the evaluation overlay.
`assets/figure_sources.json` records the exact source hashes used in the detection
and final-view montages. If the application outputs change, the audit requests
that `build_figures.py` be rerun before the report is rebuilt. No application
output files are modified.
