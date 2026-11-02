# Submission and report consistency

`myApplication.cpp` is self-contained C++17/OpenCV 4. It includes all processing,
training metadata, output rendering, evaluation and alternatives discussed in
`report v2/main.tex`. It has no project-header or saved-model dependencies.
The original modular sources and experiments remain available for provenance;
changing those files or `config.h` does not change the consolidated submission.
The constants used by the submission are inside `myApplication.cpp`.

Build and run from the project root:

```sh
cmake -S . -B build-local
cmake --build build-local --target PoolBallSubmission PoolBallDetection -j 4
./build-local/PoolBallSubmission
```

The existing menu's application option also calls `MyApplication()` directly.
It now runs all seven images without image-window pauses. A supplied harness can
link just `myApplication.cpp` and OpenCV and call `void MyApplication();`.
`POOL_BALL_STANDALONE` optionally adds the small command-line `main()` used by
`PoolBallSubmission`; it is not required by the assignment harness.

```sh
POOL_BALL_MODE=comparisons ./build-local/PoolBallSubmission
POOL_BALL_MODE=grid-search ./build-local/PoolBallSubmission
```

The default selected mode uses K=2, HSV histogram models, soft type compatibility,
unique identity assignment, +1 pixel analysis padding on originally White crops,
and largest-original-radius cue reservation when two or more original crops are
White. Training uses only the 15 labelled numbered-ball crops from AllBalls;
training crop coordinates/labels are included in the file. No Table1--6 labels
enter inference. The discarded depth fit is computed only in comparisons mode.

Comparisons mode includes fixed HSV rules, histogram recognition, independent
scores, unique assignment, padding, raw/adjusted cue selection, original K2 CCA,
direct HSV pixel counts, minimum/outer chamfer refinements, Hough-seeded/hybrid
watershed and cluster-first local Hough/watershed. The original 480-configuration
Hough search is opt-in and does not retune the default application automatically.
The watershed seeded profile keeps the original Hough centres; its segmentation
is not used to invent additional centres. The hybrid and guided profiles expose
the actual alternative detection outputs.

## Main pipeline versus trials

The file has a numbered section map at the top. Function definitions precede
calls, so the copied function prototypes have been removed; structs remain as
shared data types. `runMainPipeline()` contains the five selected per-image
stages. `runExperimentalTrials()` contains the optional comparisons and returns
separate result profiles. `runSubmission()` handles training, calls the main
pipeline, optionally adds trial profiles, then exports and evaluates results.
CCA and the fixed-HSV baseline are grouped with the other optional algorithms.
Histogram processing is in `colour_model`, reflecting its role in the final method.

## Outputs

Each run updates files under `output/myApplication/`:

- `final_results/<image>/`: four final PNGs and `balls.csv` with source circles,
  table centres and identities (cue 0, unknown -1).
- `processing/<image>/`: masked image, blurred image, raw and filtered circles.
- `metrics.csv`: per-image metrics and pooled Table1--6 totals, with AllBalls
  marked as training resubstitution. Comparisons mode includes all methods.
- `predictions.csv`: identities and table positions without annotation labels.
- `evaluation_matches.csv`: matched identities/distances, false positives and misses.
- `identity_confusion.csv`: matched Table1--6 identity confusion counts.
- `recognition_evidence.csv`: original/padded light fractions, body-pixel counts
  and cue reservation decisions.
- `trained_model.yml`, `configuration.txt`: model and settings from this run.
- `comparisons/`: optional alternative result images and segmentation stages.

Run modes overwrite their shared CSVs. Old optional output folders are retained;
use the current `metrics.csv` and `configuration.txt` to identify a run's contents.
The grid search retains its historical `output/grid_search.csv` destination.
Missing annotations do not prevent prediction exports; the affected image's
metrics are skipped with a diagnostic.

## Source mapping

| Report processing | Original sources consolidated into the submission |
| --- | --- |
| Table mask, Gaussian smoothing, circular Hough, pocket rejection | `detection/ballDetection.cpp`, `detection/geometricFilter.cpp` |
| Table homography | `geometry/perspectiveTransform.cpp` |
| Crop extraction, K=2, fixed-rule baseline | `recognition/extractBall.cpp`, `recognition/kMeansSegmentation.cpp`, `recognition/ballTypeClassifier.cpp` |
| HSV histograms | `experiments/ballColourHistograms.cpp` |
| Soft scores, Hungarian assignment, padding, raw/adjusted cue rule | Inference portions of `experiments/identityHeuristics.cpp` |
| Original CCA | `recognition/connectedComponentAnalysis.cpp`, `recognition/ballTypeClassifier.cpp` |
| Watershed | Algorithm portions of `experiments/watershedDetection.cpp` |
| Cluster-first local Hough/watershed | `experiments/segmentationGuidedDetection.cpp` |
| Chamfer refinement | Algorithm portions of `experiments/chamferRefinement.cpp` |
| HSV pixel-count alternative | Pixel evidence from `experiments/hsvPixelRecognition.cpp` |
| Hough parameter search | `evaluation/gridSearch.cpp` |
| Automatic matching, images and CSV export | `evaluation/evaluation.cpp`, `io/output.cpp`, `io/finalResults.cpp`, `visualisation/visualisation.cpp`; consolidated batch runner |

## Verified measurements and annotation basis

The selected configuration reproduces all 35 final PNG/CSV artifacts from
`output/experiments/cue_radius_comparison/final_results/largest_raw_radius/`
byte-for-byte, including AllBalls. Table1--6 give 86 TP, 2 FP, 4 FN, mean distance
4.914978163, 80/86 correct matched identities and 80/90 end-to-end identification.
The recognition progression is 68, 72, 75, 79, 80 correct identities. The ablation
adding assignment also adds soft type scores, which the report now states.
CCA gives 69; outer chamfer 77; HSV pixels with adjusted cue selection 79.
Cluster/local-Hough gives 86 TP, 16 FP, 4 FN (F1 89.5833%).

These measurements use `data/ground_truth/BallLocations.csv`, the pre-existing
working annotations, not the untouched `BallLocation_original.csv`. The working
file changes coordinate ordering and identity associations, and re-estimates
some centres. The consolidation does not change either CSV. The exact changed
pairs are recorded in `output/myApplication/annotation_revisions.csv` for review.
Their presence is disclosed in the report; the revisions need independent
justification before claiming results against the original supplied annotations.

AllBalls alone supplies the colour models, but the prior method selection used
feedback from Table1--6. The report therefore calls the results development-set
measurements. They do not establish independent performance on unseen images.

## Report and AI assistance

`report v2/main.tex` builds to eight pages with 12-point body text and 25 mm
margins. It includes the all-six-image figure, parameter/score definitions,
automatic measurements, failures, robustness discussion and AI acknowledgement.
The original `report/` is unchanged. Build with `make -C 'report v2'`.

OpenAI ChatGPT/Codex assisted with experimental design, C++ implementations,
debugging, comparisons, graphics and drafting; this revision consolidated the
reported methods, added automatic batch evaluation/comparisons, verified outputs,
and revised the report. The author must review the submitted code, results,
annotation revisions and final wording, retaining an accurate acknowledgement.
