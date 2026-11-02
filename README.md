# Pool-ball localisation and recognition

C++17 / OpenCV 4 assignment. Run from the project root so relative paths resolve.
The report configuration is now self-contained in `myApplication.cpp`, callable
through `MyApplication()`. See [SUBMISSION.md](SUBMISSION.md) for the method/source
mapping, comparisons, outputs, validation and annotation basis.

## Build and run

```sh
cmake -S . -B build-local
cmake --build build-local --target PoolBallSubmission PoolBallDetection -j 4
./build-local/PoolBallSubmission
```

For a fresh build on this Mac:

```sh
cmake -S . -B build-local \
  -DOpenCV_DIR=/Users/manon/opencv-4.12.0-install/lib/cmake/opencv4 \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build-local
```

Use the appropriate OpenCV/toolchain paths on another machine.
`PoolBallSubmission` compiles only `myApplication.cpp` and OpenCV. Its optional
standalone `main()` calls the same `MyApplication()` used by the existing menu.
The application processes AllBalls and Table1--6 automatically without GUI pauses.

```sh
POOL_BALL_MODE=comparisons ./build-local/PoolBallSubmission
POOL_BALL_MODE=grid-search ./build-local/PoolBallSubmission
```

The default mode is `selected`. Comparisons reproduce the recognition ablations
and rejected alternatives discussed in the report. Grid search is opt-in and
retains its historical `output/grid_search.csv` output path.

## Existing experimental menu

Run `./build-local/PoolBallDetection` for the menu. Enter a number and press Enter;
one task runs per invocation. Options 2--9 retain the modular experimental
implementations, and their settings/results can differ from the submission.

| Choice | Task |
| --- | --- |
| 1 | Consolidated application, final report configuration and automatic evaluation |
| 2 | Evaluate default Hough parameters on AllBalls |
| 3 | Search Hough parameters on AllBalls |
| 4 | Evaluate modular detection on Table1--6 |
| 5 | Draw CSV ground-truth centres |
| 6 | Extract AllBalls crops |
| 7 | HLS colour diagnostics |
| 8 | Compare K-means cluster counts |
| 9 | Modular recognition/CCA diagnostics |
| 0 | Exit |

`BallExperiments` remains an independent historical runner; its commands are
listed in [experiments/README.md](experiments/README.md).

## Data, results and report

Inputs are `Media/AllBalls.jpeg` and `Media/Table1.jpeg` through `Table6.jpeg`.
`data/ground_truth/BallLocations.csv` is the existing working annotation file;
`BallLocation_original.csv` is the original backup. They differ, including
re-estimated centres, coordinate ordering and identity associations. Neither was
changed during consolidation. Reported metrics use the working file, as disclosed
in the report; they also reflect evaluation-informed method selection.

Predictions are exported before annotations are read for evaluation. Source
circles use `(x,y,radius)` image pixels. Table coordinates use column `x`, row `y`,
with top-left origin. The supplied corners map to `(0,0)`, `(1433,0)`, `(1433,682)`
and `(0,682)` in a 1434-by-683 image, approximately one pixel per millimetre.
Only centres are transformed for position scoring. Diagrams use nominal 43 mm
ball diameters rather than source-image radii.

`output/myApplication/final_results/<image>/` contains final predictions as four
PNGs and `balls.csv`. Its parent directory contains automatic metrics, matches,
confusion counts, recognition evidence and the trained model. Optional comparison
results go in `output/myApplication/comparisons/`. CSVs are overwritten per run;
old optional folders are retained. See `SUBMISSION.md` for the exact output map.

`report v2/main.tex` and `main.pdf` are the revised eight-page report. Build with
`make -C 'report v2'`. The original `report/` and modular experiment sources are
retained. Both the submission and report acknowledge ChatGPT/Codex assistance.
