# Pool ball detection

C++17 / OpenCV 4 assignment. Run commands from the project root so relative
input and output paths resolve correctly.

## Build and run

With an existing configured build:

```sh
cmake --build build-local
./build-local/PoolBallDetection
```

For this Mac's OpenCV installation and compatible SDK, configure a fresh build with:

```sh
cmake -S . -B build-local \
  -DOpenCV_DIR=/Users/manon/opencv-4.12.0-install/lib/cmake/opencv4 \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build-local
```

On another machine, use its OpenCV installation and toolchain instead.

## Menu

| Choice | Task |
| --- | --- |
| 1 | Application: show raw and filtered detections for all seven photos |
| 2 | Baseline: evaluate the default parameters on AllBalls.jpeg |
| 3 | Grid search: search Hough parameters on AllBalls.jpeg |
| 4 | Validation: evaluate filtered detections on Table1–Table6 |
| 5 | Sanity check: draw the CSV ground-truth points |
| 6 | Extraction demo: save filtered ball crops from AllBalls.jpeg |
| 7 | HLS colour analysis: save a diagnostic grid per ball |
| 8 | K-means demo: compare K=2–5 in one diagnostic grid per ball |
| 9 | Recognition validation: save segmentation and CCA grids per matched ball |
| 0 | Exit |

Enter a number and press Enter. Each invocation runs one selected task. In the
application, press a key in the image window to advance to the next photo.

## Source layout

```text
main.cpp                     Terminal menu and task dispatch
myApplication.cpp/.h         Interactive application runner
config.h                     Default parameters, geometry, datasets and paths
detection/                   Hough detection, masking pipeline and pocket filter
geometry/                    Table corners and perspective transformations
recognition/                 Ball cropping and future recognition code
evaluation/                  Metrics, CSV loading and experiment runners
visualisation/               Drawing detection and ground-truth overlays
io/                          Checked image and CSV output helpers
data/ground_truth/           Working annotations and the original backup
Media/                       Input photos
output/                      Generated images, crops and CSV results
professor_utilities/         Supplied textbook utilities, outside the current target
```

`recognition/ballRecognition.*` are empty placeholders and are not built yet.
When adding an implementation file, list it in `CMakeLists.txt`. Project headers
can be included relative to the project root, e.g. `"geometry/perspectiveTransform.h"`.

## Processing and coordinate conventions

- `detectBalls(image, params)` applies grayscale conversion, Gaussian blur and
  the Hough transform directly to the supplied image.
- `runDetection(image, params)` masks the table, detects circles and applies
  pocket filtering. It returns the masked image, raw circles and filtered circles.
- `detectAndFilterBalls()` returns only the filtered circles from that pipeline.
- `extractBall()` crops the original colour pixels and returns a circular mask.
  It does not load photos, detect balls or save files. `evaluation/extractionDemo.cpp`
  performs those experiment steps.
- Detection circles use original-photo coordinates `(x, y, radius)`.
- Evaluation transforms centres to the 1434 × 683 table image. CSV pairs are
  `(Row, Column)`, loaded as `(x = Column, y = Row)` in those transformed coordinates.
- Blank pairs represent absent balls. Sanity-check labels retain CSV ball IDs;
  the sixteenth column pair is labelled White.
- Matching is one-to-one within the configured tolerance. It maximizes the number
  of valid matches, trying nearer candidates first; it does not globally minimize
  total matching distance.

The refactor preserves the existing experiment distinction: baseline and grid
search use the original unmasked image, while application, extraction and
validation use the masked pipeline. Changing this would change the experiment,
so it should be done separately before retuning parameters.

## Configuration and outputs

Edit `config.h` for default Hough parameters, table geometry, pocket radius,
evaluation tolerance, dataset names and paths. Grid-search candidate ranges and
ranking remain explicit in `evaluation/gridSearch.cpp`.

| Task | Output |
| --- | --- |
| Application | `output/detect_circles/`, `output/detect_circles_filtered/` |
| Baseline | `output/baseline.jpeg` |
| Grid search | `output/grid_search.csv`, `output/best_detection.jpeg` |
| Validation | `output/validation/` |
| Sanity check | `output/sanity_check/` |
| Extraction | `output/recognition/AllBalls/` |

Image output creates missing parent directories and reports save failures.
Colour analysis, K-means, and recognition validation save labelled
`*_comparison.png` grids instead of separate crops and masks for each ball.
Recognition validation puts incorrect predictions in `misclassified/` and
correct predictions in `cca/`, with all diagnostic views in the same image.
`visualisation/combineImages.h` provides the shared grid helper; it accepts
grayscale and colour panels, skips empty images, and preserves pixel dimensions.
CSV runners check file creation and final writing. Existing filenames are
overwritten as before; unrelated or older files are not deleted. In particular,
old numbered extraction crops can remain after a run with fewer candidates.
The printed extraction count is the candidate count.

`data/ground_truth/BallLocations.csv` is the working annotation file, including
previous manual corrections and visually estimated centres. The original backup
is `BallLocation_original.csv` in the same directory. The reorganisation moves
these files without changing their contents.
