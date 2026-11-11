# Detailed implementation and historical experiments

This directory builds independently of the consolidated submission in the parent
directory. It contains the modular demonstrations, validation tools, and historical
experiments; their settings and results can differ from the final submission.

From this directory, configure and build:

```sh
cmake -S . -B build \
  -DOpenCV_DIR=/Users/manon/opencv-4.12.0-install/lib/cmake/opencv4
cmake --build build -j 4
```

Use your own OpenCV installation path on another machine. If the compiler requires
an explicit macOS SDK, also pass `-DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)`.

Run from this directory so `Media/`, `data/`, and `output/` resolve locally:

```sh
./build/PoolBallDetection
./build/BallExperiments --help
```

The modular menu retains choices 2-9 and 0 to exit. Choice 1 belonged to the
consolidated application and is not part of this separate implementation.
See `experiments/README.md` for the historical experiment commands, using this
directory as the working directory and `build/BallExperiments` as the executable.
