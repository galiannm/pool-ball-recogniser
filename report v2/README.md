# Report v2

The report describes the `largest_raw_radius` configuration now consolidated in
`../myApplication.cpp`. Its final CSV/image files reproduce the saved experimental
outputs byte-for-byte. See [SUBMISSION.md](../SUBMISSION.md) for build commands,
source mapping, evaluation basis and comparison modes.

Build with `make` in this folder, or `make -C 'report v2'` from the project root.
The PDF has eight pages, 12-point body text, 25 mm margins and no title page.
Theory/application and results/analysis occupy approximately half each. The
report includes predicted outputs for all six table images, recognition and
cue-size ablations, failure illustrations, limitations, references and an AI
acknowledgement. The original `report/` folder is unchanged.

`provenance.json` records the new executable's measurement sources alongside the
historical experiment sources used for comparison. Scores use the pre-existing
working annotations, which differ from the original supplied CSV; the report
explicitly discloses this and evaluation-informed method selection.

`generate_assets.py` is the existing historical experiment-figure generator.
It also generates intermediate/unused graphics; the current report selects a
subset of those plus the six final PNGs. Offline Python is used for graphics
only; application processing/evaluation is C++/OpenCV.

OpenAI ChatGPT/Codex assisted with implementations, experiments, debugging,
comparisons, figures and report drafting, including this consolidation and
revision. Retain an accurate acknowledgement and review all work before submission.
