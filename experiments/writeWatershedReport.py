#!/usr/bin/env python3
"""Snapshot the fixed watershed comparison into the LaTeX report."""

import csv
from collections import Counter
from pathlib import Path
import shutil
import sys

from writeRecognitionReport import ROOT, table, escape


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/experiments/watershed_detection"


def read(name):
    with (SOURCE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    names = {"hough": "Hough baseline", "hough_seeded": "Hough-seeded watershed", "hybrid": "Hybrid watershed"}
    pooled = {r["method"]: r for r in read("metrics.csv") if r["image"] == "ALL" and r["split"] == "evaluation"}
    rows = []
    for method, label in names.items():
        r = pooled[method]
        rows.append([label, r["TP"], r["FP"], r["FN"], f"{100 * float(r['precision']):.2f}\\%",
                     f"{100 * float(r['recall']):.2f}\\%", f"{100 * float(r['F1']):.2f}\\%",
                     f"{float(r['meanError']):.2f}"])
    table("watershed_summary.tex", "Current-run detection comparison on Table1--6. The 90 annotations "
          "and 15-table-pixel matching tolerance are identical for all methods. Error is mean matched "
          "centre distance in table pixels.", "tab:watershed-summary", "lrrrrrrr",
          ["Method", "TP", "FP", "FN", "Precision", "Recall", "F1", "Error"], rows)

    missing = [r for r in read("ground_truth_audit.csv") if r["baselineMatched"] == "0"]
    table("watershed_misses.tex", "Baseline missed annotations. Source coordinates are obtained "
          "by inverse perspective transformation for evaluation and visualisation only; they never "
          "supply watershed markers.", "tab:watershed-misses", "lrrrrcc",
          ["Image", "Ball", "Source x", "Source y", "Nearest error", "Raw match", "In ROI"],
          [[escape(r["image"]), r["trueNumber"], f"{float(r['sourceX']):.1f}",
            f"{float(r['sourceY']):.1f}", f"{float(r['nearestBaselineError']):.2f}",
            "Yes" if r["rawHoughMatched"] == "1" else "No",
            "Yes" if r["insideCrowdedROI"] == "1" else "No"] for r in missing])

    outcomes = Counter(r["reason"] for r in read("proposals.csv"))
    table("watershed_peaks.tex", "Distance-peak outcomes across all seven images. Only peaks "
          "passing marker placement proceed to watershed-basin validation.", "tab:watershed-peaks",
          "lr", ["Outcome", "Count"], [[escape(key), count] for key, count in sorted(outcomes.items())])

    rows = []
    metrics = read("metrics.csv")
    for image in sorted({r["image"] for r in metrics if r["image"] != "ALL"}):
        baseline = next(r for r in metrics if r["image"] == image and r["method"] == "hough")
        hybrid = next(r for r in metrics if r["image"] == image and r["method"] == "hybrid")
        rows.append([escape(image), baseline["TP"], baseline["FP"], baseline["FN"],
                     hybrid["TP"], hybrid["FP"], hybrid["FN"], hybrid["numberCorrect"]])
    table("watershed_images.tex", "Per-image baseline and hybrid counts. Recognition is fixed "
          "to the tuned K=6 classifier plus selected solid-only CCA. AllBalls is the reference image.",
          "tab:watershed-images", "lrrrrrrr",
          ["Image", "Base TP", "FP", "FN", "Hybrid TP", "FP", "FN", "Correct no."], rows)
    for source, destination in [("Table1/region_1_diagnostics.png", "watershed_table1.png"),
                                 ("Table3/region_2_diagnostics.png", "watershed_table3.png")]:
        path = SOURCE / source
        if path.exists():
            shutil.copyfile(path, ROOT / "report/assets" / destination)
    print(f"Updated watershed report snapshots from {SOURCE}")


if __name__ == "__main__":
    main()
