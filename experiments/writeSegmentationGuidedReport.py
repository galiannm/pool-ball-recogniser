#!/usr/bin/env python3
"""Snapshot the segmentation-first Hough comparison into the report."""

import csv
from pathlib import Path
import shutil
import sys

from writeRecognitionReport import ROOT, table, escape


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/experiments/segmentation_guided_detection"


def read(name):
    with (SOURCE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    labels = {"hough": "Standard Hough", "segmentation_guided": "Segmentation-guided Hough",
              "segmentation_guided_ws": "Guided Hough + watershed"}
    metrics = read("metrics.csv")
    totals = {(r["method"], r["split"]): r for r in metrics if r["image"] == "ALL"}
    rows = []
    for method, label in labels.items():
        r = totals[method, "evaluation"]
        rows.append([label, r["TP"], r["FP"], r["FN"],
                     f"{100 * float(r['precision']):.2f}\\%", f"{100 * float(r['F1']):.2f}\\%",
                     f"{r['recovered']}/{r['lost']}", r["numberCorrect"]])
    table("segmentation_hough_summary.tex", "Segmentation-first detection comparison on Table1--6. "
          "Recoveries and losses are annotated-ball identities relative to standard Hough; correct-number "
          "counts use the same fixed K=6/CCA recognition pipeline for all methods.",
          "tab:segmentation-hough-summary", "lrrrrrrr",
          ["Method", "TP", "FP", "FN", "Precision", "F1", "Recovered/lost", "Correct no."], rows)

    rows = []
    for method, label in labels.items():
        r = next(r for r in metrics if r["method"] == method and r["image"] == "Table4.jpeg")
        recovered = any(q["method"] == method and q["image"] == "Table4.jpeg" and q["trueNumber"] == "10"
                        and q["change"] == "recovered" for q in read("changes.csv"))
        rows.append([label, r["TP"], r["FP"], r["FN"], "Yes" if recovered else "No", r["numberCorrect"]])
    table("segmentation_hough_table4.tex", "Table4: recovering ball 10 does not imply an improved "
          "overall detection result; other lost matches and false candidates must also be counted.",
          "tab:segmentation-hough-table4", "lrrrcr",
          ["Method", "TP", "FP", "FN", "Ball 10 recovered", "Correct no."], rows)

    rows = []
    for image in sorted({r["image"] for r in metrics if r["image"] != "ALL"}):
        group = {r["method"]: r for r in metrics if r["image"] == image}
        rows.append([escape(image)] + [group[m][key] for m in labels for key in ("TP", "FP")])
    table("segmentation_hough_images.tex", "Per-image true and false detections. AllBalls is the "
          "reference image, reported separately from the six comparison images.",
          "tab:segmentation-hough-images", "lrrrrrr",
          ["Image", "Base TP", "FP", "Guided TP", "FP", "Guided+WS TP", "FP"], rows)

    rows = []
    for r in read("changes.csv"):
        if r["method"] == "segmentation_guided" and r["change"] != "unchanged":
            rows.append([escape(r["image"]), r["trueNumber"], escape(r["change"])])
    table("segmentation_hough_changes.tex", "Annotation identities recovered or lost by the "
          "segmentation-guided Hough variant before the optional watershed check.",
          "tab:segmentation-hough-changes", "lrl", ["Image", "Ball", "Change"], rows)
    source = SOURCE / "Table4/recovered_ball_10_comparison.png"
    if source.exists():
        shutil.copyfile(source, ROOT / "report/assets/segmentation_hough_table4.png")
    print(f"Updated segmentation-guided report snapshots from {SOURCE}")


if __name__ == "__main__":
    main()
