#!/usr/bin/env python3
"""Refresh the CCA report tables and example figure from the experiment CSVs."""

import csv
from pathlib import Path
import shutil
import sys

from writeRecognitionReport import ROOT, table, escape


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/experiments/cca_grid_search"


def read(name):
    with (SOURCE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    labels = {"original_k2": "Original K=2", "original_k2_cca": "Original K=2 + CCA",
              "tuned_k6": "Tuned K=6", "tuned_k6_cca": "Tuned K=6 + default CCA",
              "best_number": "Tuned K=6 + selected CCA"}
    totals = {(r["label"], r["split"]): r for r in read("metrics.csv") if r["image"] == "ALL"}
    rows = []
    for key, label in labels.items():
        r, ref = totals[key, "tuning"], totals[key, "reference"]
        rows.append([label, f"{r['typeCorrect']}/{r['matched']}",
                     f"{100 * float(r['typeAccuracy']):.2f}\\%",
                     f"{r['numberCorrect']}/{r['matched']}",
                     f"{100 * float(r['numberAccuracy']):.2f}\\%",
                     f"{ref['numberCorrect']}/{ref['matched']}"])
    table("cca_search_summary.tex", "CCA controls and selected configuration. Tuning scores use "
          "the 86 matched balls in Table1--6; the reference column reports numbers on AllBalls. "
          "The K=6 default-CCA control retains original CCA rule thresholds but aligns its light "
          "definition with configuration 2153.", "tab:cca-search-summary", "lrrrrr",
          ["Configuration", "Type", "Type acc.", "Number", "Number acc.", "Ref. no."], rows)

    rules = {0: "None", 1: "Cue", 2: "Stripe", 3: "Cue + stripe", 4: "Solid",
             5: "Cue + solid", 6: "Stripe + solid", 7: "Cue + stripe + solid"}
    table("cca_search_ablation.tex", "Best observed configuration for each enabled-rule subset "
          "on the K=6 base. Fixes and regressions compare with configuration 2153. "
          "Each subset is tuned separately.", "tab:cca-search-ablation", "lrrrrr",
          ["Rules enabled", "Type", "Number", "Type fixes/losses", "No. fixes/losses", "ID"],
          [[rules[int(r["ruleMask"])], r["typeCorrect"], r["numberCorrect"],
            f"{r['typeFixed']}/{r['typeWorsened']}", f"{r['numberFixed']}/{r['numberWorsened']}",
            r["configId"]] for r in read("ablation.csv")])

    configs = {r["label"]: r for r in read("finalists.csv")}
    default, best = configs["tuned_k6_cca"], configs["best_number"]
    keys = [("minComponentPercent", r"Minimum component area (\%)"), ("boundaryKernel", "Erosion kernel size"),
            ("cueEnabled", "Cue rule enabled"), ("stripeEnabled", "Stripe rule enabled"),
            ("solidEnabled", "Solid rule enabled"), ("solidMinNonLight", r"Solid minimum non-light (\%)"),
            ("solidMaxBoundary", r"Solid maximum light boundary (\%)"),
            ("solidMaxComponents", "Maximum non-light components"),
            ("solidLargestNonLight", r"Minimum largest non-light region (\%)")]
    table("cca_search_parameters.tex", "Original CCA settings and selected CCA settings on the "
          "fixed K=6 base. For enable flags, 1 means enabled and 0 means disabled.",
          "tab:cca-search-parameters", "lrr", ["Parameter", "Default CCA", "Selected CCA"],
          [[label, default[key], best[key]] for key, label in keys])

    rows = [r for r in read("predictions.csv") if r["label"] == "best_number" and
            r["trueNumber"] != "-1" and r["typeChanged"] == "1"]
    table("cca_search_changes.tex", "Matched candidates changed by the selected CCA configuration. "
          "Candidate IDs refer to detections rather than annotated ball numbers.",
          "tab:cca-search-changes", "lrrrrll",
          ["Image", "Candidate", "True no.", "Base no.", "CCA no.", "Base type", "CCA type"],
          [[escape(r["image"]), r["candidate"], r["trueNumber"], r["baseNumber"], r["number"],
            r["baseType"], r["type"]] for r in rows])
    if rows:
        # Prefer a real correction as the report example rather than a regression.
        example = next((r for r in rows if r["numberCorrect"] == "1"), rows[0])
        path = SOURCE / Path(example["image"]).stem / f"ball_{example['candidate']}_comparison.png"
        shutil.copyfile(path, ROOT / "report/assets/cca_search_example.png")
    print(f"Updated CCA report snapshots from {SOURCE}")


if __name__ == "__main__":
    main()
