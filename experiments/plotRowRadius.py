#!/usr/bin/env python3
"""Create report figures from the saved AllBalls-only radius fit (no refitting to tests)."""
import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as effects


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("output/experiments/row_cue_size"))
    parser.add_argument("--output-directory", type=Path, default=Path("report/assets/row_radius"))
    parser.add_argument("--image", type=Path, default=Path("Media/AllBalls.jpeg"))
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with (args.data_directory / "training_samples.csv").open() as stream:
        samples = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if len(samples) < 2:
        raise ValueError("Need at least two saved training samples")
    pairs = []
    for i, first in enumerate(samples):
        for second in samples[i + 1:]:
            dy = second["row"] - first["row"]
            if abs(dy) >= 10:
                pairs.append({"first": int(first["candidate"]), "second": int(second["candidate"]),
                              "deltaRow": dy, "deltaRadius": second["radius"] - first["radius"],
                              "slope": (second["radius"] - first["radius"]) / dy})
    raw_slope = median(p["slope"] for p in pairs) if pairs else 0.0
    slope = max(0.0, raw_slope)
    intercepts = [s["radius"] - slope * s["row"] for s in samples]
    intercept = median(intercepts)
    # CSV stores six decimals; ensure the reconstructed figure agrees with the saved fit.
    for sample in samples:
        if abs(max(1.0, intercept + slope * sample["row"]) - sample["expectedRadius"]) > 1e-4:
            raise ValueError("Reconstructed fit disagrees with saved training predictions")
    with (args.output_directory / "pair_slopes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["first", "second", "deltaRow", "deltaRadius", "slope"])
        writer.writeheader()
        writer.writerows(pairs)
    with (args.output_directory / "intercept_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate", "row", "radius", "interceptCandidate"])
        writer.writerows((int(s["candidate"]), s["row"], s["radius"], a) for s, a in zip(samples, intercepts))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    image = plt.imread(args.image)
    rows = [s["row"] for s in samples]
    radii = [s["radius"] for s in samples]
    lo, hi = min(rows), max(rows)
    expected = lambda y: max(1.0, intercept + slope * y)
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"

    # Relate the measured values to their actual image locations.
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    ax.imshow(image)
    ax.axhspan(lo, hi, facecolor="#56B4E9", alpha=.20)
    for y in (lo, hi):
        ax.axhline(y, color="#56B4E9", lw=1.2, ls="--")
    ax.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0), xlabel="Original-image column (pixels)",
           ylabel="Row (pixels)", title="(a) AllBalls: the training circles occupy a narrow depth band")
    ax = axes[1]
    ax.imshow(image)
    left = min(s["x"] - s["radius"] for s in samples) - 35
    right = max(s["x"] + s["radius"] for s in samples) + 35
    upper = min(s["row"] - s["radius"] for s in samples) - 15
    lower = max(s["row"] + s["radius"] for s in samples) + 18
    for y in (lo, median(rows), hi):
        ax.axhline(y, color="#56B4E9", lw=.9, ls="--", alpha=.9)
        ax.text(left + 3, y - 2, f"row {y:.2f}", fontsize=8, color="white",
                path_effects=[effects.withStroke(linewidth=2.5, foreground="black")])
    for s in samples:
        ax.add_patch(Circle((s["x"], s["row"]), s["radius"], fill=False, ec="#F0E442", lw=1.1))
        ax.plot(s["x"], s["row"], "+", color="white", ms=5, mew=.8)
        ax.text(s["x"], s["row"] - 4, str(int(s["candidate"])), ha="center", va="bottom", fontsize=8,
                color="white", path_effects=[effects.withStroke(linewidth=2, foreground="black")])
    ax.set(xlim=(left, right), ylim=(lower, upper), xlabel="Original-image column (pixels)", ylabel="Row (pixels)",
           title=r"(b) Detection IDs, centre crosses and measured circle radii: each supplies $(y_i,r_i)$")
    fig.tight_layout(h_pad=1.8)
    save(fig, args.output_directory, "row_radius_samples")

    # Show both the data-supported fit and the much wider extrapolation.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.scatter(rows, radii, s=32, color=blue, zorder=3, label="Detected radii")
    ax.plot([lo, hi], [expected(lo), expected(hi)], color=orange, lw=2, label="Robust fitted line")
    ax.set(xlim=(lo-10, hi+10), ylim=(min(radii)-1.2, max(radii)+1.4),
           xlabel="Centre row, $y_i$ (pixels)", ylabel="Detected radius, $r_i$ (pixels)",
           title="(a) Samples and fit within the observed depth band")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=.18)
    ax = axes[1]
    ax.axvspan(lo, hi, color=green, alpha=.12, label="AllBalls row span")
    ax.plot([0, lo], [expected(0), expected(lo)], color=orange, ls="--", lw=1.8)
    ax.plot([lo, hi], [expected(lo), expected(hi)], color=orange, lw=2, label="Fitted radius")
    ax.plot([hi, image.shape[0]-1], [expected(hi), expected(image.shape[0]-1)], color=orange, ls="--", lw=1.8,
            label="Linear extrapolation")
    ax.scatter(rows, radii, color=blue, s=18, zorder=3)
    ax.set(xlim=(0, image.shape[0]-1), ylim=(0, 35), xlabel="Original-image row (pixels)",
           ylabel="Radius (pixels)", title="(b) The same model over the image height")
    ax.text(.04, .93, rf"$r(y)={intercept:.3f}+{slope:.6f}y$", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=.18)
    fig.tight_layout(w_pad=2)
    save(fig, args.output_directory, "row_radius_fit")

    # Visualise the actual robust estimators, rather than just displaying a fitted line.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    ax.hist([p["slope"] for p in pairs], bins=20, color=blue, alpha=.75, edgecolor="white")
    ax.axvline(slope, color=orange, lw=2, label=f"Median b = {slope:.6f}")
    ax.set(xlabel=r"Pairwise slope $(r_j-r_i)/(y_j-y_i)$", ylabel="Number of pairs",
           title=rf"(a) {len(pairs)} eligible pairs: $|y_j-y_i|\geq10$ pixels")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ids = [int(s["candidate"]) for s in samples]
    ax.scatter(ids, intercepts, color=blue, s=28, zorder=3)
    ax.axhline(intercept, color=orange, lw=2, label=f"Median a = {intercept:.3f}")
    ax.set(xticks=ids, xlabel="Detection ID", ylabel=r"Individual intercept $r_i-by_i$ (pixels)",
           title=r"(b) Intercept from the median of $r_i-by_i$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout(w_pad=2)
    save(fig, args.output_directory, "row_radius_estimation")

    (args.output_directory / "captions.md").write_text(
        "# Row/radius figure guide\n\n"
        "Recommended main figure: **row_radius_samples** links the measurements to the source image, "
        "and **row_radius_fit** shows what the model learns and where it extrapolates. "
        "Use **row_radius_estimation** when space permits a detailed illustration of the robust estimator.\n\n"
        "Sample figure: Original AllBalls image and enlarged detected-ball region. Yellow outlines show measured "
        "circles, crosses show centres, and numbers identify detections in training_samples.csv. Horizontal guides "
        "show original-image rows; the shaded overview band spans the observed centre rows.\n\n"
        "Fit figure: Detected radii against centre row, with a robust linear fit. The second panel marks the "
        "observed training row range and distinguishes extrapolation using dashed lines. "
        "The line is estimated from detected circles, rather than measured physical radii.\n\n"
        f"Estimator figure: Distribution of {len(pairs)} pairwise slopes from pairs separated by at least ten rows, "
        f"with median b={slope:.9f}; individual intercept estimates with median a={intercept:.6f}. "
        "Figure reconstruction uses the six-decimal saved sample CSV, so final digits may differ from the C++ model.\n\n"
        "All figures are provided as 300-dpi PNG and vector PDF. No test-image labels or measurements are used. "
        "The fitted row band is narrow; extension to other depths is an assumption. "
        "The underlying figures contain no code or library names.\n"
    )
    print(f"Saved three report figures as PNG/PDF to {args.output_directory}")
    print(f"Reconstructed median fit: a={intercept:.9f}, b={slope:.9f}, eligible pairs={len(pairs)}")


if __name__ == "__main__":
    main()
