"""Build v6-only figures from saved application output and report diagnostics.

No production code, final_output files or report v5 files are written.
"""
from pathlib import Path
import csv
import math
import hashlib
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "assets"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})


def save(figure, name):
    figure.savefig(ASSETS / name, bbox_inches="tight", pad_inches=.04, dpi=240)
    plt.close(figure)


# Refresh detection panels from the latest saved magenta-circle outputs.
detection_directory = ROOT / "final_output/detection/AllBalls"
blur_path = ASSETS / "detection_grey_blur.png"
filtered_path = detection_directory / "filtered_circles.png"
raw_path = detection_directory / "raw_circles.png"
masked_path = detection_directory / "masked.png"
detection_inputs = [masked_path, blur_path, raw_path, filtered_path]
figure, axes = plt.subplots(1, 4, figsize=(7, 1.35),
                            width_ratios=[2.5, 1, 1, 1], layout="constrained")
for index, (axis, path, title) in enumerate(zip(
    axes, [filtered_path, blur_path, raw_path, filtered_path],
    ["Detected circles on masked table", "Smoothed detail",
     "Raw pocket circle", "Pocket rejected"],
)):
    picture = Image.open(path)
    if index == 1:
        picture = picture.crop((1120, 120, 1260, 225))
    elif index >= 2:
        picture = picture.crop((920, 0, 1010, 75))
    axis.imshow(picture, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title, fontsize=8, pad=3)
    axis.axis("off")
# Keep the report's hyphenated name and the older builder name in sync.
for name in ["detection-stages.png", "detection_stages.png", "detection-stages.pdf"]:
    figure.savefig(ASSETS / name, bbox_inches="tight", pad_inches=.04, dpi=240)
save(figure, "detection_stages.pdf")

# Full frames for the separately placed preprocessing and Hough figures.
preprocessing_inputs = [masked_path, blur_path]
figure, axes = plt.subplots(1, 2, figsize=(6.8, 1.05), layout="constrained")
for axis, path, title in zip(axes, preprocessing_inputs,
                            ["Table mask", "Greyscale + Gaussian smoothing"]):
    axis.imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
    axis.set_title(title, fontsize=8, pad=3)
    axis.axis("off")
figure.savefig(ASSETS / "detection_preprocessing.png", bbox_inches="tight", pad_inches=.04, dpi=240)
save(figure, "detection_preprocessing.pdf")

starting_directory = ROOT / "final_output/experiments/past_designs/AllBalls/starting_hough"
starting_metrics_path = starting_directory / "detection_evaluation.csv"
selected_metrics_path = detection_directory / "detection_evaluation.csv"
starting_metrics = next(csv.DictReader(starting_metrics_path.open()))
selected_metrics = next(csv.DictReader(selected_metrics_path.open()))
assert int(starting_metrics["TP"]) + int(starting_metrics["FN"]) == 16
assert [int(selected_metrics[key]) for key in ["TP", "FP", "FN"]] == [16,0,0]
hough_paths = [starting_directory / "raw_circles.png", raw_path, filtered_path]
hough_inputs = [*hough_paths, starting_metrics_path, selected_metrics_path]
figure, axes = plt.subplots(1, 3, figsize=(6.8, 1.1), layout="constrained")
for axis, path, title in zip(axes, hough_paths,
    ["Starting Hough: raw circles", "Tuned Hough: raw circles",
     f"After pocket filtering: {selected_metrics['TP']} balls"]):
    # Display the entire saved image, including all false positives.
    axis.imshow(Image.open(path))
    axis.set_title(title, fontsize=8, pad=3)
    axis.axis("off")
figure.savefig(ASSETS / "hough_comparison.png", bbox_inches="tight", pad_inches=.04, dpi=240)
save(figure, "hough_comparison.pdf")

# Reference construction is a schematic: these bar heights are invented solely
# to explain averaging. Purple is placed between blue (240°) and magenta (300°).
# The comparison below uses only native application exports, with no recomputed
# histograms, cluster means or unexported per-reference similarities.
colour_names = ["yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"]
reference_inputs = []
figure = plt.figure(figsize=(7.1, 2.30), layout="constrained")
outer = figure.add_gridspec(2, 1, height_ratios=[1.1, 1.45], hspace=.15)
training = outer[0].subgridspec(1, 5, width_ratios=[1.5, .16, 1.5, .65, 1.8], wspace=.10)
hue_centres = (np.arange(18)+.5)*20
solid_profile = np.zeros(18)
stripe_profile = np.zeros(18)
solid_profile[10:16] = [.02, .08, .23, .40, .21, .06]
stripe_profile[10:16] = [.04, .14, .28, .32, .17, .05]
schematic_profiles = [solid_profile, stripe_profile, (solid_profile+stripe_profile)/2]
for column, profile, title, colour in [
    (0, schematic_profiles[0], "Solid 4 (illustration)", "#ac93b9"),
    (2, schematic_profiles[1], "Stripe 12 (illustration)", "#8c749b"),
    (4, schematic_profiles[2], "Shared purple reference", "#79518d"),
]:
    axis = figure.add_subplot(training[column])
    axis.bar(hue_centres, profile, width=18, color=colour)
    axis.set_title(title, fontsize=8, pad=3)
    axis.set(xlim=(0,360), ylim=(0,.48), xticks=[0,120,240,360], yticks=[])
    axis.set_xlabel("Hue (degrees)", fontsize=7, labelpad=1)
    axis.tick_params(labelsize=6, length=2, pad=1)
    axis.spines[["top","right"]].set_visible(False)
    if column == 0:
        axis.set_ylabel("Relative frequency\n(schematic)", fontsize=7, labelpad=2)
    if column == 4:
        axis.text(.03,.92,r"$\frac{h_4+h_{12}}{2}$",ha="left",va="top",transform=axis.transAxes,fontsize=9)
axis = figure.add_subplot(training[1])
axis.text(.5,.55,"+",ha="center",va="center",fontsize=13)
axis.axis("off")
axis = figure.add_subplot(training[3])
axis.text(.5,.75,"Average\n50 / 50",ha="center",va="center",fontsize=8)
axis.annotate("",xy=(.98,.33),xytext=(.02,.33),arrowprops={"arrowstyle":"->","color":"#555555"})
axis.axis("off")
construction_axes = list(figure.axes)

comparison = outer[1].subgridspec(2, 3, width_ratios=[1, 1.5, 3.8], hspace=.18, wspace=.30)
for row, (name, candidate, truth, expected_previous) in enumerate([
    ("Table2", 6, 4, "blue"), ("Table1", 12, 10, "purple")
]):
    score_path = ROOT / f"final_output/recognition/{name}/ball_colours.csv"
    baseline_path = ROOT / f"final_output/experiments/past_designs/{name}/identity/hard_hue/balls.csv"
    crop_path = ROOT / f"final_output/detection/{name}/ROI/ball_crops.png"
    reference_inputs.extend([score_path, baseline_path, crop_path])
    scores = {int(r["candidate"]): r for r in csv.DictReader(score_path.open())}
    baseline = {int(r["candidate"]): r for r in csv.DictReader(baseline_path.open())}
    sample = scores[candidate]
    previous_number = int(baseline[candidate]["predictedNumber"])
    assert 1 <= previous_number <= 15
    previous = colour_names[(previous_number-1)%8]
    correct_colour = colour_names[(truth-1)%8]
    assert previous == expected_previous and sample["colour"] == correct_colour
    # Extract the original pixel rectangle from the application's saved ROI grid.
    # Grid cells have 8 px padding and a 24 px heading; images are centred inside.
    montage = Image.open(crop_path)
    cell_width = montage.width // 4
    cell_height = montage.height // math.ceil(len(baseline)/4)
    side = 2*int(np.rint(float(baseline[candidate]["originalRadiusPixels"])))+1
    left = ((candidate-1)%4)*cell_width+8+(cell_width-16-side)//2
    top = ((candidate-1)//4)*cell_height+32+(cell_height-40-side)//2
    axis = figure.add_subplot(comparison[row, 0])
    axis.imshow(montage.crop((left, top, left+side, top+side)), interpolation="nearest")
    axis.set_title(f"{name}: ball {truth}", fontsize=7, pad=2)
    axis.axis("off")
    axis = figure.add_subplot(comparison[row, 1])
    axis.text(.5,.5, f"{previous.title()}\n(wrong)", ha="center", va="center", fontsize=8, color="#9c3e2b")
    if row == 0:
        axis.set_title("K-means + hue rules", fontsize=7, pad=2)
    axis.axis("off")
    axis = figure.add_subplot(comparison[row, 2])
    values = [float(sample["similarity"]), float(sample["similarity"])-float(sample["margin"])]
    axis.barh([1,0], values, height=.65, color=["#79518d" if correct_colour=="purple" else "#4169a1", "#cbd1d6"])
    axis.set(yticks=[1,0], yticklabels=[f"{correct_colour.title()} (correct)", "Next best"], xlim=(0,1), xticks=[0,.5,1])
    axis.tick_params(axis="y", length=0, labelsize=7, pad=2)
    axis.tick_params(axis="x", labelsize=6, length=2, pad=1)
    axis.spines[["top","right","left"]].set_visible(False)
    for y, value in zip([1,0],values):
        axis.text(value+.02,y,f"{value:.2f}",va="center",fontsize=7)
    if row == 0:
        axis.set_title("HSV histogram: saved matching scores", fontsize=7, pad=2)
    else:
        axis.set_xlabel("Similarity (0–1)", fontsize=7, labelpad=1)
figure.savefig(ASSETS / "reference_stages.png", bbox_inches="tight", pad_inches=.04, dpi=240)
# Split the existing drawing without changing either panel's contents.
from matplotlib.transforms import Bbox
figure.canvas.draw()
renderer = figure.canvas.get_renderer()
full_bounds = figure.get_tightbbox(renderer)
for name, axes in [("reference_construction", construction_axes),
                   ("colour_comparison", figure.axes[len(construction_axes):])]:
    bounds = Bbox.union([axis.get_tightbbox(renderer) for axis in axes])
    bounds = bounds.transformed(figure.dpi_scale_trans.inverted())
    crop = Bbox.from_extents(full_bounds.x0-.04, bounds.y0-.04,
                            full_bounds.x1+.04, bounds.y1+.04)
    for extension in ["pdf", "png"]:
        figure.savefig(ASSETS / f"{name}.{extension}", bbox_inches=crop, dpi=240)
save(figure, "reference_stages.pdf")

reference_examples_path = ASSETS / "reference_examples.csv"
reference_examples = {int(r["number"]): r for r in csv.DictReader(reference_examples_path.open())}
type_paths = [ASSETS / f"reference_{number}_{suffix}.png"
              for number in [0, 1, 9]
              for suffix in ["original", "mask", "hls", "kmeans", "light"]]
figure, axes = plt.subplots(3, 5, figsize=(7.1, 2.05),
                            width_ratios=[.85, .85, .85, .85, 2.65], layout="constrained")
light_colour = "#efcf70"
nonlight_colour = "#305673"
light_rgb = matplotlib.colors.to_rgb(light_colour)
nonlight_rgb = matplotlib.colors.to_rgb(nonlight_colour)
for row, (number, name) in enumerate([(0, "Cue"), (1, "Solid 1"), (9, "Stripe 9")]):
    sample = reference_examples[number]
    for col, suffix in enumerate(["original", "hls", "kmeans", "light"]):
        axis = axes[row, col]
        picture = np.asarray(Image.open(ASSETS / f"reference_{number}_{suffix}.png"))
        if suffix == "light":
            mask = np.asarray(Image.open(ASSETS / f"reference_{number}_mask.png")) > 0
            selected = picture > 0
            measured = 100 * np.count_nonzero(selected) / np.count_nonzero(mask)
            assert np.isclose(measured, float(sample["lightPercent"]), atol=.0001)
            picture = np.full((*mask.shape, 3), .94)
            picture[mask] = nonlight_rgb
            picture[selected] = light_rgb
        axis.imshow(picture, interpolation="nearest")
        axis.axis("off")
        if col == 0:
            axis.text(-.15, .5, name, transform=axis.transAxes, ha="right", va="center", fontsize=8)
        if col == 1:
            axis.text(.5, -.08, f"{float(sample['hlsLightPercent']):.1f}% light",
                      transform=axis.transAxes, ha="center", va="top", fontsize=6.5)
        if row == 0:
            axis.set_title(["Original crop", "HLS classes", "K=2 regions", "Light mask"][col], fontsize=8, pad=5)
    axis = axes[row, 4]
    light = float(sample["lightPercent"])
    inferred = "White" if light >= 85 else "Solid" if 100-light >= 70 else "Striped"
    assert inferred == sample["type"]
    axis.barh(.46, light, left=0, height=.29, color=light_colour)
    axis.barh(.46, 100-light, left=light, height=.29, color=nonlight_colour)
    if light > 20:
        axis.text(light/2, .46, "Light", ha="center", va="center", fontsize=7)
    if 100-light > 20:
        axis.text((100+light)/2, .46, "Non-light", color="white", ha="center", va="center", fontsize=7)
    axis.text(.5, .86, f"{light:.1f}% light → {inferred}", transform=axis.transAxes,
              ha="center", va="center", fontsize=8, fontweight="bold")
    rule = "Light ≥ 85%" if inferred == "White" else "Non-light ≥ 70%" if inferred == "Solid" else "Neither threshold reached"
    axis.text(.5, .05, rule, transform=axis.transAxes, ha="center", va="center", fontsize=7, color="#444444")
    axis.set(xlim=(0, 100), ylim=(0, 1), yticks=[], xticks=[])
    axis.spines[:].set_visible(False)
    if row == 0:
        axis.set_title("Coverage → initial type", fontsize=8, pad=5)
figure.savefig(ASSETS / "type_development.png", bbox_inches="tight", pad_inches=.04, dpi=240)
save(figure, "type_development.pdf")

# Regenerate support directly from the actual interval rule for verification.
light = np.linspace(0, 1, 501)
support = {}
figure, axes = plt.subplots(3, 1, figsize=(3.55, 2.0), sharex=True, layout="constrained")
evidence_path = ASSETS / "candidate_evidence.csv"
with evidence_path.open() as source:
    stripe = next(r for r in csv.DictReader(source)
                  if r["image"] == "Table2" and int(r["candidate"]) == 11)
stripe_light = float(stripe["lightPercent"]) / 100
for axis, (low, high, label, colour) in zip(axes, [
    (0, .30, "Solid", "#2475ad"), (.30, .85, "Striped", "#d37b13"),
    (.85, 1, "White", "#29824a"),
]):
    distance = np.maximum.reduce([np.zeros_like(light), low - light, light - high])
    support[label] = np.exp(-.5 * (distance / .10) ** 2)
    axis.axvspan(low, high, color=colour, alpha=.12, linewidth=0)
    axis.plot(light, support[label], color=colour, linewidth=1.6)
    axis.axvline(stripe_light, color="#555555", linestyle=":", linewidth=1)
    point_support = math.exp(-.5 * (max(0, low-stripe_light, stripe_light-high) / .10) ** 2)
    axis.plot(stripe_light, point_support, "o", color=colour, markersize=3)
    axis.set(ylim=(-.04, 1.12), xlim=(0, 1), yticks=[0, 1])
    axis.set_ylabel(label, fontsize=8, rotation=0, labelpad=24, va="center")
    axis.tick_params(labelsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(.98, .55, f"{low:.2f}–{high:.2f}", ha="right", va="center",
              transform=axis.transAxes, fontsize=7, color=colour)
axes[0].set_title("Type support (0–1); shaded = preferred range", fontsize=8, pad=5)
axes[-1].set(xlabel="Fraction of crop counted as light, w", xticks=[0, .30, .85, 1])
axes[-1].xaxis.label.set_size(8)
save(figure, "type_support.pdf")
with (ASSETS / "type_support.csv").open("w", newline="") as output:
    writer = csv.writer(output)
    writer.writerow(["lightFraction", "Solid", "Striped", "White"])
    writer.writerows(zip(light, support["Solid"], support["Striped"], support["White"]))

# Place the current application PNGs directly; do not redraw circles, labels or colours.
def draw_final_table(figure, slot, number, inputs):
    table = f"Table{number}"
    directory = ROOT / f"final_output/final/{table}"
    evaluation_path = directory / "identity_evaluation.csv"
    evaluation = next(csv.DictReader(evaluation_path.open()))
    correct = int(evaluation["correct"])
    truth = int(evaluation["TP"]) + int(evaluation["FN"])
    pair = slot.subgridspec(2, 1, height_ratios=[1, 1], hspace=.025)
    for row, name in enumerate(["table_numbered.png", "table_diagram.png"]):
        path = directory / name
        axes = figure.add_subplot(pair[row])
        axes.imshow(Image.open(path))
        if row == 0:
            detection_path = ROOT / f"final_output/detection/{table}/detection_evaluation.csv"
            detection = next(csv.DictReader(detection_path.open()))
            axes.set_title(
                f"{table}: TP {evaluation['TP']} | FP {evaluation['FP']} | FN {evaluation['FN']}\n"
                f"IDs {correct}/{truth} | mean centre error {float(detection['meanError']):.2f}",
                fontsize=9, pad=3,
            )
            inputs.append(detection_path)
        axes.axis("off")
        inputs.append(path)
    inputs.append(evaluation_path)


figure = plt.figure(figsize=(6.8, 8.3), layout="constrained")
grid = figure.add_gridspec(3, 2, hspace=.08, wspace=.04)
final_inputs = []
for index in range(6):
    draw_final_table(figure, grid[index // 2, index % 2], index+1, final_inputs)
save(figure, "final_views.pdf")

# Separate two-table bands let LaTeX place part of Figure 8 on an earlier page.
# Every band embeds the original numbered photograph and overhead diagram.
final_parts = []
for first in [1, 3, 5]:
    figure = plt.figure(figsize=(6.8, 2.5), layout="constrained")
    grid = figure.add_gridspec(1, 2, wspace=.04)
    inputs = []
    for column, number in enumerate([first, first+1]):
        draw_final_table(figure, grid[0, column], number, inputs)
    name = f"final_views_{first}_{first+1}.pdf"
    save(figure, name)
    final_parts.append((name, inputs))

# Compact counts chart for the report's group-performance discussion.
group_counts_path = ASSETS / "group_counts.csv"
group_counts = {int(r["identity"]): r for r in csv.DictReader(group_counts_path.open())}
assert set(group_counts) == set(range(16))
assert [sum(int(r[key]) for r in group_counts.values())
        for key in ["annotated", "matched", "correct"]] == [90, 86, 80]
assert sum(int(group_counts[n]["correct"]) for n in range(1,9)) == 41
assert sum(int(group_counts[n]["correct"]) for n in range(9,16)) == 33
assert [int(group_counts[10][key]) for key in ["annotated", "matched", "correct"]] == [6,4,2]
figure, axis = plt.subplots(figsize=(6.8, 1.35), layout="constrained")
x = np.arange(16)
axis.axvspan(-.5, .5, color="#f2f2f2", zorder=0)
axis.axvspan(8.5, 15.5, color="#eef4f7", zorder=0)
for offset, key, label, colour in [
    (-.25, "annotated", "Annotated", "#cbd3db"),
    (0, "matched", "Detected", "#87b7d3"),
    (.25, "correct", "Correct identity", "#255775"),
]:
    axis.bar(x+offset, [int(group_counts[n][key]) for n in x],
             width=.24, color=colour, label=label)
axis.set(xticks=x, xticklabels=[str(n) for n in x], yticks=[0,2,4,6],
         xlim=(-.6,15.6), ylim=(0,7.3))
axis.set_xlabel("True ball number", fontsize=8, labelpad=2)
axis.set_ylabel("Number of balls", fontsize=8, labelpad=3)
axis.tick_params(labelsize=7, length=2, pad=2)
for position, label in [(0, "Cue"), (4.5, "Solids (1–8)"), (12, "Stripes (9–15)")]:
    axis.text(position, 6.8, label, ha="center", va="center", fontsize=7)
for boundary in [.5,8.5]:
    axis.axvline(boundary, color="#aeb8bf", linewidth=.6)
axis.grid(axis="y", alpha=.15)
axis.set_axisbelow(True)
axis.spines[["top", "right"]].set_visible(False)
axis.legend(loc="lower center", bbox_to_anchor=(.5,1.03), ncol=3,
            frameon=False, fontsize=7, handlelength=1.4, columnspacing=1.8)
figure.savefig(ASSETS / "identity_counts.png", bbox_inches="tight", pad_inches=.04, dpi=240)
save(figure, "identity_counts.pdf")

# Show the neighbourhood, detected crop and evidence for each failure.
# Context comes from the original photograph and saved circle geometry; the red
# light mask is inferred from the exported K=2 image and checked against its CSV.
from matplotlib.patches import Circle
failure_inputs = []
figure = plt.figure(figsize=(6.8, 2.05))
failure_evidence_path = ASSETS / "candidate_evidence.csv"
failure_evidence = {(r["image"], int(r["candidate"])): r
                    for r in csv.DictReader(failure_evidence_path.open())}
failure_inputs.append(failure_evidence_path)
for column, (table, candidate, description, true_number, predicted_number, diagnostic, outcome) in enumerate([
    ("Table1", 7, "green stripe", 14, -1, "body", "Green stripe → Unknown"),
    ("Table6", 13, "blue stripe", 10, 4, "body", "Blue stripe → Purple solid"),
    ("Table6", 11, "red solid", 3, 11, "kmeans", "Red solid → Red stripe"),
]):
    prefix = f"{table}_candidate_{candidate}"
    photo_path = ROOT / f"Media/{table}.jpeg"
    circles_path = ROOT / f"final_output/final/{table}/balls.csv"
    failure_inputs.extend([photo_path, circles_path])
    circles = {int(r["candidate"]): r for r in csv.DictReader(circles_path.open())}
    circle = circles[candidate]
    assert int(circle["predictedNumber"]) == predicted_number
    assert int(failure_evidence[(table, candidate)]["trueNumber"]) == true_number
    x, y, radius = [round(float(circle[k])) for k in ["originalX", "originalY", "originalRadiusPixels"]]
    photo = Image.open(photo_path)
    context = photo.crop((x-45, y-45, x+46, y+46))
    left = column / 3
    centre = left + 1/6
    figure.text(centre, .97, f"{table}: {description}", ha="center", fontsize=8.5)
    axis = figure.add_axes([left+.055, .51, .223, .40])
    axis.imshow(context)
    axis.add_patch(Circle((45,45), radius, fill=False, edgecolor="#ff00ff", linewidth=1.1))
    axis.axis("off")

    original_path = ASSETS / f"{prefix}_original.png"
    diagnostic_path = ASSETS / f"{prefix}_{diagnostic}.png"
    failure_inputs.extend([original_path, diagnostic_path])
    diagnostic_image = np.asarray(Image.open(diagnostic_path).convert("RGB"))
    evidence_title = "Retained colour"
    evidence_note = ""
    if diagnostic == "kmeans":
        mask_path = ASSETS / f"{prefix}_mask.png"
        type_path = ROOT / f"final_output/recognition/{table}/ball_types.csv"
        failure_inputs.extend([mask_path, type_path])
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        maximum = diagnostic_image.max(axis=2).astype(float)
        minimum = diagnostic_image.min(axis=2).astype(float)
        saturation = np.rint(np.divide((maximum-minimum)*255, maximum,
                            out=np.zeros_like(maximum), where=maximum>0))
        light_mask = mask & (maximum >= 170) & (saturation <= 165)
        measured = 100*np.count_nonzero(light_mask)/np.count_nonzero(mask)
        type_record = next(r for r in csv.DictReader(type_path.open()) if int(r["candidate"])==candidate)
        assert np.isclose(measured, float(type_record["lightPercent"]), atol=.0001)
        diagnostic_image = np.zeros((*mask.shape,3))
        diagnostic_image[mask] = matplotlib.colors.to_rgb("#305673")
        diagnostic_image[light_mask] = matplotlib.colors.to_rgb("#efcf70")
        evidence_title = "Light mask"
        evidence_note = f"Gold = light ({measured:.1f}%)"
    for index, (picture, title) in enumerate([
        (Image.open(original_path), "Detected crop"), (diagnostic_image, evidence_title)
    ]):
        panel_left = left + .035 + index*.155
        axis = figure.add_axes([panel_left, .14, .11, .29])
        axis.imshow(picture, interpolation="nearest")
        figure.text(panel_left+.055, .45, title, ha="center", fontsize=7.5)
        axis.axis("off")
    figure.text(centre, .065, outcome, ha="center", va="center", fontsize=8, color="#873c30")
    if evidence_note:
        figure.text(centre, .005, evidence_note, ha="center", va="center", fontsize=7)
figure.savefig(ASSETS / "failure_details.png", bbox_inches="tight", pad_inches=.04, dpi=240)
save(figure, "failure_details.pdf")

# Measured Table2 competition for purple identities 4 and 12.
evidence_path = ASSETS / "candidate_evidence.csv"
scores_path = ASSETS / "colour_scores.csv"
evidence = {int(r["candidate"]): r for r in csv.DictReader(evidence_path.open())
            if r["image"] == "Table2"}
scores = {int(r["candidate"]): r for r in csv.DictReader(scores_path.open())
          if r["image"] == "Table2"}
example = []
for candidate in [6, 11]:
    light = float(evidence[candidate]["lightPercent"]) / 100
    colour = float(scores[candidate]["purple"])
    solid = math.exp(-.5 * (max(0, light - .30) / .10) ** 2)
    stripe = math.exp(-.5 * (max(0, .30 - light, light - .85) / .10) ** 2)
    example.append({"image": "Table2", "candidate": candidate,
                    "trueNumber": evidence[candidate]["trueNumber"],
                    "lightFraction": light, "purpleSimilarity": colour,
                    "score4": colour * solid, "score12": colour * stripe})
with (ASSETS / "assignment_example.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(example[0]))
    writer.writeheader()
    writer.writerows(example)
assert round(example[0]["score4"] + example[1]["score12"], 3) == 1.234
assert round(example[0]["score12"] + example[1]["score4"], 3) == .767

# The same two measured detections continue the story on the assignment page.
# Use their saved original geometry, with no annotation outlines added to the crops.
table2_path = ROOT / "Media/Table2.jpeg"
saved_circles_path = ROOT / "final_output/final/Table2/balls.csv"
saved_circles = {int(r["candidate"]): r for r in csv.DictReader(saved_circles_path.open())}
picture = Image.open(table2_path)
figure, axes = plt.subplots(2, 2, figsize=(2.9, 2.0), width_ratios=[1, 1.5], layout="constrained")
for row, sample in enumerate(example):
    circle = saved_circles[sample["candidate"]]
    x, y, radius = [round(float(circle[k])) for k in ["originalX", "originalY", "originalRadiusPixels"]]
    crop = picture.crop((x-radius, y-radius, x+radius+1, y+radius+1))
    axes[row, 0].imshow(crop, interpolation="nearest")
    axes[row, 0].axis("off")
    w = sample["lightFraction"]
    solid = sample["score4"] / sample["purpleSimilarity"]
    stripe = sample["score12"] / sample["purpleSimilarity"]
    axes[row, 1].text(0, .5,
        f"True ball {sample['trueNumber']}\nw = {w:.3f}\nSolid: {solid:.3f}\nStriped: {stripe:.3f}",
        va="center", fontsize=8, linespacing=1.5)
    axes[row, 1].axis("off")
axes[0, 0].set_title("Table2 crops", fontsize=8, pad=5)
axes[0, 1].set_title("Measured support", fontsize=8, loc="left", pad=5)
save(figure, "purple_evidence.pdf")

# Rejected trials: compare the same pixels for each detector, rather than redrawing circles.
trial_names = ["development_global_hough.png", "development_guided_hough.png",
               "development_guided_watershed.png"]
figure, axes = plt.subplots(1, 3, figsize=(6.8, 1.1), layout="constrained")
for index, (axis, name, title) in enumerate(zip(axes, trial_names,
        ["Global Hough", "Local Hough", "After watershed"])):
    picture = Image.open(ASSETS / name)
    picture = picture.crop((1421, 77, 1578, 201))
    axis.imshow(picture, cmap="gray", interpolation="nearest")
    axis.set_title(title, fontsize=7, pad=3)
    axis.axis("off")
save(figure, "development_trials.pdf")

cue_decisions_path = ASSETS / "cue_decisions.csv"
cue_decisions = {int(r["candidate"]): r for r in csv.DictReader(cue_decisions_path.open())
                 if r["image"] == "Table3"}
figure = plt.figure(figsize=(6.8, 1.0))
def label(number):
    return "Cue" if int(number) == 0 else "Unknown" if int(number) < 0 else str(number)
for example, candidate in enumerate([2, 4]):
    sample = cue_decisions[candidate]
    assert int(sample["paddingOnly"]) == int(sample["trueNumber"])
    left = example*.51
    truth = "Cue ball" if int(sample["trueNumber"]) == 0 else "Purple stripe (12)"
    figure.text(left+.235, .96, truth, ha="center", fontsize=9, weight="bold")
    for col, suffix in enumerate(["context", "padded"]):
        centre = left + [.11, .36][col]
        axis = figure.add_axes([centre-.09, .19, .18, .56])
        axis.imshow(Image.open(ASSETS / f"Table3_candidate_{candidate}_{suffix}.png"),
                    interpolation="nearest")
        axis.axis("off")
        figure.text(centre, .81, "Detected circle" if col == 0 else "Expanded crop",
                    ha="center", fontsize=8)
        colour = "#a64438" if col == 0 else "#246347"
        figure.text(centre, .09, f"Predicted: {label(sample['unique' if col == 0 else 'paddingOnly'])}",
                    ha="center", va="center", fontsize=8.5, color=colour,
                    bbox={"boxstyle":"round,pad=.22", "facecolor":"#fbefed" if col == 0 else "#edf6f0",
                          "edgecolor":"none"})
    figure.text(left+.235, .43, "→", ha="center", va="center", fontsize=17, color="#657079")
    figure.text(left+.235, .59, "+1 px", ha="center", fontsize=8, color="#657079")
figure.savefig(ASSETS / "cue_refinement.png", bbox_inches="tight", pad_inches=.04, dpi=240)
save(figure, "cue_refinement.pdf")
# Bind each refreshed montage to the exact image bytes used for its construction.
# The report audit can then detect a later application rerun that makes it stale.
def fingerprint(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

provenance = []
for name, inputs in [
    ("detection_stages.pdf", detection_inputs),
    ("detection-stages.pdf", detection_inputs),
    ("detection_preprocessing.pdf", preprocessing_inputs),
    ("hough_comparison.pdf", hough_inputs),
    ("final_views.pdf", final_inputs),
    *final_parts,
    ("identity_counts.pdf", [group_counts_path]),
    ("failure_details.pdf", failure_inputs),
    ("reference_stages.pdf", reference_inputs),
    ("reference_construction.pdf", []),
    ("colour_comparison.pdf", reference_inputs),
    ("type_development.pdf", type_paths + [reference_examples_path]),
    ("assignment_example.csv", [evidence_path, scores_path]),
    ("type_support.pdf", [ASSETS / "type_support.csv", evidence_path]),
    ("purple_evidence.pdf", [table2_path, saved_circles_path, ASSETS / "assignment_example.csv"]),
    ("development_trials.pdf", [ASSETS / name for name in trial_names]
                               + [ASSETS / "development_comparisons.csv"]),
    ("cue_refinement.pdf", [cue_decisions_path]
                           + [ASSETS / f"Table3_candidate_{n}_{suffix}.png"
                              for n in [2, 4] for suffix in ["context", "padded"]]),
]:
    provenance.append({"output": fingerprint(ASSETS / name),
                       "inputs": [fingerprint(path) for path in inputs]})
(ASSETS / "figure_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
print("Refreshed detection panels and all six saved table_numbered/table_diagram pairs.")
