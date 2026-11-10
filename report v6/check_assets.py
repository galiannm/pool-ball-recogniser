"""Check v6 final results, design comparisons, measured assignment and PDF length."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
text = (HERE / 'main.tex').read_text()
paths = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text))
paths.update(re.findall(r'\\panel\{[^}]+\}\{([^}]+)\}', text))
paths = sorted(p for p in paths if not p.startswith('#'))
assert paths, 'No report figures found'
# The gallery directly embeds both saved images for each validation table.
paths += [f'../final_output/final/Table{i}/{name}'
          for i in range(1,7) for name in ['table_numbered.png','table_diagram.png','identity_evaluation.csv']]
paths += ['../final_output/final/AllBalls/identity_evaluation.csv',
          '../final_output/detection/AllBalls/detection_evaluation.csv']
paths += ['../final_output/detection/AllBalls/raw_circles.png',
          '../final_output/detection/AllBalls/filtered_circles.png',
          'assets/detection_grey_blur.png']
paths += ['../final_output/experiments/past_designs/recognition_comparison.csv',
          '../final_output/experiments/past_designs/overall/starting_hough/detection_evaluation.csv',
          '../final_output/experiments/past_designs/overall/selected_hough/detection_evaluation.csv']
paths += ['assets/development_comparisons.csv', 'assets/cue_decisions.csv', 'assets/cca_changes.csv',
          '../more_detailed_implementation/output/experiments/watershed_detection/metrics.csv']
paths += ['assets/colour_method_evidence.csv', 'assets/colour_reference_histograms.csv']
manifest = []
for relative in paths:
    path = (HERE / relative).resolve()
    assert path.is_relative_to(ROOT), relative
    assert path.is_file(), f'Missing figure: {relative}'
    manifest.append({'path': str(path.relative_to(ROOT)),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(HERE / 'image_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

# A later application run must not silently leave old montages in the report.
for figure in json.loads((HERE / 'assets/figure_sources.json').read_text()):
    for source in [figure['output'], *figure['inputs']]:
        path = ROOT / source['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], (
            f"Figure input changed: {source['path']}. Run build_figures.py before rebuilding."
        )

def row(path):
    with path.open() as f:
        return next(csv.DictReader(f))

# The separate tuning-image illustration must agree with its saved metrics.
allballs_detection = row(ROOT / 'final_output/detection/AllBalls/detection_evaluation.csv')
allballs_identity = row(ROOT / 'final_output/final/AllBalls/identity_evaluation.csv')
assert [int(allballs_detection[k]) for k in ['TP','FP','FN']] == [16,0,0]
assert int(allballs_identity['correct']) == int(allballs_identity['groundTruth']) == 16

detection = {k: 0 for k in ['TP','FP','FN']}
correct = 0
colour = 0
weighted_error = 0.0
for i in range(1,7):
    image = f'Table{i}'
    d = row(ROOT / f'final_output/detection/{image}/detection_evaluation.csv')
    identity = row(ROOT / f'final_output/final/{image}/identity_evaluation.csv')
    for k in detection:
        detection[k] += int(d[k])
        assert int(d[k]) == int(identity[k]), f'Inconsistent matching: {image}'
    correct += int(identity['correct'])
    colour += int(row(ROOT / f'final_output/recognition/{image}/colour_evaluation.csv')['correct'])
    weighted_error += int(d['TP']) * float(d['meanError'])
assert detection == {'TP':86,'FP':2,'FN':4}, detection
assert correct == 80 and colour == 73, (correct, colour)
assert round(weighted_error / detection['TP'], 2) == 4.91
with (HERE / 'assets/candidate_evidence.csv').open() as f:
    matched = [r for r in csv.DictReader(f) if r['image'].startswith('Table') and int(r['trueNumber'])>=0]
def true_type(number):
    return 'White' if number==0 else 'Solid' if number<=8 else 'Striped'
assert len(matched)==86
assert sum(r['type']==true_type(int(r['trueNumber'])) for r in matched)==76
assert sum(int(r['predictedNumber'])==int(r['trueNumber']) for r in matched)==80

# Retained diagnostics verify body-colour decisions independently of type/assignment.
# The current reference figure uses schematic histograms and native CSV scores;
# its actual dependencies are checked in figure_sources.json above.
with (HERE / 'assets/colour_method_evidence.csv').open() as f:
    colour_comparison = list(csv.DictReader(f))
assert len(colour_comparison)==80
assert sum(r['kmeansColour']==r['trueColour'] for r in colour_comparison)==65
assert sum(r['histogramColour']==r['trueColour'] for r in colour_comparison)==73
assert sum(r['kmeansColour']!=r['trueColour'] and r['histogramColour']==r['trueColour']
           for r in colour_comparison)==8
assert not any(r['kmeansColour']==r['trueColour'] and r['histogramColour']!=r['trueColour']
               for r in colour_comparison)
with (HERE / 'assets/colour_scores.csv').open() as f:
    saved_scores = {(r['image'],r['candidate']):r for r in csv.DictReader(f)}
colour_names = ['yellow','blue','red','purple','orange','green','maroon','black']
for r in colour_comparison:
    saved = saved_scores[(r['image'],r['candidate'])]
    assert r['trueNumber']==saved['trueNumber']
    assert all(math.isclose(float(r[c]),float(saved[c]),abs_tol=1e-6) for c in colour_names)
for image, candidate, truth, previous in [('Table2','6','4','2'), ('Table4','6','12','8')]:
    r = next(r for r in colour_comparison if r['image']==image and r['candidate']==candidate)
    assert (r['trueNumber'],r['kmeansColour'],r['histogramColour'])==(truth,previous,'4')
    assert max(colour_names,key=lambda c:float(r[c]))=='purple'
with (HERE / 'assets/colour_reference_histograms.csv').open() as f:
    colour_references = list(csv.DictReader(f))
assert len(colour_references)==18*8*8
for c in colour_names:
    assert math.isclose(sum(float(r[c]) for r in colour_references),1,abs_tol=1e-6)
assert all(math.isclose(float(r['purple']),(float(r['ball4'])+float(r['ball12']))/2,abs_tol=1e-7)
           for r in colour_references)

with (HERE / 'assets/reference_construction.csv').open() as f:
    references = list(csv.DictReader(f))
assert len(references)==8
assert [int(r['crops']) for r in references]==[2,2,2,2,2,2,2,1]
assert sum(int(r['crops']) for r in references)==15

with (HERE / 'assets/type_support.csv').open() as f:
    support = list(csv.DictReader(f))
for r in support:
    w = float(r['lightFraction'])
    for label, low, high in [('Solid',0,.30), ('Striped',.30,.85), ('White',.85,1)]:
        distance = max(0, low-w, w-high)
        assert math.isclose(float(r[label]), math.exp(-.5*(distance/.10)**2))
assert all(float(r['Striped'])==1 for r in support if .30 <= float(r['lightFraction']) <= .85)

# Verify measured Table2 scores, actual inputs and the saved joint decision.
with (HERE / 'assets/assignment_example.csv').open() as f:
    example = list(csv.DictReader(f))
for r in example:
    w, similarity = float(r['lightFraction']), float(r['purpleSimilarity'])
    solid = similarity * math.exp(-.5*(max(0,w-.30)/.10)**2)
    stripe = similarity * math.exp(-.5*(max(0,.30-w,w-.85)/.10)**2)
    assert math.isclose(float(r['score4']),solid)
    assert math.isclose(float(r['score12']),stripe)
assert [int(r['candidate']) for r in example] == [6,11]
assert [int(r['trueNumber']) for r in example] == [4,12]
assert [round(float(r['score4']),3) for r in example] == [.597,.639]
assert [round(float(r['score12']),3) for r in example] == [.128,.637]
assert round(float(example[0]['score4'])+float(example[1]['score12']),3)==1.234
assert round(float(example[0]['score12'])+float(example[1]['score4']),3)==.767
past = ROOT / 'final_output/experiments/past_designs'
for method, expected in [('soft_local',[4,4]), ('soft_unique',[4,12])]:
    with (past / f'Table2/identity/{method}/balls.csv').open() as f:
        saved = {int(r['candidate']):r for r in csv.DictReader(f)}
    assert [int(saved[int(r['candidate'])]['predictedNumber']) for r in example] == expected

with (past / 'recognition_comparison.csv').open() as f:
    comparison_rows = list(csv.DictReader(f))
comparison = {r['method']:r for r in comparison_rows if r['image']=='TOTAL_TABLE1_6'}
expected = {'type_hls':(43,86,90),'type_k2':(76,86,90),
            'colour_hue':(65,80,84),'colour_histogram':(73,80,84),
            'identity_hard_hue':(68,86,90),'identity_hard_histogram':(72,86,90),
            'identity_soft_local':(73,86,90),'identity_soft_unique':(75,86,90),
            'identity_final':(80,86,90)}
for method, counts in expected.items():
    r = comparison[method]
    assert tuple(int(r[k]) for k in ['correct','matched','annotated']) == counts
    individual = [r for r in comparison_rows if r['method']==method and r['image'].startswith('Table')]
    assert tuple(sum(int(r[k]) for r in individual) for k in ['correct','matched','annotated']) == counts
    assert math.isclose(float(r['accuracy']), counts[0]/counts[1], abs_tol=1e-6)
    assert math.isclose(float(r['endToEnd']), counts[0]/counts[2], abs_tol=1e-6)
start = row(past / 'overall/starting_hough/detection_evaluation.csv')
assert tuple(int(start[k]) for k in ['TP','FP','FN']) == (59,0,31)
selected = row(past / 'overall/selected_hough/detection_evaluation.csv')
assert tuple(int(selected[k]) for k in ['TP','FP','FN']) == (86,2,4)

with (HERE / 'assets/development_comparisons.csv').open() as f:
    development_rows = list(csv.DictReader(f))
development_totals = {r['method']: r for r in development_rows if r['image']=='TOTAL_TABLE1_6'}
for method, expected_counts in {
    'type_k2':(86,2,4,76), 'type_cca':(86,2,4,73),
    'identity_unique':(86,2,4,75), 'identity_padding':(86,2,4,79), 'identity_final':(86,2,4,80),
    'global_hough':(86,2,4,-1), 'guided_hough':(89,8,1,-1), 'guided_watershed':(88,6,2,-1),
}.items():
    r = development_totals[method]
    assert tuple(int(r[k]) for k in ['TP','FP','FN','correct']) == expected_counts
    per_image = [r for r in development_rows if r['method']==method and r['image'].startswith('Table')]
    assert len(per_image)==6
    assert tuple(sum(int(r[k]) for r in per_image) for k in ['TP','FP','FN']) == expected_counts[:3]
    if expected_counts[3]>=0:
        assert sum(int(r['correct']) for r in per_image)==expected_counts[3]
with (HERE / 'assets/cue_decisions.csv').open() as f:
    cue_decisions = [r for r in csv.DictReader(f) if int(r['trueNumber'])>=0]
assert len(cue_decisions)==86
for before, after, expected_gain in [('unique','paddingOnly',4), ('paddingOnly','final',1)]:
    assert sum(r[before]!=r['trueNumber'] and r[after]==r['trueNumber'] for r in cue_decisions)==expected_gain
    assert not any(r[before]==r['trueNumber'] and r[after]!=r['trueNumber'] for r in cue_decisions)
with (HERE / 'assets/cca_changes.csv').open() as f:
    cca_changes = list(csv.DictReader(f))
assert len(cca_changes)==3
assert all(r['baselineType']==true_type(int(r['trueNumber'])) and r['ccaType']=='Striped'
           and r['baselineType']!='Striped' for r in cca_changes)
with (ROOT / 'more_detailed_implementation/output/experiments/watershed_detection/metrics.csv').open() as f:
    watershed_rows = [r for r in csv.DictReader(f) if r['image']=='ALL' and r['split']=='evaluation']
assert {r['method'] for r in watershed_rows}=={'hough','hough_seeded','hybrid'}
assert all(tuple(int(r[k]) for k in ['TP','FP','FN'])==(86,2,4) for r in watershed_rows)

for first, second, expected_change in [('hard_histogram','soft_local',(2,1)),
                                       ('soft_local','soft_unique',(5,3)),
                                       ('soft_unique','final',(5,0))]:
    gained = lost = 0
    for i in range(1,7):
        image = f'Table{i}'
        annotations = [r for r in matched if r['image']==image]
        with (past / f'{image}/identity/{first}/balls.csv').open() as f:
            before = {int(r['candidate']):r for r in csv.DictReader(f)}
        with (past / f'{image}/identity/{second}/balls.csv').open() as f:
            after = {int(r['candidate']):r for r in csv.DictReader(f)}
        for r in annotations:
            candidate, truth = int(r['candidate']), int(r['trueNumber'])
            was_correct = int(before[candidate]['predictedNumber']) == truth
            now_correct = int(after[candidate]['predictedNumber']) == truth
            gained += now_correct and not was_correct
            lost += was_correct and not now_correct
    assert (gained,lost)==expected_change
with (HERE / 'assets/group_counts.csv').open() as f:
    groups = list(csv.DictReader(f))
assert sum(int(r['annotated']) for r in groups)==90
assert sum(int(r['matched']) for r in groups)==86
assert sum(int(r['correct']) for r in groups)==80
assert '\\documentclass[12pt,a4paper]{article}' in text
assert '[margin=25mm]{geometry}' in text
assert 'assets/pipeline.png' in paths
assert not re.search(r'\\(?:fontsize|linespread)|baselinestretch',text)
assert not re.search(r'TODO|missingfigure|\\input\{(?:gallery|missing_figures)',text,re.I)
body='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))
sections = re.split(r'\\section\{Results and analysis\}', body, maxsplit=1, flags=re.I)
assert len(sections) == 2, 'Results and analysis section not found'
processing = sections[0]
assert not re.search(r'cv::|\bOpenCV\b',processing), 'Library references in processing theory'
assert not re.search(r'CLAHE|white balance',body,re.I)
assert '\\bibitem{rbf}' in text and '\\bibitem{hungarian}' in text
assert 'Per-image detection and final identity results' not in text
print(f'Verified {len(paths)} inputs, final totals, recognition steps, rejected trials and measured assignment example.')
if '--pdf' in sys.argv:
    position = sys.argv.index('--pdf') + 1
    pdf = HERE / sys.argv[position] if position < len(sys.argv) else HERE/'main.pdf'
    info = subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    assert pages <= 8, f'Report exceeds eight-page limit: {pages} pages'
    assert pages == 8, f'Expected planned eight-page layout, got {pages}'
    log = pdf.with_suffix('.log').read_text()
    assert not re.search(r'Overfull \\[hv]box',log), 'Overfull box in report'
    print(f'PDF verified: {pages} A4 pages; 12-point body, 25 mm margins.')
