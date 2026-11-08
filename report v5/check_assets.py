"""Check current report figures, measured totals, formatting and optional PDF length."""
from pathlib import Path
import csv
import hashlib
import json
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
manifest = []
for relative in paths:
    path = (HERE / relative).resolve()
    assert path.is_relative_to(ROOT), relative
    assert path.is_file(), f'Missing figure: {relative}'
    manifest.append({'path': str(path.relative_to(ROOT)),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(HERE / 'image_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

def row(path):
    with path.open() as f:
        return next(csv.DictReader(f))

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
with (ROOT / 'final_output/experiments/k2_cca/metrics.csv').open() as f:
    cca = next(r for r in csv.DictReader(f) if r['image']=='ALL')
assert (int(cca['baselineCorrect']),int(cca['ccaCorrect']),int(cca['corrected']),int(cca['regressed'])) == (76,73,0,3)
with (ROOT / 'final_output/experiments/guided_recognition/recognition_comparison.csv').open() as f:
    guided = {r['method']: r for r in csv.DictReader(f) if r['image']=='ALL'}
for method, expected in {
    'baseline': (86,2,4,80,1),
    'guided_hough': (89,8,1,81,6),
    'guided_hough_watershed': (88,6,2,80,5),
}.items():
    actual = tuple(int(guided[method][k]) for k in ['TP','FP','FN','correctIdentities','numberedFalsePositives'])
    assert actual == expected, (method,actual)
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
print(f'Verified {len(paths)} figure files and the current final/experimental totals.')
if '--pdf' in sys.argv:
    info = subprocess.check_output(['pdfinfo',str(HERE/'main.pdf')],text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    assert pages <= 8, f'Report exceeds eight-page limit: {pages} pages'
    assert pages == 8, f'Expected planned eight-page layout, got {pages}'
    log = (HERE/'main.log').read_text()
    assert not re.search(r'Overfull \\[hv]box',log), 'Overfull box in report'
    print(f'PDF verified: {pages} A4 pages; 12-point body, 25 mm margins.')
