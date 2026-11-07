"""Audit report image provenance and list missing application exports; never create images."""
from pathlib import Path
import csv
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / 'output/final-results'

gallery = []
for stem in ['AllBalls'] + [f'Table{i}' for i in range(1, 7)]:
    label = 'training resubstitution' if stem == 'AllBalls' else 'development image'
    gallery.append(r'\begin{figure}[H]\centering' + '\n'
        + rf'\panel{{.48\linewidth}}{{mainFinalResults/{stem}/table_numbered.png}}{{Labelled rectified photograph}}\hfill' + '\n'
        + rf'\panel{{.48\linewidth}}{{mainFinalResults/{stem}/table_diagram.png}}{{Overhead diagram}}' + '\n'
        + rf'\caption{{{stem}: {label}. Both views show the same predicted positions and identities.}}' + '\n'
        + r'\end{figure}' + '\n')
(HERE / 'gallery.tex').write_text('\n'.join(gallery))

text = (HERE / 'main.tex').read_text() + (HERE / 'gallery.tex').read_text()
paths = sorted(set(re.findall(r'\\panel\{[^\n]*?\}\{([^}]+\.png)\}', text)))
assert paths, 'No image references found'
manifest = []
for relative in paths:
    path = RESULTS / relative
    assert path.resolve().is_relative_to(RESULTS.resolve()), relative
    manifest.append({'path': 'output/final-results/' + relative,
                     'exists': path.is_file(),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None})
missing = [r['path'].removeprefix('output/final-results/') for r in manifest if not r['exists']]
(HERE / 'image_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

descriptions = {
 'initial_manual_circles.png': 'Export the actual initial manual Hough configuration; record its parameters. Do not relabel the selected configuration.',
 'predicted_types.png': 'Run the specified type-only method on AllBalls and export its labelled overlay. Training performance must stay separate from Tables1–6.',
 '_original.png': 'Export the original-colour circular candidate crop before selective recognition padding, using the same candidate index as recognition_evidence.csv.',
 '_mask.png': 'Export the crop-aligned circular extraction mask before recognition padding.',
 '_histogram.png': 'Plot/export the measured HSV descriptor with labelled axes or projections; do not substitute a backprojection map.',
 'hard_types.png': 'Export the hard-type histogram identity control with detection fixed, before unique assignment.',
 'soft_types.png': 'Export independent soft-score identities with the same detections, before unique assignment.',
 'unpadded_numbered.png': 'Export soft scores plus unique assignment, without selective padding or cue reservation.',
 'padded_numbered.png': 'Export selective +1 White-crop padding plus assignment, without largest-radius cue reservation.',
}
lines = ['# Missing final-results image exports', '',
 'All paths below are relative to `output/final-results/`. They are proposed destinations, not existing results.',
 'The report never falls back to another output folder. Re-run `python3 "report v3/check_assets.py"` after exporting.', '',
 'Do not overwrite a selected result with a control. Export a configuration identifier, matching metric row, and candidate indices for every comparison.', '']
tex = []
for path in missing:
    description = next((v for k,v in descriptions.items() if path.endswith(k)), 'Export the specified diagnostic from its actual method/configuration.')
    lines.extend([f'- `{path}`', f'  {description}'])
    tex.append(r'\par\noindent\begin{minipage}{\linewidth}\small\path{' + path + r'}\par ' + description.replace('_', r'\_').replace('–','--').replace('→',r'$\rightarrow$') + '\n' + r'\end{minipage}\par\medskip')
(HERE / 'missing_figures.md').write_text('\n'.join(lines) + '\n')
(HERE / 'missing_figures.tex').write_text('\n'.join(tex) + '\n')
# Validate the current selected counts without running or modifying application outputs.
rows = list(csv.DictReader((RESULTS / 'metrics.csv').open()))
dev = [r for r in rows if r['image'] in [f'Table{i}.jpeg' for i in range(1, 7)]]
assert len(dev) == 6
counts = {key: sum(int(r[key]) for r in dev) for key in ['groundTruth','TP','FP','FN','correctNumbers']}
assert counts == {'groundTruth':90,'TP':86,'FP':2,'FN':4,'correctNumbers':80}, counts
print(f'{len(paths)} unique image references: {len(paths)-len(missing)} available; {len(missing)} missing.')
print('All image paths confined to output/final-results. Selected metric totals verified:', counts)
