"""Check the submission's referenced images and selected metrics without changing results."""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'output/final-results'
source = (HERE / 'main.tex').read_text()
# Ignore comments; inspect only figures actually used by this document.
source = re.sub(r'(?m)(?<!\\)%.*$', '', source)
paths = set(re.findall(r'\\(?:smallpanel|panel)\{[^\n]*?\}\{([^}]+\.png)\}', source))
paths.update(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{\\resultsroot\s*([^}]+\.png)\}', source))
assert paths, 'No figure references found'
manifest = []
for relative in sorted(paths):
    path = RESULTS / relative
    assert path.resolve().is_relative_to(RESULTS.resolve()), relative
    assert path.is_file(), f'Missing figure: {relative}'
    manifest.append({'path': 'output/final-results/' + relative, 'exists': True,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(HERE / 'image_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
rows = list(csv.DictReader((RESULTS / 'metrics.csv').open()))
dev = [r for r in rows if r['image'] in [f'Table{i}.jpeg' for i in range(1, 7)]]
assert len(dev) == 6
counts = {key: sum(int(r[key]) for r in dev)
          for key in ['groundTruth', 'TP', 'FP', 'FN', 'correctNumbers']}
assert counts == {'groundTruth': 90, 'TP': 86, 'FP': 2, 'FN': 4, 'correctNumbers': 80}, counts
assert '\\input{gallery.tex}' not in source and '\\input{missing_figures.tex}' not in source
assert 'TODO' not in source
print(f'{len(paths)} referenced images exist; selected metric totals verified: {counts}')
if '--check-pdf' in sys.argv:
    result = subprocess.run(['pdfinfo', str(HERE / 'main.pdf')], check=True,
                            capture_output=True, text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', result.stdout, re.M).group(1))
    assert pages <= 8, f'Report exceeds eight pages: {pages}'
    log = (HERE / 'main.log').read_text()
    assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log, 'Layout overflow'
    print(f'PDF verified: {pages} pages, no overfull boxes.')
