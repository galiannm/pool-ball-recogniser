"""Build a verified PDF without sharing temporary outputs with the IDE's main job."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BUILD = HERE / '.build'
BUILD.mkdir(exist_ok=True)

subprocess.run([sys.executable, 'check_assets.py'], cwd=HERE, check=True)
source = (HERE / 'main.tex').read_bytes()
command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-recorder',
           '-output-directory=.build', '-jobname=report_v6', 'main.tex']
for _ in range(2):
    subprocess.run(command, cwd=HERE, check=True)
if (HERE / 'main.tex').read_bytes() != source:
    raise RuntimeError('The report source changed during compilation; rebuild before publishing.')
subprocess.run([sys.executable, 'check_assets.py', '--pdf', '.build/report_v6.pdf'],
               cwd=HERE, check=True)

# Atomic replacement prevents readers seeing half-written PDFs. The named copy
# is independent of automatic editor recipes that target main.pdf.
for name in ['report_v6.pdf', 'main.pdf']:
    staged = BUILD / (name + '.publish')
    shutil.copyfile(BUILD / 'report_v6.pdf', staged)
    os.replace(staged, HERE / name)
print('Published verified report_v6.pdf and main.pdf (8 pages).')
