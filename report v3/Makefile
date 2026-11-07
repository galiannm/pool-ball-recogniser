.PHONY: all audit
all: audit
	pdflatex -interaction=nonstopmode -halt-on-error -recorder main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -recorder main.tex
	python3 check_assets.py --check-pdf

audit:
	python3 check_assets.py
