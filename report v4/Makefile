.PHONY: all audit
all: audit
	pdflatex -interaction=nonstopmode -halt-on-error -recorder main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -recorder main.tex

audit:
	python3 check_assets.py
