.PHONY: all assets clean
all:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

assets:
	cd .. && MPLCONFIGDIR=/tmp/pool-matplotlib python3 "report v2/generate_assets.py"

clean:
	latexmk -c main.tex
