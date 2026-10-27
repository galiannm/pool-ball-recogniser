.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

clean:
	latexmk -c -outdir=build main.tex
