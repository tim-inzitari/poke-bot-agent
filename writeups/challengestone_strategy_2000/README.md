# Build instructions

This is a self-contained LaTeX source package for the under-2,000-word local
Strategy-track draft. It has not been submitted.

Compile with either:

```sh
tectonic main.tex
```

or:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The resulting `main.pdf` should match the delivered PDF. `WORD_COUNT.txt`
records the counting method and safety margin. `SOURCE_MANIFEST.md` identifies
the public and repository evidence used; no competition engine, replay bytes,
model weights, card artwork, or private data are included.
