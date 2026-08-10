# Build instructions

This is the self-contained source package for the unrestricted-length technical
companion. It has not been submitted or published by this workflow.

Compile with either:

```sh
tectonic main.tex
```

or:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The resulting `main.pdf` should match the delivered full PDF. The package
contains no engine binary, model weights, replay bytes, external artwork, or
private data. See `SOURCE_MANIFEST.md` for evidence and claim boundaries.
