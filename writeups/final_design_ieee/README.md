# IEEE long-form manuscript

This directory contains the current, human-readable multi-file LaTeX template
for the long technical paper. It uses `IEEEtran` in conference mode and binds
design claims to the revision-27 derivative contract rather than the older
r175 article draft.

## File map

- `main.tex`: assembly order only.
- `preamble.tex`: IEEE packages and claim-status macros.
- `metadata.tex`: title and anonymous author block.
- `sections/`: one section per source file.
- `figures/`: native LaTeX architecture and evaluation diagrams.
- `tables/`: component-state and evidence tables.
- `references.bib`: IEEE-style BibTeX database.
- `SOURCE_MANIFEST.md`: project and public-source provenance.
- `CLAIMS_AND_PLACEHOLDERS.md`: publication-safe claims and pending evidence.
- `AUTHOR_CHECKLIST.md`: finalization checklist.

## Build

From this directory:

```sh
tectonic main.tex
```

The compiled delivery PDF is copied to
`output/pdf/challengestone_final_design_ieee.pdf`.

The rendered template intentionally labels the author as anonymous and the
manuscript as not submitted. Those fields are ordinary LaTeX in `metadata.tex`.
