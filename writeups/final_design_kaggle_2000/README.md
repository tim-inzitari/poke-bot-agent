# Kaggle 2,000-word math-and-design manuscript

This directory contains the current multi-file LaTeX template for the compact
Kaggle-facing article. The narrative is intentionally method-first: it explains
the controller through equations, legal-action architecture, and Alakazam deck
constraints. Runtime operations, training services, revision history, and
submission cadence are outside the main story.

## File map

- main.tex: assembly order only.
- preamble.tex: packages, colors, layout, and evidence macros.
- metadata.tex: title, author, and evidence date.
- sections/: one readable mathematical or design unit per file.
- figures/mathematical_controller.tex: native LaTeX architecture figure.
- tables/math_checks.tex: compact component-level evidence table.
- references.bib: public and internal design references.
- SOURCE_MANIFEST.md: equation and evidence provenance.
- CLAIMS_AND_PLACEHOLDERS.md: safe claim boundaries and update rules.
- WORD_BUDGET.md: editorial budget for the 2,000-word ceiling.

## Build

From this directory:

~~~sh
tectonic main.tex
~~~

The compiled delivery PDF is copied to
output/pdf/challengestone_final_design_kaggle_2000.pdf.

The PDF remains an anonymous, not-submitted manuscript template. Replace the
author line only after publication metadata is settled. Recount the rendered
PDF after any prose, caption, bibliography, or metadata change.
