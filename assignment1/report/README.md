# Assignment 1 report

English LaTeX report for CO3133, Group 05, based on the Assignment 1 code
on branch `a1` (commit `82656a7`).

The report currently includes the cover page, a table of contents, three main
parts, references, and appendices. Detailed content is intentionally `TBD`.
It uses Latin Modern fonts and preserves Vietnamese names.

## Files

- `main.tex`: report source, including the cover and outline.
- `images/01_logobachkhoasang.png`: university logo supplied by the group.
- `../../AI_USAGE.md`: AI assistance and pending student verification record.

## Compile locally

Install a LaTeX distribution with XeLaTeX, Latin Modern fonts, and the
`geometry`, `fontspec`, `graphicx`, `fancyhdr`, and `hyperref` packages.
From this directory, run twice to resolve the table of contents:

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `main.pdf`. LuaLaTeX is also compatible with the source.
Generated PDFs and compiler files are excluded from version control.

## Work together

Use branch `a1-report` as the shared report branch. Create a personal branch
from it, edit your assigned sections, and submit changes back to `a1-report`.
Pull the latest changes before starting; keep edits focused on your own sections
to reduce merge conflicts. Keep `main.tex` and `images/` together when exporting
or importing the project into another LaTeX editor.

## Current scope and verification

The source outline follows Sections 3--5 and 8--14 of the course handbook
`handbook-ene-v2.pdf`. The sample report informed formatting only.
The source was compiled successfully with the desktop editor's compiler.
No results or completed experiments are claimed by the `TBD` placeholders.
Students must verify and complete the report and AI disclosure before submission.
