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

## Download the PDF without installing LaTeX

GitHub automatically builds the report when report files change on `a1-report`
and checks report changes in pull requests targeting that branch.

1. Sign in to GitHub and open [Build Assignment 1 PDF](https://github.com/Sunnyngooo/CO3133_Group05_Assignment/actions/workflows/a1-report-pdf.yml?query=branch%3Aa1-report).
2. Open the latest successful run for `a1-report` (green check mark).
3. In the run summary, follow **Download the report PDF**, or find **Artifacts**
   and click **Group05_A1_Report**.
4. Extract the downloaded ZIP and open **Group05_A1_Report.pdf**.

Artifacts are kept for 30 days. Check the run's commit to ensure you have the
version you need. If a build fails, inspect its log; the previous successful
artifact remains available until it expires.

The build uses XeLaTeX and automatically resolves the table of contents.
PDFs are provided as downloadable build artifacts rather than committed files;
`git pull` retrieves the report source, not the generated PDF.
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
