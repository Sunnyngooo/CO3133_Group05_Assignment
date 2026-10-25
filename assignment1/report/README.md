# Assignment 1 report

English LaTeX report for CO3133, Group 05, based on the Assignment 1 code
on branch `a1` (source snapshot `eb89385269bf1384913a011cf4d8ed6184690172`).

The report includes the cover, contents, data analysis, methodology, preliminary
Linear results, references and appendices. Unavailable final results remain `TBD`.
It uses Latin Modern fonts and preserves Vietnamese names.

## Files

- `main.tex`: cover and document assembly.
- `sections/`: data, methodology, results, references and appendices.
- `images/`: logo, EDA figures and preliminary learning curves.
- `data/preliminary_linear_history.csv`: saved notebook values used for the curves.
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
`geometry`, `fontspec`, `graphicx`, `fancyhdr`, `hyperref`, `amsmath`,
`amssymb`, `booktabs`, `tabularx`, `caption`, `placeins`, and `fix-cm` packages.
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

The structure follows the course handbook. Content describes the cited `a1`
source snapshot and stored EDA artifacts. The notebook Linear experiment uses
a different setup and is explicitly preliminary, not an official test result.
Final comparative experiments and other unavailable evidence remain `TBD`.

The reviewed source compiled twice with XeLaTeX into a 23-page preview on
25 October 2026. Table 4 was placed below section 1.5.4 on printed Page 6;
the following page starts Methodology. Those two pages were visually checked.
Other previously identified presentation concerns were outside this targeted fix.
Students must verify and complete the report and AI disclosure before submission.
