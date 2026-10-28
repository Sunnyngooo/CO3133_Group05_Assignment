# Assignment 1 report

English LaTeX report for CO3133, Group 05. Experiment evidence comes from the
committed `a1` snapshot `27ef182308f0f6ac6f04510ee9fc77b9a16cc14a`.

The report covers Fashion-MNIST data analysis, all five architectures, test
results, training/inference costs, learning curves, class-level diagnostics,
error analysis and conclusions. CNN achieves 91.69% test Accuracy and 0.9163
Macro-F1 in the saved single-seed comparison.

## Source and figures

- `main.tex`: cover, contents and document assembly.
- `sections/01-problem-data.tex`: problem, EDA, splitting and preprocessing.
- `sections/02-methodology.tex`: models, training, verification and concise
  reproduction information.
- `sections/03-results.tex`: results and all model diagnostics; Section 3.3.1
  groups quality, cost and interpretation by model.
- `sections/references.tex`: references.
- `sections/04-appendices.tex`: closing AI Usage Disclosure (last in the report).
- `images/`: unchanged original EDA and experiment figures from the cited `a1`
  snapshot, plus the university logo supplied by the group. No assistant-redrawn
  chart or cropped prediction panel is used.
- `data/`: committed comparison values and source-provenance records.
- `../../AI_USAGE.md`: AI-assistance history.

Selected runs are `linear_baseline_001`, `mlp_baseline_002`, `cnn_baseline_002`,
`lstm_baseline_002` and `transformer_baseline_002`. Checkpoints and full run
records remain in the cited `a1` snapshot; they are not duplicated here.

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

## Current verification

The final source compiled successfully with the desktop editor's built-in
LaTeX compiler on 28 October 2026. The package was copied byte-for-byte from
that source, and every included experiment/EDA image was checked against the
cited `a1` snapshot. All source references resolve. The previously reviewed
Conclusion is unchanged by the latest list and section-organization edits.

The latest source has not undergone a fresh exported-PDF visual inspection.
Review the PDF produced by the workflow before submission. User approval to
publish does not imply completed member-specific scientific verification.
