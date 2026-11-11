# AI Usage Statement

AI tools were used to assist with:

- restructuring repository documentation,
- drafting reproducibility instructions,
- generating template scripts and configuration stubs.

All generated content was reviewed and adjusted by the project contributors before inclusion.

## Assignment 1 report outline - 24 October 2026

- Tool: OpenAI Codex; exact model/version not recorded.
- Used by: requesting group member; identity confirmation TBD.
- Stage: report cover, formatting, outline, and repository preparation.
- Purpose: create a LaTeX cover from a visual reference and supplied logo;
  switch to Latin Modern fonts; organize an English report outline against
  the handbook and the current Assignment 1 implementation.
- Affected files/sections: `assignment1/report/main.tex`, report cover,
  table of contents, all outline headings, and supporting README.
- Representative prompts: create the supplied cover in LaTeX; use the
  traditional LaTeX font; create titles/subtitles from the handbook and
  branch a1 while leaving detailed content as TBD.
- AI contribution: generated and revised LaTeX layout and headings,
  prepared collaboration instructions, and packaged the report source.
- Verification performed by the assistant: read handbook Sections 3--5
  and 8--14, inspected branch a1 at commit 82656a7, used the supplied sample
  only for structure/format, and compiled the current source successfully
  with the desktop editor's compiler. The initial cover was also rendered
  and checked for names and IDs.
- Verification sources: `handbook-ene-v2.pdf`, branch a1 source/configuration
  files and notebook, the user-supplied reference report, and compiler diagnostics.
- Student edits and verification: TBD; no student verification is claimed.
- Responsible member for final verification: TBD.
- Detailed experimental content and results remain TBD.

### Assignment 1 automatic PDF builds - 24 October 2026

- Tool: OpenAI Codex; exact model/version not recorded.
- Used by: requesting group member; identity confirmation TBD.
- Purpose and prompt summary: configure GitHub to build and provide the PDF
  automatically after changes, so group members need not compile locally.
- Affected files: `.github/workflows/a1-report-pdf.yml` and
  `assignment1/report/README.md`.
- AI contribution: XeLaTeX build workflow, artifact upload and download guide.
- Verification: checked the LaTeX action's documented inputs and GitHub artifact
  documentation; resolved action versions to commit hashes. Build execution is
  checked against the corresponding GitHub Actions run.
- Student review and final responsible member: TBD.

## Assignment 1 report draft and table placement - 25 October 2026

- Tool: OpenAI Codex; exact model/version not recorded.
- Used by: requesting group member; identity confirmation TBD.
- Purpose: draft report content from a1 snapshot eb89385269bf1384913a011cf4d8ed6184690172 and stored EDA/notebook evidence, then fix the isolated Table 4 page.
- Affected files: assignment1/report/main.tex, sections/, images/, data/ and README.md.
- AI contribution: drafted data/methodology/preliminary-result discussion, packaged figures and saved curve values, and adjusted only the final batch table placement and local spacing in the last formatting step.
- Evidence limits: final comparative results remain TBD; saved notebook validation scores are labeled preliminary and use a different configuration. No new training or test results were generated.
- Verification: approved source compiled twice with XeLaTeX; 23 pages; no overfull/underfull or unresolved-reference warnings. Table 4 and the following Methodology page were visually checked. Other body pages retained their extracted text apart from page numbering. Published LaTeX and assets were copied byte-for-byte from the approved source.
- User review: user accepted the table placement preview and authorized commit/push. This does not imply final scientific verification of the report.
- Student verification and final responsible member: TBD.

## Assignment 1 completed-run report update - 28 October 2026

- Tool: OpenAI Codex; exact model/version not recorded.
- Used by: requesting report author; member identity and final verifier are not recorded.
- Purpose: finish the report using only committed a1 evidence at
  27ef182308f0f6ac6f04510ee9fc77b9a16cc14a and publish the user-approved source.
- Affected files: assignment1/report/main.tex, sections/, images/, data/ and README.md.
- Contribution: source/artifact inspection, English narrative, model/metric tables,
  error and cost analysis, LaTeX formatting and bullet organization. Reproduction
  notes were condensed into Methodology, diagnostics moved into Results, and AI
  disclosure placed last. Parameter-count formulas were labeled with parameters.
- Figure provenance: the final report uses unchanged image files already present
  in a1; assistant-redrawn charts and cropped panels from intermediate drafts were
  removed. No new training or local-rerun scores enter the comparison.
- Representative requests: complete the report from a1; use only figures already
  in that branch; omit unperformed work; group Results into bullets while retaining
  Conclusion; merge appendices; commit and push to a1-report.
- Verification: reconciled saved comparison, history, checkpoint-epoch and metric
  records; recomputed Accuracy/Macro-F1 from all five stored confusion matrices;
  checked source references and original-image bytes. The saved source compiled
  successfully with the desktop editor compiler before publication. Copied source
  and included assets are hash-checked against the compiled local project.
- Limits: latest exported-PDF visual QA and final member-specific scientific
  verification remain outstanding. No new full test-pass count is asserted.
- Approval: the user explicitly requested commit/push of the edited report.


## Assignment 1 member-confirmed AI disclosure - 28 October 2026

This entry records the member-supplied development account and supersedes the
unresolved report-author identity in earlier entries: Nguyen Huu Minh Khoi
(Nguyễn Hữu Minh Khôi) uses Codex for the report and is responsible for its final
verification. Naming that responsibility does not assert that final student
scientific verification has already been completed.

### ChatGPT (OpenAI, GPT-6)

- Used by and responsible member: Ngô Diễm Quyên.
- Stages: architecture/design, implementation guidance, debugging, sanity testing,
  and evaluation/visualization. Tool/model identification is supplied by the member.
- Representative prompts and contributions: requested five-model configurations,
  hyperparameters and representations; AI suggested layers/dropout and a fair
  validation protocol. Requested module organization and function contracts; AI
  explained pipeline structure. Requested tensor/configuration/checkpoint/JSON/plot
  debugging; AI proposed targeted fixes. Requested forward/shape/parameter/gradient/
  timing/confusion-matrix checks; AI generated example tests and assertions.
  Requested benchmarking, curves and prediction panels; AI recommended procedures
  and layouts.
- Student editing/verification, as reported by the member: independently implemented
  and integrated the main pipeline, reviewed configurations and tests, applied
  relevant fixes, ran scripts/tests, inspected training/validation behavior and
  figures, and compared computed metrics with saved results.
- Affected paths (under assignment1/): src/models/, configs/, src/engine/,
  src/metrics/, src/visualization/, scripts/, test/, and report Results.
- Verification evidence described by the member: sanity checks, local script/test
  results, training/validation behavior, generated figures and saved test metrics.
  This disclosure update does not independently rerun the full test suite or models.

### Gemini Flash (exact version not recorded)

- Used by and responsible member: Ngô Diễm Quyên.
- Stages and prompts: environment documentation (retrieve notebook package versions;
  structure hardware specifications) and notebook debugging (inspect tensor labels
  without the tensor wrapper).
- Contributions: dependency-inspection code, formatting of supplied hardware
  information, and suggestions to use .tolist()/.numpy().
- Student verification, as reported: reviewed package versions, cross-checked system
  information, and executed the conversion snippet and inspected its list output.
- Affected paths: assignment1/requirements.txt, assignment1/README.md,
  assignment1/draft.ipynb. Sources: installed versions, system information and the
  executed notebook snippet.

### OpenAI Codex (exact model/version not recorded)

- Used by and responsible for final report verification: Nguyễn Hữu Minh Khôi.
- Stage: report drafting and revision, 24-28 October 2026.
- Current prompt: summarize/format the supplied ChatGPT and Gemini activities in the
  final report disclosure, retain actual report-authoring assistance, then push.
- Contribution: grouped the disclosure by tool, retained purpose/prompt summaries,
  affected paths and verification/responsibility fields, and corrected tests/ to
  the source branch's actual test/ path. No new experimental result was introduced.
- Affected files: assignment1/report/sections/04-appendices.tex and AI_USAGE.md.
- Assistant checks: compared the supplied account with report/source organization,
  kept all other report sources/images unchanged, checked the saved LaTeX source
  with the desktop compiler before publication, and hash-checked the copied file.
- Sources: the member's supplied disclosure and identity confirmation, handbook
  Sections 3 and 5, and a1 snapshot 27ef182308f0f6ac6f04510ee9fc77b9a16cc14a.
- Authorization: the user explicitly requested the report update and push.
