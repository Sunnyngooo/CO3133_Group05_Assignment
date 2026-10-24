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
