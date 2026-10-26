# AI Usage Disclosure

## Assignment 3 preparation - 14 October 2026

- **Tool:** Codex; exact model identifier not recorded.
- **Used by / final verifier:** Contributor names to be filled in before submission.
- **Purpose:** Dataset selection, local course-material review, experimental
  planning, image downloading and verification, group-aware splitting, EDA,
  proposal drafting and visual pair-quality inspection.
- **Representative prompts:** "hãy thiết kế plan làm assignment 3";
  "thực hiện những việc cần làm đi".
- **Affected sections:** Assignment 3 Problem/Data, planned Methodology,
  preparation scripts/tests, dataset proposal and documentation.
- **Verification by the agent:** Metadata provenance and hashes; decode/hash
  checks for 12,000 selected images; manifest/label/file/split checks; seven
  passing tests; visual inspection of 120 metadata-rendered pilot pairs and 24
  original-asset images. These visual assessments are AI-assisted, not an
  independent human quality audit.
- **Sources:** Course handbook/lectures, the authors' N24News repository and
  paper, downloaded metadata and executed local scripts/tests.
- **Human edits and verification:** Pending contributor review; no human review,
  instructor approval or model-training results are asserted for the new work.

The detailed preparation log is maintained in the repository's `AI_USAGE.md`.

## Full-data scope update

At the user's request ("dùng toàn bộ dữ liệu"), the main scope now includes all
61,218 metadata records without a per-class cap. Codex updated the preparation
defaults, documentation and configuration, and ran nine passing tests including
full-selection/class-count and parallel/sequential grouping checks. All 61,218
images decode; independent file SHA-256 and manifest checks passed. The full-data
split is 48,976/6,122/6,120, with 58,727 groups and unchanged global class counts.
Verification reports and a new 120-pair human review form are available under
`full_seed44_original`. No new semantic review or model result is claimed.
Human review and instructor approval remain pending.
