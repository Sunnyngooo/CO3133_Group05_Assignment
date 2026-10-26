# AI Usage Statement

AI tools were used to assist with:

- restructuring repository documentation,
- drafting reproducibility instructions,
- generating template scripts and configuration stubs.

The earlier repository scaffolding was recorded as reviewed and adjusted by the
project contributors. New Assignment 3 material has the separate verification
status below.

## Assignment 3 preparation log - 14 October 2026

This entry is a draft disclosure; the contributor must review it and fill in their
name and the person responsible for final verification before submission.

- **Tool:** Codex; exact model identifier not recorded.
- **Used by:** Contributor name to be filled in.
- **Stage/purpose:** Dataset comparison and selection, course-material review,
  experimental planning, and initial N24News metadata preparation.
- **Representative prompts:** "hãy thiết kế plan làm assignment 3" and
  "ok dùng cái này" after discussing N24News.
- **Affected sections/files:** Assignment 3 dataset and methodology planning;
  `assignment3/PLAN.md`, `assignment3/DATASET.md`, `assignment3/PROPOSAL.md`,
  preparation/download scripts and tests, the EDA notebook, pinned data dependencies,
  `configs/assignment3.yaml`, `docs/assignment3.md`, and EDA/audit outputs.
- **AI contribution:** Proposed unimodal/fusion comparisons, prepared the plan,
  dataset notes and proposal, implemented an image-only downloader with resume,
  performed metadata/image audits, created group-aware manifests and EDA charts,
  and visually inspected 120 pilot pairs and 24 original-asset images.
- **Verification performed by the agent:** Read relevant local lectures and
  handbook sections; checked the authors' repository; downloaded metadata from
  revision `047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106`; recorded its SHA-256;
  executed `audit_metadata.py` on all 61,218 metadata records; downloaded and
  decoded all 12,000 selected original-asset images; verified actual manifest
  file hashes, labels, image existence and seven cross-split overlap constraints;
  ran seven preparation tests successfully. Per-image hashes, split hashes,
  source variant and deterministic selection are recorded. The semantic review
  is AI-assisted and limited to the stated samples, not a full human audit.
  No model training/results or instructor approval is claimed. Complete dataset
  usage conditions and human final verification remain pending.
- **Sources:** Local course lectures and `handbook-ene-v2.pdf`;
  https://github.com/billywzh717/N24News and
  https://aclanthology.org/2022.lrec-1.729.pdf.
- **Human edits and final verification:** Pending contributor review; no human
  verification or instructor approval is claimed for these new materials.
- **Responsible member:** To be filled in.

## Assignment 3 scope change: full data

- **User instruction:** "dùng toàn bộ dữ liệu" (use the entire dataset).
- **Change:** Main scope is all 61,218 metadata records, no class cap. Preserve
  the old 12,000-pair outputs as development history; reuse valid local images
  and download missing images into the original-asset directory. Full-data
  manifests, EDA and review materials use a separate `full_seed44_original` run.
- **Implementation:** Updated CLI defaults, configuration, proposal, plan and
  reproduction instructions. Added a full-selection test that checks every ID
  and the original class imbalance are retained. Image feature reading now uses
  four workers with ordered results; a sequential/parallel equivalence test was
  added. All nine tests passed. Downloader concurrency is bounded to 32 workers;
  partial audit logs from resumed invocations are retained locally.
- **Current verification:** All 61,218 images decode, and independent actual-file
  SHA-256 checks passed (6,363,053,805 bytes). No records excluded; manifests contain
  48,976 train / 6,122 validation / 6,120 test records and 58,727 groups. Independent
  metadata/label/path/hash/near-duplicate and seven-field cross-split checks passed;
  reports are in `outputs/assignment3/full_seed44_original/`. No additional semantic review,
  model training or instructor approval is asserted. Human verifier pending.
