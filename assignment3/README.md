# Assignment 3: N24News preparation

Task: classify the topic of a news article from its image and headline. Dataset
scope is now all 61,218 metadata records, with no per-class cap. The previous
12,000-record subset remains a development artifact, not the main evaluation set.
Full-data preparation is complete: 61,218 valid pairs, split into 48,976 train,
6,122 validation and 6,120 test records. No records were excluded. Model training
has not started. Main experiments
require the Dataset Proposal approval described in the course handbook.

## Environment (Windows PowerShell)

```powershell
python -m venv .venv-a3
.venv-a3/Scripts/python.exe -m pip install -r assignment3/requirements-data.txt
```

The data requirements file pins the versions actually installed for this
preparation run. It does not install PyTorch or a training environment.

## Metadata

Metadata is at `data/assignment3/raw/nytimes_dataset.json`. Its SHA-256 is
`47145500b7161cf90914c2dfc7e4d269931e06f74958d7b89d8a6d40f82b2ede`.
The source revision is `047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106` from the
[authors' repository](https://github.com/billywzh717/N24News).

Raw metadata, images, the Python environment and generated manifests are not
stored in Git. After cloning this branch, retrieve the pinned metadata first:

```powershell
New-Item -ItemType Directory -Force -Path data/assignment3/raw | Out-Null
if (-not (Test-Path data/assignment3/raw/nytimes_dataset.json)) {
    Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/billywzh717/N24News/047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106/nytimes_dataset.json' -OutFile data/assignment3/raw/nytimes_dataset.json
}
Get-FileHash data/assignment3/raw/nytimes_dataset.json -Algorithm SHA256
```

Compare the hash with the value above, then run the image downloader and
preparation commands below. Reports committed under `outputs/assignment3/` are
snapshots from the documented runs; they do not substitute for local data.

```powershell
.venv-a3/Scripts/python.exe assignment3/audit_metadata.py --source-revision 047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106 --retrieved-date 2026-10-14
```

The audit command above reproduces the original retrieval-date record. For a
new metadata retrieval, supply its actual date with `--retrieved-date`.

## Download images (resume supported)

```powershell
.venv-a3/Scripts/python.exe assignment3/download_images.py --per-class 0 --workers 32 --run-name full_seed44_original --image-source original
```

- Selects all metadata records (`--per-class 0`, the default), retaining the
  observed class imbalance. All records in each duplicate group stay together.
- Validates image decoding, dimensions and hashes, skips valid existing files,
  and retries transient errors. A failed replacement preserves an existing file.
- Outputs the exact selection, per-image audit and download summary under
  `data/assignment3/processed/full_seed44_original/`.
- `original` uses the same public image asset path with query-only rendering
  parameters removed. The metadata URLs observed during the pilot produced
  NYT Published-year banners. The original asset route was checked on a 24-class
  pilot; source transformation is recorded explicitly, and failed images are
  logged rather than silently replaced with another image.
- Original-asset images are stored in `data/assignment3/raw/images_original/`.
  Metadata-URL pilot images remain separately in `raw/images/` for provenance.
- `--image-source metadata` reproduces the authors' metadata URLs. Use a
  different run name for different image source variants.
- The user selected full-corpus scope. Instructor approval is still pending.
  Counts of usable pairs and exclusions come from the completed audit, not from
  assuming every metadata URL still works.

This image-only downloader does not crawl article bodies or execute the authors'
`build_fully_dataset.py`. The selected task only uses the existing headlines.

## Prepare splits and EDA (after downloads finish)

```powershell
.venv-a3/Scripts/python.exe assignment3/prepare_data.py --run-name full_seed44_original
```

The preparation stage excludes failed images/empty headlines, groups full-metadata
canonical article/image URL components, exact file/pixel duplicates, and strict
perceptual similarities. It also groups normalized exact headlines as a
conservative control against text memorization (not an event-identity claim).
Image feature reading uses four workers and preserves input order; grouping and
split assignment remain deterministic.
It then assigns entire groups to train/validation/test near 80/10/10 while
preserving per-class proportions as closely as whole groups
allow. It does not undersample classes to equal size. It checks that IDs, groups, image hashes
and canonical URLs do not overlap across splits.
Normalized exact headlines are checked as well.

Outputs:

- `data/assignment3/processed/full_seed44_original/{train,validation,test}.jsonl`
- `data/assignment3/processed/full_seed44_original/label_mapping.json`
- `outputs/assignment3/full_seed44_original/eda_summary.json`
- Class/input/split plots and a pair-review gallery in the same output directory.
- `outputs/assignment3/pilot_seed44/pair_review.csv`: AI-assisted visual review of
  120 pilot pairs (5 per class), with reviewer and limitations recorded. Human
  verification remains pending. This pilot predates the full-data preparation;
  the reviewed pilot used metadata rendering rather than the original-asset variant.
- [Full-data EDA notebook](notebooks/eda_full.ipynb): interactive inspection of these generated
  outputs. The standalone scripts are sufficient; using the notebook requires
  an IPython/Jupyter kernel in the selected environment.

The full-data group-aware split and current image assets form a custom protocol,
not a reproduction of the paper's original experiment. Do not combine the old
subset train/test manifests with the new full-data manifests: group membership
can change when more records are included. Generic portraits, overlapping
editorial categories and unobserved event-level duplicates remain limitations.
No model performance is claimed by the preparation artifacts.

## Verification

```powershell
.venv-a3/Scripts/python.exe -m unittest discover -s assignment3/tests -v
```

Checks cover download resume, preservation on failure, verification of a valid
download, full selection without changing class counts, transitive duplicate
grouping, parallel/sequential image-reading equivalence, and deterministic
group-aware splitting. All nine tests passed. Independent full-data verification
is recorded in `outputs/assignment3/full_seed44_original/download_verification.json`
and `preparation_verification.json` in the same directory.

## Next milestone

Review [PROPOSAL.md](PROPOSAL.md), the EDA and dataset usage conditions with the
instructor. Record Approved/Approved with conditions and any conditions before
starting main training. See [PLAN.md](PLAN.md) and [DATASET.md](DATASET.md).
