# CO3133 G-05

Repository structure and reproducibility details for all three assignments.

## Assignment 3 current status

Assignment 3 uses N24News image + headline classification. Data preparation code
is in `assignment3/`; model training and evaluation are not implemented yet.
Use its separate environment and preparation instructions linked below.

Main data scope is all 61,218 N24News metadata records (no per-class cap), retaining
every usable image/headline pair. Full-data preparation is complete under
`full_seed44_original`: 61,218 valid pairs, split 48,976/6,122/6,120. The earlier
12,000-pair subset is retained for development;
its manifests must not be mixed with the new full-data split.

- [Preparation commands](assignment3/README.md)
- [Dataset proposal](assignment3/PROPOSAL.md)
- [Plan and requirement checklist](assignment3/PLAN.md)
- [Dataset provenance](assignment3/DATASET.md)

The preparation environment is `.venv-a3` on Windows. Main training starts after
the Dataset Proposal is approved; no Assignment 3 model results are claimed.

## Repository layout

```text
assignment_1/      # Assignment 1 code
assignment_2/      # Assignment 2 code
assignment3/       # Assignment 3 preparation code, proposal and plan
configs/           # Assignment configs (includes seeds)
requirements.txt   # Pinned dependency versions
AI_USAGE.md        # AI usage declaration
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation

Set up the dataset path for each assignment in the corresponding config file:

- `configs/assignment1.yaml`
- `configs/assignment2.yaml`
- `configs/assignment3.yaml`

Then prepare data:

```bash
python assignment_1/train.py --config configs/assignment1.yaml --prepare-only
python assignment_2/train.py --config configs/assignment2.yaml --prepare-only
.venv-a3/Scripts/python.exe assignment3/prepare_data.py --run-name full_seed44_original
```

## Train

```bash
python assignment_1/train.py --config configs/assignment1.yaml
python assignment_2/train.py --config configs/assignment2.yaml
# Assignment 3 model training is not implemented yet.
```

## Evaluate

```bash
python assignment_1/evaluate.py --config configs/assignment1.yaml
python assignment_2/evaluate.py --config configs/assignment2.yaml
# Assignment 3 model evaluation is not implemented yet.
```

## Reproducibility details

- **Seed settings**: Stored in each config file (`seed` field).
- **Dependency versions**: Pinned in `requirements.txt`.
- **Hardware information**: Experiments were run on a Linux machine with CPU-only execution (minimum 8 GB RAM recommended). If GPU is available, set device in config as needed.
- **Checkpoints**: Use `checkpoint_dir` from each config. If a checkpoint is missing, rerun the train command for that assignment to reconstruct it.

## Links

- Report: [Report placeholder](./report/README.md)
- Assignment page: [Assignment brief](./ASSIGNMENT_PAGE.md)

## Additional documentation

- [AI usage statement](./AI_USAGE.md)
