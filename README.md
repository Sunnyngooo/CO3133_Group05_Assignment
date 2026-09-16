# CO3133 Group 5

Repository structure and reproducibility details for all three assignments.

## Repository layout

```text
assignment_1/      # Assignment 1 code
assignment_2/      # Assignment 2 code
assignment_3/      # Assignment 3 code
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
python assignment_3/train.py --config configs/assignment3.yaml --prepare-only
```

## Train

```bash
python assignment_1/train.py --config configs/assignment1.yaml
python assignment_2/train.py --config configs/assignment2.yaml
python assignment_3/train.py --config configs/assignment3.yaml
```

## Evaluate

```bash
python assignment_1/evaluate.py --config configs/assignment1.yaml
python assignment_2/evaluate.py --config configs/assignment2.yaml
python assignment_3/evaluate.py --config configs/assignment3.yaml
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
