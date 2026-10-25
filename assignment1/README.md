# Assignment 1: Fashion-MNIST Classification

This project trains and evaluates five image classifiers on Fashion-MNIST: Linear, MLP, CNN, LSTM, and Transformer.

## Project structure

```text
assignment1/
├── configs/
│   ├── base.yaml
│   ├── linear.yaml
│   ├── mlp.yaml
│   ├── cnn.yaml
│   ├── lstm.yaml
│   └── transformer.yaml
├── data/                         # Dataset cache and generated train/validation split
├── outputs/eda/                  # Generated dataset statistics and plots
├── runs/<run_id>/                # Resolved config, training history, and test metrics
├── checkpoints/<run_id>/         # Best validation checkpoint
├── scripts/
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   └── compare.py                # Placeholder; not required to train or evaluate
├── src/
│   ├── data/                     # Dataset, split, statistics, and transforms
│   │ ├── dataset.py
│   │ ├── split.py
│   │ ├── statistics.py
│   │ └── transforms.py
│   ├── engine/                   # Training, evaluation, and checkpoint handling
│   │ ├── trainer.py
│   │ ├── evaluator.py
│   │ └── checkpoint.py
│   ├── metrics/                  # Classification metrics and model statistics
│   | ├── model_stats.py
│   │ └── cls_metrics.py
│   ├── models/                   # Linear, MLP, CNN, LSTM, Transformer, and factory
│   │ ├── linear.py
│   │ ├── mlp.py
│   │ ├── cnn.py <!--blank file-->
│   │ ├── rnn.py <!--blank file-->
│   │ ├── transformer.py <!--blank file-->
│   │ └── factory.py
│   ├── utils/                    # Config, run IDs, and random seeds
│   │ ├── config.py
│   │ ├── logger.py <!--blank file-->
│   │ ├── timing.py <!--blank file-->
│   │ └── seed.py
│   └── visualization/
├── test/                         # Assignment 1 tests for debugging
├── draft.ipynb                   # draft of assignment 1
├── requirements.txt
└── README.md
```

### Hardware configuration

- **Platform:** Windows 64-bit, x64-based architecture
- **Processor (CPU):** Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz (Max 2.11 GHz, 4 cores, 8 threads)
- **System Memory (RAM):** 16.0 GB (15.7 GB usable)
- **GPU Accelerator:** None / Integrated Intel(R) UHD Graphics 620
- **Execution Target:** CPU execution using PyTorch CPU build (`2.14.0+cpu`)

## Environment setup

The project was tested with Python 3.12.10 and the CPU-only PyTorch build. From the repository root, create and activate a virtual environment, then install Assignment 1's pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r assignment1\requirements.txt
```

Run the commands below from the **repository root** (the directory containing `assignment1`). The scripts use the CPU-only PyTorch dependency in `requirements.txt`.

## Run the experiments

The first run downloads Fashion-MNIST automatically. The project uses seed `42`, a deterministic train/validation split, and per-model YAML configuration merged with `configs/base.yaml`.

Run exploratory data analysis (optional):

```powershell
python -m assignment1.scripts.eda
```

Train one model:

```powershell
python -m assignment1.scripts.train --model model_name
```

Available model names are `linear`, `mlp`, `cnn`, `lstm`, and `transformer`. To train all five:

```powershell
python -m assignment1.scripts.train --model linear
python -m assignment1.scripts.train --model mlp
python -m assignment1.scripts.train --model cnn
python -m assignment1.scripts.train --model lstm
python -m assignment1.scripts.train --model transformer
```

Training saves the resolved configuration and history in `assignment1/runs/<run_id>/`, and the best validation checkpoint in `assignment1/checkpoints/<run_id>/`. Run IDs are generated as `<model>_baseline_001`, then increment for each later run of that model. On a fresh checkout, the first MLP run is therefore `mlp_baseline_001`; use the actual folder name under `assignment1/runs/` when evaluating.

Evaluate a trained run using its run ID:

```powershell
python -m assignment1.scripts.evaluate --run-id mlp_baseline_001
```

Evaluation writes `test_metrics.json` into that run's directory. Replace the example ID with the ID of the run you want to evaluate. To reproduce the existing run IDs and metrics, keep the same Python/dependency versions, use the CPU environment, and do not remove or modify the saved split, run configuration, or checkpoint.

## Tests

Run the Assignment 1 test suite from the repository root:

```powershell
python -m pytest assignment1\test
```
