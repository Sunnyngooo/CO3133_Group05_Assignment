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
├── data/                         # downloaded dataset and generated split (local)
├── outputs/
│   └── eda/                      # statistics, class distribution, sample images
├── runs/
│   └── <run_id>/                 # config, history, test/inference metrics, error analysis
├── checkpoints/
│   └── <run_id>/                 # best.pt selected by validation macro-F1
├── reports/
│   ├── comparison.csv
│   └── figures/                  # learning curves, confusion matrices, predictions
├── scripts/                      # eda.py, train.py, evaluate.py, benchmark.py,
│                                  # analyze_error.py, compare.py
├── src/
│   ├── data/                     # dataset.py, split.py, statistics.py, transforms.py
│   ├── engine/                   # trainer.py, evaluator.py, checkpoint.py, predictions.py
│   ├── metrics/                  # classification, model stats, errors, inference timing
│   ├── models/                   # linear.py, mlp.py, cnn.py, lstm.py, transformer.py,
│   │                             # factory.py
│   ├── utils/                    # config.py, run_id.py, seed.py
│   └── visualization/            # learning_curve.py, confusion_plots.py,
│                                  # prediction_sample.py
├── test/
├── draft.ipynb                   # development notebook
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

Run exploratory data analysis:

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

Training saves the resolved configuration and history in `assignment1/runs/<run_id>/`, and the best validation checkpoint in `assignment1/checkpoints/<run_id>/`. Run IDs are generated as `<model_name>_baseline_001`, then increment for each later run of that model. On a fresh checkout, the first MLP run is therefore `mlp_baseline_001`; use the actual folder name under `assignment1/runs/` when evaluating.

Evaluate a trained run using its run ID:

```powershell
python -m assignment1.scripts.evaluate --run-id mlp_baseline_001
```

Evaluation writes `test_metrics.json` into that run's directory. Replace the example ID with the ID of the run you want to evaluate. Run IDs identify experiments, not for reproducibility. Therefore, in order to reproduce a result, use the same code, configuration, data split, seed and dependencies. however, small numerical differences may occur due to differences in hardward and software environments.

## Tests

Run the Assignment 1 test suite from the repository root:

```powershell
python -m pytest assignment1\test
```

Or run `python -m pytest -q` from the repository root; `pytest.ini` selects the Assignment 1 suite. All tests run on CPU with small synthetic inputs. Dataset/EDA tests replace the downloader with an in-memory dataset, exercise the real splitting, transforms and loaders, and write generated files only under pytest's temporary directory. They do not download Fashion-MNIST or overwrite saved experiment artifacts. Passing these tests verifies pipeline behavior, not the reported benchmark accuracy; use the evaluation commands above to check saved model results.

## Dataset, split and preprocessing

| Item              | Main comparison protocol                                                                                                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task              | Single-label classification: one grayscale garment image to one of 10 class logits; prediction is argmax.                                                                        |
| Dataset           | Fashion-MNIST, loaded by `torchvision.datasets.FashionMNIST`; 70,000 images, 28 x 28 pixels. No extension dataset is included in these results.                                  |
| Training          | 51,000 images: 85% of the official 60,000-image training set.                                                                                                                    |
| Validation        | 9,000 images: 15% of the official training set; used to select checkpoints.                                                                                                      |
| Test              | Official 10,000-image test set; 1,000 images per class; excluded from training and checkpoint selection.                                                                         |
| Split policy      | Image-level `torch.utils.data.random_split`, seed 42, without stratification; same split configuration for all five models.                                                      |
| Split persistence | `data/splits/fashion_mnist_split_seed42_val0p15.npz` under `assignment1`; generated on first use and reused. Loader checks coverage, overlap, duplicates, sizes and index range. |
| Leakage control   | Train/validation indices are disjoint; normalization statistics use training indices only. This does not establish that visually duplicate items have been audited.              |

Training class counts (class IDs 0-9):

- T-shirt/top 5,030
- Trouser 5,073
- Pullover 5,110
- Dress 5,142
- Coat 5,131
- Sandal 5,108
- Shirt 5,058
- Sneaker 5,168
- Bag 5,115
- Ankle boot 5,065.

The largest/smallest count ratio is 1.0274: the training subset is nearly balanced, so no class weighting is used. These counts come from the saved EDA statistics, not an assumption of exact stratification.

| Preprocessing stage   | Applied behavior                                                                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Conversion            | Convert to image tensor and float32, scaling raw intensities to [0, 1].                                                                        |
| Normalization         | `(x - 0.2856669724) / 0.3528567851`; statistics estimated on the 51,000 training images and shared with validation/test.                       |
| Resize / augmentation | No resize; augmentation disabled for the reported runs. Optional random affine rotation exists in code but was not evaluated here.             |
| Batch                 | Images `[B, 1, 28, 28]`, integer class labels `[B]`; B=128 except the final partial batch. Only the training loader shuffles.                  |
| Model representation  | Flattened 784 pixels for Linear/MLP; 2D feature maps for CNN; 28 rows of 28 values for LSTM; 16 non-overlapping 7 x 7 patches for Transformer. |

## Training setup

| Setting                    | Reported runs                                                                                                                                                                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Framework / versions       | PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu, Python 3.12.10; other pinned dependencies in `assignment1/requirements.txt`.                                                                                                                  |
| Hardware                   | Windows x64; Intel Core i7-8665U (4 cores / 8 threads), 16 GB RAM; CPU execution.                                                                                                                                                         |
| Optimization               | Adam, learning rate 0.0005, weight decay 0; batch size 128, 20 epochs for every model.                                                                                                                                                    |
| Loss / prediction          | CrossEntropyLoss on raw logits (no preceding softmax); argmax for predicted class.                                                                                                                                                        |
| Regularization             | Dropout 0.2 in MLP hidden layers, CNN classifier, LSTM inter-layer/output stages and Transformer encoder; none in Linear.                                                                                                                 |
| Scheduler / early stopping | Neither used; all runs complete 20 epochs.                                                                                                                                                                                                |
| Checkpoint                 | Highest validation macro-F1; strict improvement replaces the checkpoint, so an exact tie retains the earlier epoch.                                                                                                                       |
| Randomness / loader        | Seed 42, deterministic mode enabled; num_workers=0.                                                                                                                                                                                       |
| Precision                  | Default float32; no mixed precision.                                                                                                                                                                                                      |
| Timing                     | Training wall time spans the fit loop, including validation and checkpoint writes; excludes dataset setup. Inference is CPU forward-only, batch 1, using 4 CPU threads, after 5 warmups over 20 timed repeats.                            |
| Experiment design          | Architecture and representation vary; split, preprocessing, optimizer, learning rate, batch size, epoch budget and metrics remain fixed. Compare macro-F1 and accuracy alongside cost. No systematic hyperparameter search is documented. |

## Architectures

| Model       | Implemented architecture and representation                                                                                                                                                                                                                                                                                           | Parameters |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| Linear      | Flatten 1 x 28 x 28 to 784; Linear(784, 10). A linear decision rule provides a low-cost reference.                                                                                                                                                                                                                                    |      7,850 |
| MLP         | 784 -> 256 -> 128 -> 10; ReLU and dropout 0.2 after each hidden layer. Nonlinear global pixel interactions without explicit spatial weight sharing.                                                                                                                                                                                   |    235,146 |
| CNN         | Two Conv3x3(padding=1)-ReLU-MaxPool2 blocks, channels 1 -> 32 -> 64; feature maps 32 x 14 x 14 then 64 x 7 x 7; flatten 3,136 -> 64 -> 10, with ReLU/dropout 0.2 in the head.                                                                                                                                                         |    220,234 |
| LSTM        | `[B, 28, 28]`: each timestep is one image row; two unidirectional LSTM layers, hidden size 128, dropout 0.2; last layer's final hidden state -> dropout -> Linear(128, 10).                                                                                                                                                           |    214,282 |
| Transformer | Conv7x7(stride=7) projects patches to `[B, 16, 128]`; learned positional embeddings; two pre-norm encoder layers, 4 heads, FFN width 256, GELU and dropout 0.2; mean token pooling -> LayerNorm -> Linear(128, 10). Self-attention queries/keys/values come from the token sequence, preserving its shape through each encoder layer. |    274,954 |

The project defines these architectures and its training loop; layers, automatic differentiation, optimizers, datasets and transforms come from PyTorch/torchvision. All five main models are trained from scratch without pretrained backbones.

## Reported results

| Model       | Test accuracy | Test macro-F1 | Parameters | Training (s) | Median inference (ms/image) |
| ----------- | ------------: | ------------: | ---------: | -----------: | --------------------------: |
| cnn         |        91.69% |        0.9163 |    220,234 |      1748.41 |                      0.8216 |
| lstm        |        89.66% |        0.8960 |    214,282 |      1798.18 |                      1.9652 |
| mlp         |        89.08% |        0.8907 |    235,146 |       715.37 |                      0.2037 |
| transformer |        87.98% |        0.8790 |    274,954 |      2133.30 |                      1.9598 |
| linear      |        84.31% |        0.8424 |      7,850 |       941.05 |                      0.0484 |

Source: [comparison.csv](reports/comparison.csv). These are single-seed results, not means across independent runs. Median inference measures one preprocessed image with a loaded model; it excludes loading, preprocessing and model startup.

| Model / selected run                     | Curves                                                                             | Confusion matrices                                                                                                | Predictions                                                                | Run evidence                                                                                                                                                                                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| linear / `linear_baseline_001`           | [Train/validation](./reports/figures/linear_baseline_001_learning_curves.png)      | [Counts](./reports/figures/linear_cm_raw.png), [normalized](./reports/figures/linear_cm_normalized.png)           | [Correct/incorrect](./reports/figures/linear_prediction_examples.png)      | [Config](./runs/linear_baseline_001/config.yaml), [history](./runs/linear_baseline_001/history.json), [errors](./runs/linear_baseline_001/error_analysis.json), [checkpoint](./checkpoints/linear_baseline_001/best.pt)                     |
| mlp / `mlp_baseline_002`                 | [Train/validation](./reports/figures/mlp_baseline_002_learning_curves.png)         | [Counts](./reports/figures/mlp_cm_raw.png), [normalized](./reports/figures/mlp_cm_normalized.png)                 | [Correct/incorrect](./reports/figures/mlp_prediction_examples.png)         | [Config](./runs/mlp_baseline_002/config.yaml), [history](./runs/mlp_baseline_002/history.json), [errors](./runs/mlp_baseline_002/error_analysis.json), [checkpoint](./checkpoints/mlp_baseline_002/best.pt)                                 |
| cnn / `cnn_baseline_002`                 | [Train/validation](./reports/figures/cnn_baseline_002_learning_curves.png)         | [Counts](./reports/figures/cnn_cm_raw.png), [normalized](./reports/figures/cnn_cm_normalized.png)                 | [Correct/incorrect](./reports/figures/cnn_prediction_examples.png)         | [Config](./runs/cnn_baseline_002/config.yaml), [history](./runs/cnn_baseline_002/history.json), [errors](./runs/cnn_baseline_002/error_analysis.json), [checkpoint](./checkpoints/cnn_baseline_002/best.pt)                                 |
| lstm / `lstm_baseline_002`               | [Train/validation](./reports/figures/lstm_baseline_002_learning_curves.png)        | [Counts](./reports/figures/lstm_cm_raw.png), [normalized](./reports/figures/lstm_cm_normalized.png)               | [Correct/incorrect](./reports/figures/lstm_prediction_examples.png)        | [Config](./runs/lstm_baseline_002/config.yaml), [history](./runs/lstm_baseline_002/history.json), [errors](./runs/lstm_baseline_002/error_analysis.json), [checkpoint](./checkpoints/lstm_baseline_002/best.pt)                             |
| transformer / `transformer_baseline_002` | [Train/validation](./reports/figures/transformer_baseline_002_learning_curves.png) | [Counts](./reports/figures/transformer_cm_raw.png), [normalized](./reports/figures/transformer_cm_normalized.png) | [Correct/incorrect](./reports/figures/transformer_prediction_examples.png) | [Config](./runs/transformer_baseline_002/config.yaml), [history](./runs/transformer_baseline_002/history.json), [errors](./runs/transformer_baseline_002/error_analysis.json), [checkpoint](./checkpoints/transformer_baseline_002/best.pt) |

## Analysis and assignment page

Read the [Assignment 1 report page](../docs/assignment1.md) for the problem formulation, EDA, learning behavior, comparison, error analysis and limitations. CNN has the highest accuracy and macro-F1 in these runs; MLP offers lower measured training and inference costs. These observations apply to the saved configurations and do not establish a universal architecture ranking.

EDA evidence: [statistics](outputs/eda/dataset_stats.json), [class distribution](outputs/eda/class_distribution.png), [sample images](outputs/eda/sample_images.png).

After evaluating a run, regenerate its diagnostics and benchmark with:

```powershell
python -m assignment1.scripts.analyze_error --run-id cnn_baseline_002
python -m assignment1.scripts.benchmark --run-id cnn_baseline_002 --device cpu --warmup 5 --repeats 20
python -m assignment1.scripts.compare --run-ids linear_baseline_001 mlp_baseline_002 cnn_baseline_002 lstm_baseline_002 transformer_baseline_002
```

Replace IDs with your own to compare new runs. The final command rebuilds the comparison CSV and learning curves from saved run files; each selected run needs test and inference metrics first.

AI disclosures: [repository statement](../AI_USAGE.md) and [assignment disclosure](../docs/assignment1.md#5-ai-usage-disclosure).
