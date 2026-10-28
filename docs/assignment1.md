# Foundations of Deep Learning Pipelines and Architectures

_Topic:_ From Linear Models to Modern Sequence Models: A Comparative Study for Image Classification

_Group members:_ Đàm Hoài An; Nguyễn Hữu Minh Khôi; Ngô Diễm Quyên.

_Instructor:_ Lê Thành Sách.

## **1. Problem and data**

### **Problem statement**

This project focuses on clothing image classification, which can support applications such as product catalog organization and garment tagging. Using Fashion-MNIST as a controlled benchmark, we conduct a systematic comparative study accross different deep learning architectures.

- **Task & prediction unit:** single-label classification with 10 classes where the prediction unit is a single label.
- **Input & Output:** Each input is a 28 x 28 grayscale image, mapped to one of ten categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.
- **Key challenge:** Low spatial resolution (28 x 28), absence of color channels and visual similiarity between overlapping categories such as Shirt, TShirt/top, Pullover.
- **Objective & Research Question:** We implement, train, and evaluate five mandatory architectures (Linear/Softmax, MLP, CNN, LSTM, and Transformer) under an identical data split and evaluation protocol. Because these models process images through fundamentally different representations (flattened pixel vectors, 2D spatial feature maps, row/column sequences, and patch tokens), our primary research question is: _How do input representation and architectural inductive bias impact predictive performance and computational efficiency on Fashion-MNIST?_

### **Dataset description and EDA**

#### a. Dataset and Splitting protocol

We use **Fashion-MNIST** (Zalando Research, MIT License, compressed IDX format) loaded via `torchvision.datasets.FashionMNIST` as the primary benchmark. No extension dataset is included in the main comparison. All five models share the exact same data partitions and preprocessing pipeline:

| Component                      | Protocol & Configuration                                                                                                                                                                                                                                                                                                                          |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Dataset Scope**              | 70,000 grayscale images ($28 \times 28$ pixels) across 10 garment categories.                                                                                                                                                                                                                                                                     |
| **Train / Val / Test Split**   | **Train:** 51,000 (85% of official train) \| **Val:** 9,000 (15% of official train) \| **Test:** 10,000 (official test set).                                                                                                                                                                                                                      |
| **Splitting & Leakage Policy** | Non-stratified `random_split` (`seed=42`), persisted at `data/splits/fashion_mnist_split_seed42_val0p15.npz`. Train/val indices are strictly disjoint; normalization statistics are computed on the **training subset only**.                                                                                                                     |
| **Preprocessing**              | Scaled to $[0, 1]$ (`float32`), then standardized using training mean ($0.2857$) and std ($0.3529$). No resizing is used since inputs are natively $28 \times 28$. Data augmentation (`RandomAffine`) is implemented in the pipeline but intentionally disabled in main runs to strictly isolate architecture inductive bias across all 5 models. |
| **Batching & Input Shapes**    | Batch size $B = 128$ (shuffled for training only). Base tensor shape $[B, 1, 28, 28]$, transformed into: 784-D vector (Linear/MLP), $1 \times 28 \times 28$ grid (CNN), $28 \times 28$ row sequence (LSTM), and 16 patches of $7 \times 7$ (Transformer).                                                                                         |

---

#### b. Exploratory Data Analysis (EDA)

- **Class Distribution & Imbalance Analysis:**  
  Across the 51,000 training images, class counts range from **5,030** (_T-shirt/top_) to **5,168** (_Sneaker_), yielding an imbalance ratio of **$1.0274$**. Because the training subset is nearly balanced, no class weighting or resampling is required, though **Macro-F1** is still tracked alongside Accuracy for fair multi-class evaluation.

  ![Class Distribution](https://raw.githubusercontent.com/Sunnyngooo/DeepLearning_Assignment/a1/assignment1/outputs/eda/class_distribution.png)

- **Visual Characteristics & Representative Samples:**  
   Items are centered against dark backgrounds with uniform $28 \times 28$ dimensions. While categories like _Trouser_, _Sandal_, and _Bag_ exhibit distinct silhouettes, upper-body garments (_T-shirt/top, Pullover, Coat,_ and _Shirt_) share heavily overlapping outlines. At $28 \times 28$ grayscale resolution, fine texture, collar, and sleeve details are compressed, making fine-grained distinction among upper-body classes the primary challenge for our models.

  ![Representative Samples](https://raw.githubusercontent.com/Sunnyngooo/DeepLearning_Assignment/a1/assignment1/outputs/eda/sample_images.png)

  > **EDA Artifacts:** Raw statistics are logged in [`dataset_stats.json`](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/outputs/eda/dataset_stats.json).

  > **Note:** while index overlap is strictly prevented, visual near-duplicate auditing and annotation noise checks are outside the scope of this automated EDA.

## **2. Methodology**

### **End-to-end pipeline overview**

We implement a unified, reproducible PyTorch pipeline where all five architectures share the exact same data partitions, preprocessing, optimization loop, and evaluation protocol:

```text
Raw Fashion-MNIST [28x28] ──► Disjoint Split (Seed 42) ──► Normalization on Training data ──► DataLoader (B=128)
──► Model-Specific Representation & Forward Pass  ──► Cross Entropy Loss + Adam Optimization
──► Best checkpoint selection (Val Macro-F1) ──► Evaluation & Error Analysis
```

- Self-Implemented vs. Library Components: We implement all five model classes, the training/validation loop, checkpoint manager, and CPU inference-latency benchmarking. Core tensor primitives, layers (nn.Module), autograd, optimizers (torch.optim.Adam), and dataset loaders come from PyTorch and torchvision.
- Loss & Prediction Rule: All models output 10 raw class logits $z \in \mathbb{R}^{B \times 10}$ and are trained with nn.CrossEntropyLoss (no softmax is applied before the loss). Batch loss is weighted by exact batch sample size to accurately account for the final partial batch. Predictions follow $\hat{y}_i = \arg\max_k z_{i,k}$.

### **Input representations & Model architecture**

All five models are trained from scratch (no pretrained weights). Each architecture receives normalized batches of shape $[B, 1, 28, 28]$ and internally transforms them according to its inductive bias:

| Model       | Input Representation                  |                                                                                                                                                                    Architecture & Tensor Flow | Inductive Bias (Strengths & Limits)                                                                                                                                                                  | Parameters |
| ----------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Linear      | Flattened vector with size `[B, 784]` |                                                                                                                              Flatten $\rightarrow$ Linear(784, 10). No hidden layer / dropout | Fast linear baseline. Assumes class separability via global pixel hyperplanes; ignores 2D spatial locality and nonlinear pixel interactions.                                                         | 7,850      |
| MLP         | Flattened vector with size `[B, 784]` |    Linear(784, 256) $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear(256, 128) $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear(128, 10). No hidden layer / dropout | Nonlinear global combinations. Captures cross-pixel interactions via ReLU, but flattening discards 2D topology and lacks translation invariance.                                                     | 235,146    |
| CNN         | 2D spatial grid `[B, 1, 28, 28]`      | 2 Convo Blocks: Conv 3x3 (pad=1) - ReLU - MaxPool 2x2 $\rightarrow$ Flatten (3136) $\rightarrow$ Linear (3136, 64) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(64,10). | Spatial locality & weight sharing. Convolutions extract local edges/textures; pooling builds hierarchical translation invariance with fewer parameters than MLP.                                     | 220,234    |
| LSTM        | Row sequence `[B, 28, 28]`            |                                                                              Two-layer LSTM (hidden=128) $\rightarrow$ final hidden state $\rightarrow$ Dropout $\rightarrow$ Linear (128,10) | Sequential vertical progression. Preserves top-to-bottom row order via recurrent memory, but compresses the whole image into a final bottleneck state and splits vertical neighbors across steps.    | 214,282    |
| Transformer | 16 patch tokens `[B, 16, 128]`        |     Patch projection $\rightarrow$ learned positional embeddings $\rightarrow$ two Transformer encoder layers $\rightarrow$ mean pooling $\rightarrow$ LayerNorm $\rightarrow$ Linear(128,10) | Global pairwise patch attention. Models long-range dependencies across all 16 patches from layer 1, but relies on coarse $7 \times 7$ patch projection and lacks built-in fine-grained spatial bias. | 274,954    |

The project defines these architectures and its training loop; layers, automatic differentiation, optimizers, datasets and transforms come from PyTorch/torchvision. All five main models are trained from scratch without pretrained backbones.

## **3. Experimental setup**

### **Training & Environment configuration**

To ensure a strictly fair comparison, all models are trained and evaluated under the same environment and hyperparameter settings:
| Category | Configuration |
| ----------- | ------------: |
| Hardware & OS |Intel Core i7-8665U CPU (4 cores / 8 threads), 16 GB RAM, Windows 64-bit; CPU execution. |
| Packages | Python 3.12.10, PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu, NumPy 2.5.3, scikit-learn 1.9.1, Matplotlib 3.11.2 (in [requirements.txt](https://github.com/Sunnyngooo/CO3133_Group05_Assignment/blob/a1/requirements.txt)
). |
| Optimization | Adam |
| Learning rate | 0.0005 |
| Weight decay | 0 |
| Batch size | 128 |
| Training epoches | 20 per model |
| Loss | Cross Entropy Loss |
| Augmentation | Disabled |
| Regularization| Dropout (p=0.2) in MLP, CNN, LSTM and Transformer|
| Checkpoint criterion | Highest validation Macro-F1|
| Timing & Latency protocol| Training time: Duration of the full 20-epoch fit loop (including validation and checkpoint, excluding dataset download).<br>Inference latency: CPU.eval() mode, batch_size=1, 4 CPU threads, 5 warmup passes + 20 timed forward repetitions (reporting median ms/image). |
| Reproducibility | Global `seed=42`, deterministic mode enabled, `num_workers=0`. Each run logs its resolved config, per-epoch history, checkpoint, and test predictions under a dedicated run directory.|

### Experimental design & Fairness scope

- Hypotheses: <br>
  (1) Nonlinear hidden representations (MLP) will outperform the Linear baseline <br>
  (2) Explicit 2D spatial filtering (CNN) will achieve superior accuracy-per-parameter over flattened (MLP) and sequential/patch representations (LSTM, Transformer) <br>
  (3) Sequence and attention models will incur higher compute latency on CPU due to recurrent sequential dependencies and quadratic/projection overhead

- Changed factors: Model architectures,internal input representation, parameter countand
- Fixed factors: train/val/test split indices, normalization, optimizer (Adam, `lr=0.0005`), batch size (128), number of epoches (20), loss function and evaluation metrics
- Decision criteria & Fairness boundaries: Models are evaluated on metrics such as Test Accuracy, Macro-F1, per-class confusion patterns and computational cost such as parameter count, training time, single-image CPU latency. While a shared 20-epoch budget and fixed learning rate guarantee baseline protocol consistency, they do not guarantee equal compute expenditure or per-architecture hyperparameter optimality. Furthermore, results reflect single-seed (seed=42) runs without multi-seed variance estimation or isolated hyperparameter tuning.

## **4. Results**

All five models were evaluated on the same 10,000-image Fashion-MNIST test set. Each result uses the checkpoint selected by the highest validation macro-F1.

| Model       | Test accuracy | Test macro-F1 | Parameters | Training (s) | Median inference (ms/image) |
| ----------- | ------------: | ------------: | ---------: | -----------: | --------------------------: |
| cnn         |        91.69% |        0.9163 |    220,234 |      1748.41 |                      0.8216 |
| lstm        |        89.66% |        0.8960 |    214,282 |      1798.18 |                      1.9652 |
| mlp         |        89.08% |        0.8907 |    235,146 |       715.37 |                      0.2037 |
| transformer |        87.98% |        0.8790 |    274,954 |      2133.30 |                      1.9598 |
| linear      |        84.31% |        0.8424 |      7,850 |       941.05 |                      0.0484 |

Median inference is the median CPU forward-pass time with batch size 1 and four CPU threads, measured after five warmup passes over twenty repetitions. It excludes preprocessing and model loading. Results represent one run per model with seed 42.

Source: [comparison CSV](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/comparison.csv).

## **5. Comparison and Discussion**

- CNN achieves the highest accuracy and macro-F1, exceeding MLP by **2.61 accuracy percentage points** while using fewer parameters. This result is consistent with the usefulness of local spatial features and shared convolutional filters for garment classification, although the experiment does not isolate their individual effects.

- MLP offers a useful quality–cost trade-off: it achieves 89.08% accuracy with the shortest measured training time and approximately four times lower inference latency than CNN. Linear remains the smallest and fastest model for inference, but has the lowest classification scores.

- LSTM slightly improves accuracy over MLP, at a substantially higher inference cost. Transformer does not outperform CNN under the tested configuration despite having more parameters and the longest training time. Patch representation, optimization and training budget are possible explanations requiring further experiments.

These results support CNN as the strongest quality choice in this comparison, while MLP provides a lower-cost alternative. They do not establish a universal ranking of architecture families.

## **6. Error Analysis**

Errors are concentrated in visually similar upper-body garment categories.

- **Shirt is a major weakness across models.** CNN correctly identifies 71.2% of Shirts, compared with 57.9% for Linear and 61.6% for Transformer.
- **CNN's most frequent confusion is Shirt → T-shirt/top:** 100 cases, or 10.0% of the 1,000 true Shirts.
- Other CNN confusions include Shirt → Pullover (80 cases), Shirt → Coat (72 cases), and Coat → Pullover (62 cases).
- CNN performs more strongly on visually distinctive categories, with recall of 98.7% for Bag and 98.0% for Sandal.

Similar silhouettes and limited detail in 28 × 28 grayscale images are plausible explanations for garment confusion. These are hypotheses rather than verified causes.

![CNN normalized confusion matrix](https://raw.githubusercontent.com/Sunnyngooo/DeepLearning_Assignment/a1/assignment1/reports/figures/cnn_cm_normalized.png)

The prediction grid displays five correct and five incorrect predictions selected in test-set order. These illustrate model behavior; error frequencies are calculated from the full test set.

![CNN prediction examples](https://raw.githubusercontent.com/Sunnyngooo/DeepLearning_Assignment/a1/assignment1/reports/figures/cnn_prediction_examples.png)

## **6. Limitations**

- Only one seed and one configuration per model are reported; variability across independent runs is unknown.
- Shared training settings do not ensure equally tuned models or matched computational budgets.
- Architecture, representation and regularization change together, so their individual effects cannot be isolated.
- No augmentation or controlled representation ablation is included.
- CPU timings depend on runtime conditions, and the inference benchmark uses a single image with a limited number of repetitions.
- Performance on small grayscale benchmark images does not establish generalization to real-world fashion photographs.

## **7. Conclusion**

CNN provides the best observed classification quality, reaching **91.69% accuracy and 0.9163 macro-F1**. MLP offers a practical balance between predictive quality and measured computational cost, while Linear provides a lightweight baseline.

The comparison shows that greater architectural complexity does not automatically improve results under a fixed training protocol. Future work should prioritize multiple seeds, controlled augmentation or representation studies, and further investigation of Shirt-related errors.

### Run evidence and visualizations

| Model / selected run                   | Curves                                                                                                                                                     | Confusion matrices                                                                                                                                                                                                                                                | Predictions                                                                                                                                        | Run evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| linear / linear_baseline_001           | [Train/validation](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/linear_baseline_001_learning_curves.png)      | [Counts](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/linear_cm_raw.png), [normalized](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/linear_cm_normalized.png)           | [Correct/incorrect](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/linear_prediction_examples.png)      | [Config](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/linear_baseline_001/config.yaml), [history](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/linear_baseline_001/history.json), [errors](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/linear_baseline_001/error_analysis.json), [checkpoint](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/checkpoints/linear_baseline_001/best.pt)                     |
| mlp / mlp_baseline_002                 | [Train/validation](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/mlp_baseline_002_learning_curves.png)         | [Counts](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/mlp_cm_raw.png), [normalized](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/mlp_cm_normalized.png)                 | [Correct/incorrect](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/mlp_prediction_examples.png)         | [Config](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/mlp_baseline_002/config.yaml), [history](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/mlp_baseline_002/history.json), [errors](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/mlp_baseline_002/error_analysis.json), [checkpoint](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/checkpoints/mlp_baseline_002/best.pt)                                 |
| cnn / cnn_baseline_002                 | [Train/validation](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/cnn_baseline_002_learning_curves.png)         | [Counts](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/cnn_cm_raw.png), [normalized](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/cnn_cm_normalized.png)                 | [Correct/incorrect](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/cnn_prediction_examples.png)         | [Config](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/cnn_baseline_002/config.yaml), [history](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/cnn_baseline_002/history.json), [errors](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/cnn_baseline_002/error_analysis.json), [checkpoint](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/checkpoints/cnn_baseline_002/best.pt)                                 |
| lstm / lstm_baseline_002               | [Train/validation](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/lstm_baseline_002_learning_curves.png)        | [Counts](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/lstm_cm_raw.png), [normalized](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/lstm_cm_normalized.png)               | [Correct/incorrect](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/lstm_prediction_examples.png)        | [Config](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/lstm_baseline_002/config.yaml), [history](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/lstm_baseline_002/history.json), [errors](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/lstm_baseline_002/error_analysis.json), [checkpoint](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/checkpoints/lstm_baseline_002/best.pt)                             |
| transformer / transformer_baseline_002 | [Train/validation](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/transformer_baseline_002_learning_curves.png) | [Counts](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/transformer_cm_raw.png), [normalized](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/transformer_cm_normalized.png) | [Correct/incorrect](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/reports/figures/transformer_prediction_examples.png) | [Config](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/transformer_baseline_002/config.yaml), [history](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/transformer_baseline_002/history.json), [errors](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/runs/transformer_baseline_002/error_analysis.json), [checkpoint](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/checkpoints/transformer_baseline_002/best.pt) |

## Source and deliverables

- [Source code](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/assignment1/README.md)
- [M1 draft](https://github.com/Sunnyngooo/CO3133_Group05_Assignment/blob/main/assignment1/draft.ipynb)
- [Reproduction instructions](https://github.com/Sunnyngooo/CO3133_Group05_Assignment/blob/main/assignment1/README.md)
- [Saved checkpoints](https://github.com/Sunnyngooo/CO3133_Group05_Assignment/tree/main/assignment1/checkpoints)
- [Run configurations and logs](https://github.com/Sunnyngooo/CO3133_Group05_Assignment/tree/main/assignment1/runs)
- [Detailed repository AI statement](https://github.com/Sunnyngooo/DeepLearning_Assignment/blob/a1/AI_USAGE.md)
- [Report](https://drive.google.com/file/d/1z_WInHes8nzveDOxyur9ldGsMSyPwfoT/view?usp=sharing)
- [Presentation slide](https://canva.link/a9tolhmr3rypids)
- [YouTube presentation](https://youtu.be/vfxV-s4yhtc)

## AI Usage Disclosure

This document records the use of AI assistance during Assignment 1. The activities below describe AI contributions and the corresponding student verification. The student remained responsible for implementing, testing, and validating the submitted work.

#### 1. Model Architecture and Experimental Design

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Planning model architectures and a consistent experimental protocol.
- **Prompt summary:** Requested recommendations on selecting model configurations and training strategies for five Fashion-MNIST classifiers.
- **AI contribution:** Suggested layer configurations for CNN, LSTM and Transformers.
- **Student verification:** Reviewed the proposed configurations, implemented the models, performed sanity checks, and evaluated their behavior through training and validation results.
- **Affected files/sections:** `src/models/`, `configs/`
- **Responsible member:** Ngô Diễm Quyên

#### 2. Code Implementation Guidance

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Implementing the model training, evaluation, and analysis pipeline.
- **Prompt summary:** Requested recommendations on organizing modules to satisfy the handbook requirements.
- **AI contribution:** Explained implementation logic, function responsibilities, expected inputs and outputs, and recommended module organization.
- **Student verification:** Independently implemented the main pipeline, integrated the components, and executed the scripts to verify functionality.
- **Affected files/sections:** `src/engine/`, `src/metrics/`, `src/models/`, `src/visualization/`, `scripts/`
- **Responsible member:** Ngô Diễm Quyên

#### 3. Code Review and Debugging

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Assisting with debugging coding errors.
- **Prompt summary:** Requested assistance with debugging and explanations of errors involving tensor handling, configuration, checkpoint loading, JSON serialization, and visualization.
- **AI contribution:** Explained the causes of identified errors and suggested targeted corrections without replacing the overall implementation.
- **Student verification:** Applied the relevant corrections, reran affected functions and scripts, and inspected the resulting outputs.
- **Affected files/sections:** `src/models/`, `src/metrics/`, `src/visualization/`, `scripts/`
- **Responsible member:** Ngô Diễm Quyên

#### 4. Sanity Test Generation and Debugging Support

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Creating sanity tests.
- **Prompt summary:** Requested test cases for model forward passes, tensor dimensions, parameter counts, gradient propagation, inference benchmarking, confusion matrices, and classification error analysis to ensure that the code worked as expected.
- **AI contribution:** Generated example sanity-test code, expected outputs, and validation assertions for detecting implementation errors.
- **Student verification:** Reviewed the generated tests to ensure they covered the requirements, executed the tests locally, reviewed the results, and corrected identified issues before running the full experiments.
- **Affected files/sections:** `tests/`
- **Responsible member:** Ngô Diễm Quyên

#### 5. Evaluation and Visualization Support

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Designing evaluation procedures and visualizations for model comparison.
- **Prompt summary:** Requested guidance on inference benchmarking, learning curves, and correct/incorrect prediction visualizations.
- **AI contribution:** Recommended evaluation procedures, visualization layouts, and approaches to organizing experimental results.
- **Student verification:** Implemented the evaluation and visualization functions, checked generated figures, and compared calculated metrics with saved test results.
- **Affected files/sections:** `src/metrics/`, `src/visualization/`, `scripts/compare.py`, `scripts/analyze_error.py`; report Results section.
- **Responsible member:** Ngô Diễm Quyên

#### 6. Creating requirements.txt

- **Tool:** Gemini Flash
- **Used by:** Ngô Diễm Quyên
- **Task:** Create requirements.txt
- **Prompt summary:** Asked how to get the version of packages used in the notebook
- **AI contribution:** Generate code to get the dependency of packages used in the assignment
- **Student verification:** review the version of packages
- **Affected files/sections:** `assignment1/requirements.txt`
- **Responsible member:** Ngô Diễm Quyên

#### 7. Documenting hardware specification

- **Tool:** Gemini Flash
- **Used by:** Ngô Diễm Quyên
- **Task:** Documenting hardware specification
- **Prompt summary:** Asked how to structure the hardware information in the README.md
- **AI contribution:** format computer's information into the desired outlines
- **Student verification:** cross-checked the result with the system information
- **Affected files/sections:** `assignment1/README.md`
- **Responsible member:** Ngô Diễm Quyên

#### 8. Converting PyTorch tensor targets

- **Tool:** Gemini Flash
- **Used by:** Ngô Diễm Quyên
- **Task:** Converting PyTorch tensor targets
- **Prompt summary:** Asked how to remove the tensor(...) wrapper when inspecting labels
- **AI contribution:** Suggested .tolist() and .numpy() conversion methods
- **Student verification:** Ran snippet in notebook cell and verified list output
- **Affected files/sections:** `draft.ipynb`
- **Responsible member:** Ngô Diễm Quyên
