# AI Usage Disclosure — Assignment 1

This document records the use of AI assistance during Assignment 1. The activities below describe AI contributions and the corresponding student verification. The student remained responsible for implementing, testing, and validating the submitted work.

## 1. Model Architecture and Experimental Design

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Planning model architectures and a consistent experimental protocol.
- **Prompt summary:** Requested recommendations on selecting model configurations and training strategies for five Fashion-MNIST classifiers.
- **AI contribution:** Suggested layer configurations for CNN, LSTM and Transformers.
- **Student verification:** Reviewed the proposed configurations, implemented the models, performed sanity checks, and evaluated their behavior through training and validation results.
- **Affected files/sections:** `src/models/`, `configs/`
- **Responsible member:** Ngô Diễm Quyên

## 2. Code Implementation Guidance

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Implementing the model training, evaluation, and analysis pipeline.
- **Prompt summary:** Requested recommendations on organizing modules to satisfy the handbook requirements.
- **AI contribution:** Explained implementation logic, function responsibilities, expected inputs and outputs, and recommended module organization.
- **Student verification:** Independently implemented the main pipeline, integrated the components, and executed the scripts to verify functionality.
- **Affected files/sections:** `src/engine/`, `src/metrics/`, `src/models/`, `src/visualization/`, `scripts/`
- **Responsible member:** Ngô Diễm Quyên

## 3. Code Review and Debugging

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Assisting with debugging coding errors.
- **Prompt summary:** Requested assistance with debugging and explanations of errors involving tensor handling, configuration, checkpoint loading, JSON serialization, and visualization.
- **AI contribution:** Explained the causes of identified errors and suggested targeted corrections without replacing the overall implementation.
- **Student verification:** Applied the relevant corrections, reran affected functions and scripts, and inspected the resulting outputs.
- **Affected files/sections:** `src/models/`, `src/metrics/`, `src/visualization/`, `scripts/`
- **Responsible member:** Ngô Diễm Quyên

## 4. Sanity Test Generation and Debugging Support

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Creating sanity tests.
- **Prompt summary:** Requested test cases for model forward passes, tensor dimensions, parameter counts, gradient propagation, inference benchmarking, confusion matrices, and classification error analysis to ensure that the code worked as expected.
- **AI contribution:** Generated example sanity-test code, expected outputs, and validation assertions for detecting implementation errors.
- **Student verification:** Reviewed the generated tests to ensure they covered the requirements, executed the tests locally, reviewed the results, and corrected identified issues before running the full experiments.
- **Affected files/sections:** `tests/`
- **Responsible member:** Ngô Diễm Quyên

## 5. Evaluation and Visualization Support

- **Tool:** ChatGPT (OpenAI, GPT-6)
- **Used by:** Ngô Diễm Quyên
- **Task:** Designing evaluation procedures and visualizations for model comparison.
- **Prompt summary:** Requested guidance on inference benchmarking, learning curves, and correct/incorrect prediction visualizations.
- **AI contribution:** Recommended evaluation procedures, visualization layouts, and approaches to organizing experimental results.
- **Student verification:** Implemented the evaluation and visualization functions, checked generated figures, and compared calculated metrics with saved test results.
- **Affected files/sections:** `src/metrics/`, `src/visualization/`, `scripts/compare.py`, `scripts/analyze_error.py`; report Results section.
- **Responsible member:** Ngô Diễm Quyên
