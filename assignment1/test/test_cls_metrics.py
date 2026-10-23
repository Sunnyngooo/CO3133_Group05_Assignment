
import numpy as np
import pytest
import torch

from assignment1.src.metrics.cls_metrics import compute_classification_metrics


def test_perfect_predictions():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]

    result = compute_classification_metrics(y_true, y_pred, num_classes=3)

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_partially_correct():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = compute_classification_metrics(y_true, y_pred, num_classes=2)

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(0.7333333333)


def test_missing_class():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = compute_classification_metrics(y_true, y_pred, num_classes=3)

    expected_macro_f1 = (2 / 3 + 0.8 + 0.0) / 3

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(expected_macro_f1)


def test_mismatched_lengths():
    y_true = [0, 1, 2]
    y_pred = [0, 1]

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, num_classes=3)


def test_invalid_labels():
    y_true = [0, 1]
    y_pred = [0, 10]

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, num_classes=10)


@pytest.mark.parametrize("num_classes", [0, -1])
def test_invalid_val_num_classes(num_classes):
    y_true = [0, 1]
    y_pred = [0, 1]

    with pytest.raises(ValueError):
        compute_classification_metrics(
            y_true, y_pred, num_classes=num_classes
        )
        
@pytest.mark.parametrize("num_classes", [2.5, True])
def test_invalid_type_num_classes(num_classes):
    y_true = [0, 1]
    y_pred = [0, 1]

    with pytest.raises(TypeError):
        compute_classification_metrics(
            y_true, y_pred, num_classes=num_classes
        )


def test_torch_tensor_input():
    y_true = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    y_pred = torch.tensor([0, 1, 2, 0], dtype=torch.int64)

    result = compute_classification_metrics(y_true, y_pred, num_classes=3)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"accuracy", "macro_f1"}

    assert isinstance(result["accuracy"], float)
    assert isinstance(result["macro_f1"], float)

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(0.7777777778)


def test_numpy_array_input():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    result = compute_classification_metrics(y_true, y_pred, num_classes=3)

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_empty_input():
    with pytest.raises(ValueError):
        compute_classification_metrics([], [], num_classes=10)


def test_invalid_dimensions():
    y_true = [[0, 1], [1, 0]]
    y_pred = [[0, 1], [1, 0]]

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, num_classes=2)


def test_float_labels():
    y_true = [0.0, 1.0]
    y_pred = [0.0, 1.0]

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, num_classes=2)
