
import pytest
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from assignment1.test.dummy_model import DummyModel
from assignment1.src.engine.evaluator import evaluate, validate, test as evaluate_test
from assignment1.src.metrics.cls_metrics import compute_classification_metrics


DEVICE = torch.device("cpu")


def make_dataloader(num_samples=11, batch_size=4):
    """Create synthetic Fashion-MNIST-shaped data."""
    generator = torch.Generator().manual_seed(42)

    images = torch.randn(
        num_samples, 1, 28, 28,
        generator=generator
    )
    labels = torch.arange(num_samples, dtype=torch.long) % 10

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )


def make_model_and_loss():
    """Initialize deterministic model and loss function."""
    torch.manual_seed(42)

    model = DummyModel().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    return model, loss_fn


def test_evaluate_valid_metrics():
    """Evaluation should return valid metrics."""
    loader = make_dataloader()
    model, loss_fn = make_model_and_loss()

    result = evaluate(model, loader, loss_fn, DEVICE)

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "loss", "accuracy", "macro_f1"
    }

    assert all(
        isinstance(value, float)
        for value in result.values()
    )

    assert all(
        torch.isfinite(torch.tensor(value)).item()
        for value in result.values()
    )

    assert result["loss"] >= 0
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["macro_f1"] <= 1


def test_parameters_unchanged():
    """Evaluation must not modify model parameters or buffers."""
    loader = make_dataloader()
    model, loss_fn = make_model_and_loss()

    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    evaluate(model, loader, loss_fn, DEVICE)

    state_after = model.state_dict()

    for name, before in state_before.items():
        assert torch.equal(before, state_after[name]), (
            f"Model state changed during evaluation: {name}"
        )


def test_evaluation_mode_enabled():
    """evaluate() must switch the model into eval mode."""
    loader = make_dataloader()
    model, loss_fn = make_model_and_loss()

    model.train()
    assert model.training is True

    evaluate(model, loader, loss_fn, DEVICE)

    assert model.training is False


def test_no_grad_during_forward():
    """Forward pass must run without gradient tracking."""
    loader = make_dataloader()
    model, loss_fn = make_model_and_loss()

    grad_states = []

    def forward_hook(module, inputs, output):
        grad_states.append(torch.is_grad_enabled())

    handle = model.register_forward_hook(forward_hook)

    try:
        evaluate(model, loader, loss_fn, DEVICE)
    finally:
        handle.remove()

    assert len(grad_states) == len(loader)
    assert all(state is False for state in grad_states)


def test_epoch_loss_and_metrics():
    """Metrics must be calculated over all samples."""
    loader = make_dataloader(num_samples=11, batch_size=4)
    model, loss_fn = make_model_and_loss()

    images, labels = loader.dataset.tensors

    model.eval()

    with torch.no_grad():
        logits = model(images)
        expected_loss = loss_fn(logits, labels).item()
        predictions = logits.argmax(dim=1)

    expected_metrics = compute_classification_metrics(
        y_true=labels,
        y_pred=predictions,
        num_classes=10
    )

    result = evaluate(model, loader, loss_fn, DEVICE)

    assert result["loss"] == pytest.approx(
        expected_loss, rel=1e-5
    )
    assert result["accuracy"] == pytest.approx(
        expected_metrics["accuracy"]
    )
    assert result["macro_f1"] == pytest.approx(
        expected_metrics["macro_f1"]
    )


@pytest.mark.parametrize(
    "evaluation_fn",
    [validate, evaluate_test]
)
def test_validate_and_test_return_metrics(evaluation_fn):
    """Both wrappers must return the evaluation dictionary."""
    loader = make_dataloader()
    model, loss_fn = make_model_and_loss()

    result = evaluation_fn(model, loader, loss_fn, DEVICE)

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "loss", "accuracy", "macro_f1"
    }


def test_validate_test_consistency():
    """Same model and dataset must produce identical results."""
    loader = make_dataloader()
    model, loss_fn = make_model_and_loss()

    val_result = validate(model, loader, loss_fn, DEVICE)
    test_result = evaluate_test(model, loader, loss_fn, DEVICE)

    for metric in ("loss", "accuracy", "macro_f1"):
        assert val_result[metric] == pytest.approx(
            test_result[metric]
        )


def test_empty_dataloader():
    """Evaluation of an empty loader must raise ValueError."""
    loader = make_dataloader(num_samples=0)
    model, loss_fn = make_model_and_loss()

    with pytest.raises(ValueError):
        evaluate(model, loader, loss_fn, DEVICE)


@pytest.mark.parametrize(
    "evaluation_fn",
    [validate, evaluate_test]
)
def test_empty_dataloader_wrappers(evaluation_fn):
    """Validation and testing should propagate empty-loader errors."""
    loader = make_dataloader(num_samples=0)
    model, loss_fn = make_model_and_loss()

    with pytest.raises(ValueError):
        evaluation_fn(model, loader, loss_fn, DEVICE)
