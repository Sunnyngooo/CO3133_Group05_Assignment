
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from assignment1.src.models.linear import LinearClassifier
from assignment1.src.engine.trainer import train_one_epoch
from assignment1.src.metrics.cls_metrics import compute_classification_metrics


DEVICE = torch.device("cpu")


def make_dataloader(num_samples=11, batch_size=4):
    """Create synthetic Fashion-MNIST-shaped data for testing."""
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


def make_training_components(lr=0.01):
    """Create a model, loss function and optimizer."""
    torch.manual_seed(42)

    model = LinearClassifier().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fn, optimizer


def test_train_one_epoch_smoke():
    """Training should run successfully and return valid metrics."""
    loader = make_dataloader()
    model, loss_fn, optimizer = make_training_components()

    result = train_one_epoch(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"loss", "accuracy", "macro_f1"}

    assert all(isinstance(value, float) for value in result.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())

    assert result["loss"] >= 0
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["macro_f1"] <= 1


def test_model_parameters_updated():
    """At least one model parameter should change after training."""
    loader = make_dataloader()
    model, loss_fn, optimizer = make_training_components()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    train_one_epoch(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE
    )

    parameters_after = list(model.parameters())

    changed = [
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, parameters_after)
    ]

    assert any(changed), "Model parameters were not updated."


def test_epoch_metrics_and_partial_batch():
    """
    Check loss and metrics across the full epoch.

    11 samples, batch_size=4 -> batches of 4, 4, 3.
    Using lr=0 means predictions remain unchanged during training.
    """
    loader = make_dataloader(num_samples=11, batch_size=4)
    model, loss_fn, optimizer = make_training_components(lr=0.0)

    images, labels = loader.dataset.tensors

    with torch.no_grad():
        logits = model(images)
        expected_loss = loss_fn(logits, labels).item()
        predictions = logits.argmax(dim=1)

    expected_metrics = compute_classification_metrics(
        y_true=labels,
        y_pred=predictions,
        num_classes=10
    )

    result = train_one_epoch(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE
    )

    assert result["loss"] == pytest.approx(expected_loss, rel=1e-5)
    assert result["accuracy"] == pytest.approx(
        expected_metrics["accuracy"]
    )
    assert result["macro_f1"] == pytest.approx(
        expected_metrics["macro_f1"]
    )


def test_training_mode_enabled():
    """train_one_epoch must switch the model into training mode."""
    loader = make_dataloader()
    model, loss_fn, optimizer = make_training_components()

    model.eval()
    assert model.training is False

    train_one_epoch(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE
    )

    assert model.training is True


def test_empty_dataloader():
    """An empty training loader should raise a meaningful error."""
    loader = make_dataloader(num_samples=0)
    model, loss_fn, optimizer = make_training_components()

    with pytest.raises(ValueError):
        train_one_epoch(
            model=model,
            train_loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=DEVICE
        )
