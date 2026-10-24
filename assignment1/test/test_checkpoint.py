"""Tests for assignment1/src/engine/checkpoint.py.

Run from the directory containing assignment1/:
    python -m pytest assignment1/test/test_checkpoint.py -v
"""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from assignment1.src.engine.checkpoint import load_checkpoint, save_checkpoint


def make_components():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def make_metadata():
    val_metrics = {"loss": 0.31, "accuracy": 0.91, "macro_f1": 0.89}
    config = {"model": "linear", "training": {"epochs": 5, "lr": 0.001}}
    return val_metrics, config


def save_sample(path, model, optimizer):
    val_metrics, config = make_metadata()
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        val_metrics=val_metrics,
        config=config,
    )
    return val_metrics, config


def train_one_step(model, optimizer):
    """Populate Adam's internal state (exp_avg, exp_avg_sq, step)."""
    x = torch.randn(8, 1, 28, 28)
    y = torch.arange(8, dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()


def assert_nested_equal(actual, expected):
    """Compare nested optimizer state, including tensors."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


def test_save_checkpoint_creates_file_and_directories(tmp_path):
    model, optimizer = make_components()
    path = tmp_path / "cnn" / "run_01" / "best.pt"

    save_sample(path, model, optimizer)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_load_restores_model_weights(tmp_path):
    model, optimizer = make_components()
    path = tmp_path / "best.pt"
    expected_weights = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    save_sample(path, model, optimizer)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    assert any(
        not torch.equal(model.state_dict()[name], original)
        for name, original in expected_weights.items()
    )

    load_checkpoint(path=path, model=model, device="cpu")

    for name, original in expected_weights.items():
        torch.testing.assert_close(
            model.state_dict()[name], original, rtol=0, atol=0
        )


def test_load_restores_optimizer_state(tmp_path):
    model, optimizer = make_components()
    train_one_step(model, optimizer)
    assert optimizer.state_dict()["state"]  # Adam state must be initialized.

    path = tmp_path / "best.pt"
    expected_state = deepcopy(optimizer.state_dict())
    save_sample(path, model, optimizer)

    restored_optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
    assert restored_optimizer.state_dict()["state"] == {}

    load_checkpoint(
        path=path,
        model=model,
        optimizer=restored_optimizer,
        device="cpu",
    )

    assert_nested_equal(restored_optimizer.state_dict(), expected_state)


def test_checkpoint_preserves_metadata(tmp_path):
    model, optimizer = make_components()
    path = tmp_path / "best.pt"
    expected_metrics, expected_config = save_sample(path, model, optimizer)

    checkpoint = load_checkpoint(path=path, model=model, device="cpu")

    assert checkpoint["epoch"] == 3
    assert checkpoint["val_metrics"] == expected_metrics
    assert checkpoint["config"] == expected_config
    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint


def test_load_nonexistent_checkpoint_raises_clear_error(tmp_path):
    model, _ = make_components()
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_checkpoint(path=missing, model=model, device="cpu")


def test_load_checkpoint_on_cpu(tmp_path):
    model, optimizer = make_components()
    train_one_step(model, optimizer)
    path = tmp_path / "best.pt"
    save_sample(path, model, optimizer)

    new_model, new_optimizer = make_components()
    checkpoint = load_checkpoint(
        path=path,
        model=new_model,
        optimizer=new_optimizer,
        device=torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 3
    assert all(t.device.type == "cpu" for t in new_model.state_dict().values())
    for state in new_optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"


def test_invalid_path_type(tmp_path):
    model, optimizer = make_components()

    with pytest.raises(TypeError):
        save_checkpoint(
            path=str(tmp_path / "best.pt"),
            model=model,
            optimizer=optimizer,
            epoch=1,
            val_metrics={"macro_f1": 0.8},
            config={}
        )

    with pytest.raises(TypeError):
        load_checkpoint(
            path=str(tmp_path / "best.pt"),
            model=model,
            optimizer=optimizer,
            device="cpu"
        )


def test_load_without_optimizer_is_allowed(tmp_path):
    model, optimizer = make_components()
    path = tmp_path / "best.pt"
    save_sample(path, model, optimizer)

    new_model, _ = make_components()
    checkpoint = load_checkpoint(path=path, model=new_model, device="cpu")

    assert checkpoint["epoch"] == 3
