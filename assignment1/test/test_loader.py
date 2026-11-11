import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from assignment1.src.data import dataset as dataset_module


def test_loader_split_batches_and_training_only_normalization(monkeypatch, config, fake_fashion_mnist):
    monkeypatch.setattr(dataset_module, 'FashionMNIST', fake_fashion_mnist)
    train, val, test, stats = dataset_module.build_dataloaders(config)
    assert [len(loader.dataset) for loader in (train, val, test)] == [85, 15, 20]
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    pixels = train.dataset.dataset.data[train.dataset.indices].float() / 255.0
    assert stats['mean'] == pytest.approx(pixels.mean().item())
    assert stats['std'] == pytest.approx(pixels.std(unbiased=True).item())
    images, labels = next(iter(train))
    assert images.shape == (16, 1, 28, 28)
    assert labels.shape == (16,)
    assert labels.dtype == torch.int64
    assert images.dtype == torch.float32
    assert torch.isfinite(images).all()
    assert labels.min() >= 0 and labels.max() < 10
    raw_image = val.dataset.dataset.data[val.dataset.indices[0]].float() / 255.0
    image, _ = val.dataset[0]
    torch.testing.assert_close(image.squeeze(0), (raw_image - stats['mean']) / stats['std'])


def test_loader_seed_reproduces_first_training_batch(monkeypatch, config, fake_fashion_mnist):
    monkeypatch.setattr(dataset_module, 'FashionMNIST', fake_fashion_mnist)
    first, *_ = dataset_module.build_dataloaders(config)
    second, *_ = dataset_module.build_dataloaders(config)
    first_images, first_labels = next(iter(first))
    second_images, second_labels = next(iter(second))
    assert torch.equal(first_images, second_images)
    assert torch.equal(first_labels, second_labels)
