"""Small offline datasets shared by Assignment 1 tests."""

from copy import deepcopy

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from assignment1.src.utils.config import load_config


class FakeFashionMNIST(Dataset):
    """Exercise real transforms/loaders without downloading Fashion-MNIST."""

    classes = [f"class_{index}" for index in range(10)]

    def __init__(self, root, train=True, download=False, transform=None):
        self.transform = transform
        size = 100 if train else 20
        generator = torch.Generator().manual_seed(42 if train else 43)
        self.data = torch.randint(0, 256, (size, 28, 28), generator=generator, dtype=torch.uint8)
        self.targets = torch.arange(size) % 10

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index].numpy())
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index])


@pytest.fixture
def config(tmp_path):
    config = deepcopy(load_config('cnn'))
    config['data']['root'] = str(tmp_path / 'data')
    config['data']['batch_size'] = 16
    return config


@pytest.fixture
def fake_fashion_mnist():
    return FakeFashionMNIST
