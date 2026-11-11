import numpy as np
import pytest
import torch
from PIL import Image

from assignment1.src.data.transforms import build_train_transform, build_val_transform


@pytest.mark.parametrize('augmentation', [False, True])
def test_transforms_produce_finite_image_tensors(config, augmentation):
    config['data']['augmentation']['enabled'] = augmentation
    image = Image.fromarray(np.arange(784, dtype=np.uint8).reshape(28, 28))
    for transform in (build_train_transform(config, 0.3, 0.35), build_val_transform(config, 0.3, 0.35)):
        result = transform(image)
        assert result.shape == (1, 28, 28)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()


def test_normalization_and_evaluation_are_deterministic(config):
    pixels = np.arange(784, dtype=np.uint8).reshape(28, 28)
    image = Image.fromarray(pixels)
    transform = build_val_transform(config, mean=0.3, std=0.35)
    expected = (torch.from_numpy(pixels).float() / 255.0 - 0.3) / 0.35
    torch.testing.assert_close(transform(image).squeeze(0), expected)
    assert torch.equal(transform(image), transform(image))
    torch.testing.assert_close(build_train_transform(config, 0.3, 0.35)(image), transform(image))
