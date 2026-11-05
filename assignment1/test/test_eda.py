import json

import matplotlib
matplotlib.use('Agg')
import pytest
import torch

from assignment1.scripts import eda


def test_eda_generates_statistics_and_figures_in_isolation(monkeypatch, tmp_path, config, fake_fashion_mnist):
    # Redirect the script's assignment directory so tracked EDA artifacts stay untouched.
    script = tmp_path / 'assignment1' / 'scripts' / 'eda.py'
    monkeypatch.setattr(eda, '__file__', str(script))
    monkeypatch.setattr(eda, 'load_config', lambda model_name: config)
    monkeypatch.setattr(eda, 'FashionMNIST', fake_fashion_mnist)
    eda.main('cnn')
    output = tmp_path / 'assignment1' / 'outputs' / 'eda'
    stats = json.loads((output / 'dataset_stats.json').read_text())
    assert stats['num_samples'] == 85
    assert stats['raw_image_shape'] == [28, 28]
    assert stats['model_input_image_shape'] == [1, 28, 28]
    assert sum(stats['class_counts'].values()) == 85
    assert 0 <= stats['min_pixel'] <= stats['max_pixel'] <= 1
    counts = list(stats['class_counts'].values())
    assert stats['imbalance_ratio'] == pytest.approx(max(counts) / min(counts))
    for name in ('class_distribution.png', 'sample_images.png'):
        assert (output / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_statistics_exclude_unselected_images(fake_fashion_mnist, tmp_path):
    dataset = fake_fashion_mnist(tmp_path)
    dataset.data[0].fill_(0)
    dataset.data[1].fill_(255)
    stats = eda.compute_statistics(dataset, [0, 1])
    assert stats['mean'] == pytest.approx(0.5)
    assert stats['num_samples'] == 2
    expected_std = torch.cat([torch.zeros(784), torch.ones(784)]).std().item()
    assert stats['std'] == pytest.approx(expected_std)
