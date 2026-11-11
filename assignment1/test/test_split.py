import numpy as np
import pytest

from assignment1.src.data.split import load_or_create_split


def test_split_is_disjoint_complete_and_persisted(tmp_path):
    dataset = range(100)
    path = tmp_path / 'splits' / 'split.npz'
    train, val = load_or_create_split(dataset, path, val_ratio=0.15, seed=42)
    assert len(train) == len(set(train)) == 85
    assert len(val) == len(set(val)) == 15
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(100))
    loaded_train, loaded_val = load_or_create_split(dataset, path, 0.15, 42)
    np.testing.assert_array_equal(train, loaded_train)
    np.testing.assert_array_equal(val, loaded_val)
    other_train, _ = load_or_create_split(dataset, tmp_path / 'other.npz', 0.15, 42)
    np.testing.assert_array_equal(train, other_train)


@pytest.mark.parametrize('size,ratio,seed', [(100, 0.15, 43), (100, 0.2, 42), (101, 0.15, 42)])
def test_saved_split_rejects_configuration_mismatch(tmp_path, size, ratio, seed):
    path = tmp_path / 'split.npz'
    load_or_create_split(range(100), path, 0.15, 42)
    with pytest.raises(ValueError, match='does not match'):
        load_or_create_split(range(size), path, ratio, seed)


def test_saved_split_rejects_train_validation_overlap(tmp_path):
    path = tmp_path / 'split.npz'
    np.savez_compressed(path, train_indices=np.arange(85), val_indices=np.arange(15),
                        seed=42, val_ratio=0.15, n_samples=100)
    with pytest.raises(ValueError, match='overlapped'):
        load_or_create_split(range(100), path, 0.15, 42)
