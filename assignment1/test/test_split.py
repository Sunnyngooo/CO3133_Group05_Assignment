import torch
import numpy as np
from torchvision import datasets
from pathlib import Path

from assignment1.src.utils.config import load_config
from assignment1.src.data.split import load_or_create_split

def main(model_name):
    ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]

    config = load_config(model_name) 
    val_ratio = config['data']['val_ratio']
    dataset = datasets.FashionMNIST(root=ASSIGNMENT_DIR / 'data', train=True, download=True)
    split_path = ASSIGNMENT_DIR / "data" / "splits" / "fashion_mnist_split.npz"
    train_indices_1, val_indices_1 = load_or_create_split(dataset=dataset, split_path=split_path,
                                                            val_ratio=val_ratio, 
                                                            seed=config['experiment']['seed'])
    # Check duplicate indices
    train_set = set(train_indices_1)
    val_set = set(val_indices_1) 
    
    assert len(train_set) == len(train_indices_1)
    assert len(val_set) == len(val_indices_1)
    
    # Check disjoint 
    assert train_set.isdisjoint(val_set)
    
    # Check total indices = len(dataset)
    assert len(train_set | val_set) == len(dataset)

    # Check out of range index
    assert not any(i < 0 or i >= len(dataset) for i in train_set | val_set)
    
    # check if the created split file is used
    train_indices_2, val_indices_2 = load_or_create_split(dataset=dataset, split_path=split_path,
                                                    val_ratio=val_ratio, 
                                                    seed=config['experiment']['seed'])

    assert np.array_equal(train_indices_1, train_indices_2)
    assert np.array_equal(val_indices_1, val_indices_2)
    print("All tests passed !")

if __name__ == "__main__":
    main('cnn')  