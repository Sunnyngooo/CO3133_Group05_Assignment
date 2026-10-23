import torch
import numpy as np
from torchvision import datasets
from pathlib import Path

from assignment1.src.utils.config import load_config
from assignment1.src.data.split import load_or_create_split

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]

config = load_config("cnn") 
val_ratio = config['data']['val_ratio']
dataset = datasets.FashionMNIST(root=ASSIGNMENT_DIR / 'data', train=True, download=True)
split_path = ASSIGNMENT_DIR / "data" / "splits" / "fashion_mnist_split.npz"
train_indices_1, val_indices_1 = load_or_create_split(dataset=dataset, split_path=split_path,
                                                        val_ratio=val_ratio, 
                                                        seed=config['experiment']['seed'])
try:
    assert len(train_indices_1) == 51000
except AssertionError as e:
    print(len(train_indices_1))
    print(e)
try:
    assert len(val_indices_1) == 9000
except AssertionError as e:
    print(len(val_indices_1))
    print(e)

train_set = set(train_indices_1)
val_set = set(val_indices_1) 
if len(train_set) != len(train_indices_1):
    raise ValueError("Duplicate indices in training split.")

if len(val_set) != len(val_indices_1):
    raise ValueError("Duplicate indices in validation split.")

if any(i < 0 or i >= len(dataset) for i in train_set | val_set):
    raise ValueError("Split indices out of range.")
train_indices_2, val_indices_2 = load_or_create_split(dataset=dataset, split_path=split_path,
                                                  val_ratio=val_ratio, 
                                                  seed=config['experiment']['seed'])

assert np.array_equal(train_indices_1, train_indices_2)
assert np.array_equal(val_indices_1, val_indices_2)
print("passed")

#read file npz 
from numpy import load
data = load('assignment1/data/splits/fashion_mnist_split.npz')
lst = data.files 
for item in lst:
    print(item)
    print(data[item])
    
