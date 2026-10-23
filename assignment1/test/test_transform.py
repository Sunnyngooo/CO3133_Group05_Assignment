import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from assignment1.src.utils.config import load_config
from assignment1.src.data.transforms import build_train_transform,  build_val_transform
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
config = load_config('cnn')
dataset = FashionMNIST(root= ASSIGNMENT_DIR / 'data', train=True, download=True)

img, label = dataset[0]
train_transform = build_train_transform(config)
val_transform = build_val_transform(config)

train_img = train_transform(img)
eval_img = val_transform(img)
print(train_img.shape)
print(train_img.dtype)
print(train_img.min(), train_img.max())

# check deterministic 
eval_1 = val_transform(img)
eval_2 = val_transform(img)
assert torch.equal(eval_1, eval_2)