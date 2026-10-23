import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from assignment1.src.utils.config import load_config
from assignment1.src.data.transforms import build_train_transform,  build_val_transform
from assignment1.src.data.split import get_split_path, load_or_create_split
from assignment1.src.data.statistics import compute_mean_std
from pathlib import Path

def main(model_name): 
    ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
    config = load_config(model_name)
    dataset = FashionMNIST(root= ASSIGNMENT_DIR / 'data', train=True, download=True)
    img, label = dataset[0]

    split_path = get_split_path(config)
    train_indices, _ = load_or_create_split(dataset=dataset, split_path=split_path, val_ratio=config['data']['val_ratio'], seed=config['experiment']['seed'])
    mean, std = compute_mean_std(dataset, train_indices)

    train_transform = build_train_transform(config, mean, std)
    val_transform = build_val_transform(config, mean, std)

    train_img = train_transform(img)
    eval_img = val_transform(img)

    # Check shape and dtype 
    assert train_img.shape ==  (1,28,28)
    assert eval_img.shape ==  (1,28,28)
    assert train_img.dtype == torch.float32
    assert eval_img.dtype == torch.float32
    
    # Check normalization
    raw_pixels = torch.from_numpy(np.array(img, copy=True)).float() / 255.0
    expected_norm = (raw_pixels - mean) / std

    assert torch.allclose(eval_img.squeeze(0), expected_norm, atol=1e-6)

    # check deterministic 
    eval_1 = val_transform(img)
    eval_2 = val_transform(img)
    assert torch.equal(eval_1, eval_2)
    print("All tests passed !")

if __name__ == "__main__":
    main('cnn')