import torch 
from torch.utils.data import random_split
import numpy as np
from pathlib import Path

def get_split_path(config: dict) -> Path: 
    assignment_dir = Path(__file__).resolve().parents[2]
    data_path = Path(config['data']['root'])
    if not data_path.is_absolute():
            data_path = assignment_dir / data_path
        
    seed = config['experiment']['seed']
    val_ratio = config['data']['val_ratio']
    
    ratio_name = str(val_ratio).replace('.','p')
    
    return (data_path / 'splits' / f"fashion_mnist_split_seed{seed}_val{ratio_name}.npz")
    
def load_or_create_split(dataset, split_path: Path, val_ratio: float, seed: int):
    n_samples = len(dataset)
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved:
            saved_seed = int(saved['seed'])
            saved_ratio = float(saved['val_ratio'])
            saved_n = int(saved['n_samples'])
            
            if saved_seed != seed or not np.isclose(saved_ratio, val_ratio) or n_samples != saved_n:
                raise ValueError("Saved split configuration does not match current configuration. Check/delete old split file and retry.")
            
            train_indices = saved['train_indices']
            val_indices = saved['val_indices']
            
            # check overlap
            train_set = set(train_indices)
            val_set = set(val_indices)
            if not train_set.isdisjoint(val_set):
                raise ValueError("The training data is overlapped with the validation data.")
            
            if len(train_set | val_set) != n_samples:
                raise ValueError("The split does not cover the original training dataset.")
            
            # Check duplicate
            if len(train_indices) != len(train_set):
                raise ValueError("There are duplicates indices in training split.")
            
            if len(val_indices) != len(val_set):
                raise ValueError("There are duplicates indices in validation split.")
            
            # Check index range
            for i in train_set | val_set:
                if i < 0 or i >= n_samples: 
                    raise ValueError("Splits contain out of range index.")
                
            # Check split size 
            expected_size_val = int(val_ratio * n_samples)
            if expected_size_val != len(val_indices):
                raise ValueError(f"Validation split size does not meet the expected size: current size - {len(val_indices)}; expected size - {expected_size_val}")
            
            if (n_samples - expected_size_val) != len(train_indices):
                raise ValueError(f"Training split size does not meet the expected size: current size - {len(train_indices)}; expected size - {n_samples - expected_size_val}")
    else: 
        generator = torch.Generator().manual_seed(seed)
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size
        
        train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, 
                            train_indices=train_indices, 
                            val_indices=val_indices, 
                            seed=seed, val_ratio=val_ratio, n_samples=n_samples)
    return train_indices, val_indices
            
        