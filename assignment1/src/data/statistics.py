import torch

def compute_mean_std(dataset, train_indices):
    pixels = dataset.data[train_indices].float() / 255.0
    mean = pixels.mean().item()
    std = pixels.std(unbiased=True).item()
    
    return mean, std

def compute_class_distribution(labels, num_classes):
    class_counts = torch.bincount(labels, minlength=num_classes)
    
    min_count = class_counts.min().item()
    max_count = class_counts.max().item()
    
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return class_counts, imbalance_ratio