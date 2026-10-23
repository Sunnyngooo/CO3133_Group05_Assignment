
def compute_mean_std(dataset, train_indices):
    pixels = dataset.data[train_indices].float() / 255.0
    mean = pixels.mean().item()
    std = pixels.std(unbiased=True).item()
    
    return mean, std