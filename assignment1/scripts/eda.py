import torch 
import math
import json
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from assignment1.src.utils.config import load_config
from assignment1.src.data.split import load_or_create_split, get_split_path
from assignment1.src.data.statistics import compute_mean_std, compute_class_distribution
    
def plot_class_distribution(class_counts, classes, output_dir):
    plt.figure(figsize=(12,6))
    plt.bar(classes, class_counts.tolist())
    plt.title("Class distribution of training data")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()
    
    print(f"The class distribution graph is saved in {output_dir}.")
    
def plot_sample_images(images, labels, classes, output_dir):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat): 
        idx = torch.where(labels == i)[0]
        
        image = images[idx[0]]
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"Label {i}: {classes[i]}")
        ax.axis("off")    
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close(fig)
    print(f"The sample images are saved in {output_dir}.")

def compute_statistics(dataset, indices):
    images = dataset.data[indices]
    
    mean, std = compute_mean_std(dataset, indices)
    
    pixels = images.float() / 255.0 
    
    statistics = {
        'num_samples': len(images),
        'raw_image_shape': list(images[0].shape), # (H, W)
        'model_input_image_shape': list(images[0].unsqueeze(0).shape), # (C, H, W)
        'min_pixel': pixels.min().item(),
        'max_pixel': pixels.max().item(),
        'mean': mean,
        'std': std
    }
    return statistics

def main(model_name):
    # Load config
    config = load_config(model_name=model_name)
    val_ratio = config['data']['val_ratio'] 
    seed = config['experiment']['seed']
    
    assignment_dir = Path(__file__).resolve().parents[1]    
    
    # Load raw data
    data_path = Path(config['data']['root'])
    if not data_path.is_absolute():
        data_path = assignment_dir / data_path
    
    raw_dataset = FashionMNIST(root= data_path, train=True, download=True)
    
    # Get split indices      
    split_path = get_split_path(config)
    train_indices, _ = load_or_create_split(dataset=raw_dataset, split_path=split_path, val_ratio=val_ratio, seed=seed)
    
    # Extract training data
    train_images = raw_dataset.data[train_indices]
    train_labels = raw_dataset.targets[train_indices]
    class_names = raw_dataset.classes
    
    # Prepare output dir
    output_dir = assignment_dir / 'outputs' / 'eda'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute class distribution
    class_counts, imbalance_ratio = compute_class_distribution(train_labels, len(class_names))
    
    # Class distribution
    plot_class_distribution(class_counts=class_counts, classes=class_names, output_dir=(output_dir / "class_distribution.png"))
    
    # Extract sample for each classes 
    plot_sample_images(train_images, train_labels, class_names, output_dir / "sample_images.png")
    
    # Compute pixel statistics using only the training split
    statistics = compute_statistics(raw_dataset, train_indices)
    statistics['class_counts'] = {class_name: count for class_name, count in zip(class_names, class_counts.tolist())}
    statistics['imbalance_ratio'] = imbalance_ratio if math.isfinite(imbalance_ratio) else None
    
    # Save statistics 
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(statistics, f, indent=4)
        
if __name__ == "__main__":
    main("cnn")