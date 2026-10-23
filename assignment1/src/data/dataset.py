from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from assignment1.src.data.transforms import build_val_transform, build_train_transform
from assignment1.src.data.split import load_or_create_split, get_split_path
from assignment1.src.utils.seed import make_generator, seed_worker
from assignment1.src.data.statistics import compute_mean_std

def build_dataloaders(config):
    """
    Return training, validation, test loader.
    """    
    assignment_path = Path(__file__).resolve().parents[2]
    data_config = config['data']
    seed = config['experiment']['seed']
    val_ratio = config["data"]["val_ratio"]
    data_path = Path(config['data']['root'])
    if not data_path.is_absolute():
        data_path = assignment_path / data_path
    
    # Load raw dataset
    train_full_dataset = FashionMNIST(root=data_path, train=True, download=True)
    val_full_dataset = FashionMNIST(root=data_path, train=True, download=True)
    test_dataset = FashionMNIST(root=data_path, train=False, download=True)
    
    # Create split indices 
    split_path = get_split_path(config=config)
    train_indices, val_indices = load_or_create_split(train_full_dataset, 
                                                      split_path, 
                                                      val_ratio=data_config['val_ratio'], 
                                                      seed=seed)
    # Compute mean and standard deviation for normalization (training only)
    mean, std = compute_mean_std(train_full_dataset, train_indices)

    # Build and apply transform 
    train_transform = build_train_transform(config=config, mean=mean, std=std)
    val_transform = build_val_transform(config=config, mean=mean, std=std)
    
    train_full_dataset.transform = train_transform
    val_full_dataset.transform = val_transform
    test_dataset.transform = val_transform
    
    # Create Dataloader 
    train_subset = Subset(train_full_dataset, train_indices)
    val_subset = Subset(val_full_dataset, val_indices)
    train_loader = DataLoader(dataset=train_subset,
                              batch_size=data_config['batch_size'],
                              shuffle=True, 
                              generator=make_generator(seed),
                              num_workers=data_config['num_workers'],
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(dataset=val_subset,
                            batch_size=data_config['batch_size'],
                            shuffle=False,      
                            num_workers=data_config['num_workers'],
                            worker_init_fn=seed_worker)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=data_config['batch_size'],
                            shuffle=False,
                            num_workers=data_config['num_workers'],
                            worker_init_fn=seed_worker)
    return train_loader, val_loader, test_loader, {'mean': mean, 'std': std}