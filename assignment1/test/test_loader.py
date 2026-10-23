import torch
from pathlib import Path
from assignment1.src.utils.config import load_config
from assignment1.src.data.dataset import build_dataloaders

def main(model_name):   
    ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
    config = load_config(model_name)
    train_loader, val_loader, test_loader, stats = build_dataloaders(config)
    mean, std = stats
    
    # check batch size and data type       
    imgs, labels = next(iter(train_loader))
    batch_size = config['data']['batch_size']
     
    assert imgs.shape == (batch_size, 1, 28, 28)
    assert labels.shape == (batch_size,)
    assert imgs.dtype == torch.float32
    assert labels.dtype == torch.int64
    
    # Check label 
    assert labels.min().item() >= 0
    assert labels.max().item() < 10 
    
    # Check loader size with val_ratio = 0.15
    assert len(train_loader.dataset) == 51000
    assert len(val_loader.dataset) == 9000
    assert len(test_loader.dataset) == 10000
    print("All tests passed !")
    
if __name__ == "__main__": 
    main("cnn")
