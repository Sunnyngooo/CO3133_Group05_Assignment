import torch
from pathlib import Path

def save_checkpoint(path: Path, model, optimizer, epoch, val_metrics, config):
    # Checkpoint path
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path object.")
    path.parent.mkdir(parents=True, exist_ok=True)
        
    # Checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'config': config
    }
    # Save as files
    torch.save(checkpoint, path)
    
def load_checkpoint(path: Path, model, optimizer=None, device='cpu'):
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path object.")
    
    if not path.is_file(): 
        raise FileNotFoundError(f"Checkpoint in {path} not found.")
        
    checkpoint = torch.load(path, map_location=device, weights_only=True) # limit pytorch unpickling 
    
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    return checkpoint