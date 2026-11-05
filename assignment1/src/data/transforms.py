import torch
from torchvision.transforms import v2
def build_train_transform(config, mean, std): 
    """Build optional transformation for training data."""
    transformation = []
     
    aug_cfg = config['data']['augmentation']
    
    if aug_cfg['enabled']:
        transformation.append(v2.RandomAffine(degrees=aug_cfg['rotation_degrees']))
    
    transformation.extend([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std])
    ])
    return v2.Compose(transformation)

def build_val_transform(config, mean, std): 
    """Build optional transformation for validation data."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std])
    ])