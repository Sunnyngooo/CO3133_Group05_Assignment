import random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    
def seed_worker(worker_id: int) -> None:
    """Seed inside the Dataloader's workers.""" 
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def make_generator(seed: int) -> torch.Generator:
    """Create an reproducible generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator    