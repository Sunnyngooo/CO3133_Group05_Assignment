import random
import numpy as np 
import torch 

from src.utils.seed import set_seed, make_generator

def sample_random_values():
    return {
        "python": random.random(),
        "numpy": np.random.rand(),
        "torch": torch.rand(3),
    }
    
set_seed(42)
first = sample_random_values()
set_seed(42)
second = sample_random_values()

assert first["python"] == second['python']
assert first["numpy"] == second['numpy']
assert torch.equal(first["torch"], second['torch'])

g1 = make_generator(42)
g2 = make_generator(42)

assert torch.equal(torch.randperm(100, generator=g1), torch.randperm(100, generator=g2))

print("All seed producibility tests passed.")