import random

import numpy as np
import torch

from assignment1.src.utils.seed import set_seed, make_generator


def test_seed_reproduces_python_numpy_and_torch():
    set_seed(42)
    first = random.random(), np.random.rand(), torch.rand(3)
    set_seed(42)
    second = random.random(), np.random.rand(), torch.rand(3)
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_generator_reproduces_shuffle_order():
    first = torch.randperm(100, generator=make_generator(42))
    second = torch.randperm(100, generator=make_generator(42))
    different = torch.randperm(100, generator=make_generator(43))
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
