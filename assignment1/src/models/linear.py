import torch 
import torch.nn as nn
from assignment1.src.metrics.model_stats import count_parameters

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        return self.linear(x)