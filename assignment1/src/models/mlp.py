import torch.nn as nn

class MLPClassifier(nn.Module): 
    def __init__(self, input_dim=784, hidden_dims=(256,128), output_dim=10, dropout=0.2):
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLP must have at least one hidden layer.")
        
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("Hidden dimensions must be positive.")
        
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in range [0, 1).")
        
        layers = [nn.Flatten(start_dim=1)]
        prev_dim = input_dim
         
        for hidden_dim in hidden_dims: 
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    