import torch
import torch.nn as nn
from assignment1.src.models.linear import LinearClassifier

def build_model(config: dict) -> nn.Module:
    model_cfg = config['model']
    model_name = model_cfg['name']
    
    if model_name == 'linear':
        input_dim = model_cfg['input_dim']
        output_dim = model_cfg['num_classes']
        return LinearClassifier(input_dim=input_dim, output_dim=output_dim)
    
    raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    config = {
        "model": {
            "name": "linear",
            "input_dim": 784,
            "num_classes": 10
        }
    }
    
    model = build_model(config)
    assert isinstance(model, LinearClassifier)
    
    x = torch.rand(8,1,28,28)
    logits = model(x)
    assert logits.shape == (8,10)
    print("Passed")