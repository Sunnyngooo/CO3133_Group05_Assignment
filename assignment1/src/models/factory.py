import torch
import torch.nn as nn
from assignment1.src.models.linear import LinearClassifier
from assignment1.src.models.mlp import MLPClassifier
from assignment1.src.models.cnn import CNNClassifier

def build_model(config: dict) -> nn.Module:
    model_cfg = config['model']
    model_name = model_cfg['name']
    
    if model_name == 'linear':
        input_dim = model_cfg['input_dim']
        output_dim = model_cfg['num_classes']
        return LinearClassifier(input_dim=input_dim, output_dim=output_dim)
    
    if model_name == 'mlp':
        input_dim = model_cfg['input_dim']
        hidden_dims = model_cfg['hidden_dims']
        output_dim = model_cfg['num_classes']
        dropout = model_cfg['dropout']
        return MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout=dropout)
    
    if model_name == "cnn":
        return CNNClassifier(
            input_channel=model_cfg['input_channel'],
            image_size=model_cfg['image_size'],
            channels=model_cfg['channels'],
            kernel_size=model_cfg['kernel_size'],
            hidden_dim=model_cfg['hidden_dim'], 
            output_dim=model_cfg['num_classes'], 
            dropout=model_cfg['dropout']
        )
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