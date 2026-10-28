import torch
import torch.nn as nn
from assignment1.src.models.linear import LinearClassifier
from assignment1.src.models.mlp import MLPClassifier
from assignment1.src.models.cnn import CNNClassifier
from assignment1.src.models.lstm import LSTMClassifier
from assignment1.src.models.transformer import TransformerClassifier

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
    
    if model_name == "lstm":
        return LSTMClassifier(
            input_size=model_cfg['input_size'],
            hidden_size=model_cfg['hidden_size'],
            num_layers=model_cfg['num_layers'],
            output_dim=model_cfg['num_classes'],
            dropout=model_cfg['dropout']
        )
    
    if model_name == "transformer":
        return TransformerClassifier(
            input_channels=model_cfg['input_channels'],
            image_size=model_cfg['image_size'],
            patch_size=model_cfg['patch_size'],
            embed_dim=model_cfg['embed_dim'],
            num_heads=model_cfg['num_heads'],
            num_layers=model_cfg['num_layers'],
            ff_dim=model_cfg['ff_dim'],
            num_classes=model_cfg['num_classes'],
            dropout=model_cfg['dropout'],
            pooling='mean'
        )
        
    raise ValueError(f"Unsupported model: {model_name}")