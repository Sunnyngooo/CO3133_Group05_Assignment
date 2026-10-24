import torch.nn as nn

def count_parameters(model: nn.Module):
    """Compute total, trainable and non-trainable parameters"""
    if not isinstance(model, nn.Module):
        raise TypeError("Model must be nn.Module object.")
    
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable, 
        "non_trainable_params": non_trainable, 
    }
if __name__ == "__main__":
    layer = nn.Linear(784, 10)
    layer.bias.requires_grad=False 
    models =[nn.Linear(784, 10), layer, nn.Linear(20, 5, bias=False)]
    result = [count_parameters(model) for model in models]
    print(result)