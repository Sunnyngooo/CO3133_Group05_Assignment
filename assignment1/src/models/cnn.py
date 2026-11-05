import torch 
import torch.nn as nn

class CNNClassifier(nn.Module): 
    def __init__(self, input_channel, image_size, channels, kernel_size, hidden_dim, output_dim=10, dropout=0.2)    :
        super().__init__()
        if not channels:
            raise ValueError("CNN must have at least one convolution block layer.")
        
        if any(channel <= 0 for channel in channels):
            raise ValueError("All channel dimensions must be positive.")
        
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer.")
        
        if image_size // (2 ** len(channels)) < 1: 
            raise ValueError("Too many pooling layers for the input image.")
        
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in range [0, 1).")
             
        layers = []
        prev_channel = input_channel
        spatial_size = image_size
        padding = kernel_size // 2
        
        for channel in channels:
            layers.append(nn.Conv2d(prev_channel, channel, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            prev_channel = channel
            spatial_size = spatial_size // 2
        self.features = nn.Sequential(*layers)
        
        classifier_layers = [nn.Flatten(start_dim=1)]
        flatten_dim = prev_channel * spatial_size * spatial_size
        classifier_layers.extend([nn.Linear(flatten_dim, hidden_dim), nn.ReLU(), 
                       nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)])
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        output = self.features(x)
        return self.classifier(output)