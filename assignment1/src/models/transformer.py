import torch 
import torch.nn as nn

class TransformerClassifier(nn.Module): 
    def __init__(self, input_channels, image_size, patch_size, embed_dim, num_heads, num_layers, 
                 ff_dim, num_classes=10, dropout=0.2, pooling="mean"):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        
        if num_layers < 1: 
            raise ValueError("Number of layers must be larger than 1")
        
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in range [0, 1).")
        
        if pooling != 'mean':
            raise ValueError(f"Pooling {pooling} is not supported.")
        
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        
        num_patches = (image_size // patch_size) ** 2
        
        # Patch Embedding 
        self.patch_embed = nn.Conv2d(in_channels=input_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # Transformer Encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=ff_dim, 
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, enable_nested_tensor=False)        
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        output = self.patch_embed(x)
        output = output.flatten(2)
        output = output.transpose(1,2)
        output = output + self.pos_embed
        output = self.encoder(output)
        output = output.mean(dim=1) 
        return self.classifier(output)