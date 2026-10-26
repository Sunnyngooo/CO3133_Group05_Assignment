import torch 
import torch.nn as nn
from assignment1.src.metrics.model_stats import count_parameters

class LSTMClassifier(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, output_dim=10, dropout=0.2)    :
        super().__init__()
        
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in range [0, 1).")
                
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.classifiers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )
    def forward(self, x):
        x = x.squeeze(1) # convert to row sequence by discarding channel dimension 
        _, (h_n, _)  = self.lstm(x)
        return self.classifiers(h_n[-1]) # take the hidden state of the last layer