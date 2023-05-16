import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_features : int, n_neurons : int, n_out : int) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(n_features, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_out)
        )
    
    
    def forward(self, x : torch.tensor) -> torch.tensor:
        return self.stack(x)

