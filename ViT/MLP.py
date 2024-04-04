import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, embedding_size, num_of_patches):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.LayerNorm((num_of_patches, embedding_size)),
            nn.Linear(embedding_size, 256),
            nn.LayerNorm((num_of_patches, 256)),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.LayerNorm((num_of_patches, 512)),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.LayerNorm((num_of_patches, 256)),
            nn.Linear(256, embedding_size),
            nn.ELU(),
            nn.LayerNorm((num_of_patches, embedding_size))
        )
    
    def forward(self, inputs):
        return self.linear(inputs)