import torch
from torch import nn
from ViT.MLP import MLP

class Encoder(nn.Module):
    def __init__(self, embedding_size, num_of_patches, last_layer=False):
        super().__init__()

        self.norm_layer = nn.LayerNorm((num_of_patches, embedding_size))
        
        self.q_layer = nn.Linear(embedding_size, embedding_size) 
        self.k_layer = nn.Linear(embedding_size, embedding_size) 
        self.v_layer = nn.Linear(embedding_size, embedding_size) 

        self.attn_layer = nn.MultiheadAttention(embedding_size, 16)

        self.mlp_layer = MLP(embedding_size, num_of_patches)
        self.last_layer = last_layer
    
    def forward(self, inputs):

        x1 = self.norm_layer(inputs)
        
        q = self.q_layer(x1)
        k = self.k_layer(x1)
        v = self.v_layer(x1)

        x1, attn_weights = self.attn_layer(q, k, v)
        x1 = x1 + inputs
        x2 = self.mlp_layer(x1)
        x1 = x2 + x1

        if self.last_layer:
            return x1[:, 0, :]
    
        else: 
            return x1