import torch
from torch import nn
from ViT.PatchEmbeder import PatchEmbeder
from ViT.Encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, patch_pixel_count, embedding_size, num_of_patches, position_encodings):
        super().__init__()

        self.patch_encoder = PatchEmbeder(patch_pixel_count, embedding_size, num_of_patches, position_encodings)
        self.transformer_encoder_1 = Encoder(embedding_size, num_of_patches)
        self.transformer_encoder_2 = Encoder(embedding_size, num_of_patches)
        self.transformer_encoder_3 = Encoder(embedding_size, num_of_patches)
        self.transformer_encoder_4 = Encoder(embedding_size, num_of_patches)
        self.transformer_encoder_5 = Encoder(embedding_size, num_of_patches)
        self.transformer_encoder_6 = Encoder(embedding_size, num_of_patches, last_layer=True)
    
    def forward(self, x):

        
        x = self.patch_encoder(x)
        x = self.transformer_encoder_1(x)
        x = self.transformer_encoder_2(x)
        x = self.transformer_encoder_3(x)
        x = self.transformer_encoder_4(x)
        x = self.transformer_encoder_5(x)
        x = self.transformer_encoder_6(x)
        return x