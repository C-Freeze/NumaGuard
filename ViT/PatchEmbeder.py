import torch
from torch import nn

class PatchEmbeder(nn.Module):
    def __init__(self, patches_pixel_count, embedding_size, num_of_patches, position_encodings):
        super().__init__()

        self.embedding_layer = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(patches_pixel_count, 32),
            nn.Linear(32, 64),
            nn.Linear(64,embedding_size)
        )

        self.embedding_size = embedding_size
        self.num_of_patches = num_of_patches
        self.position_encodings = position_encodings

        self.cls_token = nn.Parameter(torch.zeros((1, self.embedding_size)))
    
    def forward(self, inputs):
        outer_n_input = None
        
        for batch in inputs:

            n_input = None

            for idx in range(len(batch[0])):
                x = batch[:, idx, :, :]
                x = self.embedding_layer(x)
                x = torch.unsqueeze(x, 0)

                if n_input is None:
                    n_input = x
                
                else:
                    n_input = torch.cat((n_input, x))

            n_input = torch.cat((self.cls_token, n_input))

            patches = torch.add(self.position_encodings, n_input)
            patches = torch.unsqueeze(patches, 0)


            if outer_n_input is None:
                outer_n_input = patches
            
            else:
                outer_n_input = torch.cat((outer_n_input, patches))

        return outer_n_input