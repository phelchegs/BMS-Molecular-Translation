import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

def pair(t):
    if isinstance(t, tuple):
        temp = t
    else:
        temp = (t, t)
    return temp

class Embedding(nn.Module):
    def __init__(self, image_size, patch_size, patch_embed_dim, channels = 3, emb_dropout = 0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        nb_patches = (image_height//patch_height)*(image_width//patch_width)
        assert image_height%patch_height == 0 and image_width%patch_width == 0, 'Image size not divisible by patch size'
        #embedding of patches, input must be (batch, channels, height, width), after embedding, it changes to (batch, number of patches, dimension of patches).
        self.patch_embedding = nn.Sequential(Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = patch_height, pw = patch_width), nn.Linear(patch_height*patch_width*channels, patch_embed_dim))
        
        #positional encoding, create pe as (number of patches, dimension of patches), finally change pe as (1, number of patches, dimension of patches)
        pe = torch.zeros(nb_patches, patch_embed_dim)
        positions = torch.arange(0, nb_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, patch_embed_dim, 2)*-(math.log(10000)/patch_embed_dim))
        pe[:, 0::2] = torch.sin(positions*div_term)
        pe[:, 1::2] = torch.cos(positions*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
        self.dropout = nn.Dropout(emb_dropout)
    
    def forward(self, images):
        x =  self.patch_embedding(images)
        
        #broadcast pe to add patch_embedding.
        x = x + self.pe.requires_grad_(False)
        x =  self.dropout(x)
        return x
