import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Encoder(nn.Module):
    def __init__(self, patch_dim, nb_layers, nb_heads, one_head_dim, ff_dim, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(nb_layers):
            self.layers.append(nn.ModuleList([PreNorm(patch_dim, Attention(patch_dim = patch_dim, nb_heads = nb_heads, one_head_dim = one_head_dim, dropout = dropout)), 
                                              PreNorm(patch_dim, FeedForward(patch_dim = patch_dim, hidden_dim = ff_dim, dropout = dropout))]))
    def forward(self, x):
        for a, f in self.layers:
            x = a(x) + x
            x = f(x) + x
        return x
    
class Attention(nn.Module):
    def __init__(self, patch_dim, nb_heads, one_head_dim, dropout = 0.0):
        super().__init__()
        heads_dim = one_head_dim*nb_heads
        self.nb_heads = nb_heads
        self.scale = one_head_dim**-0.5
        self.qkv = nn.Linear(patch_dim, heads_dim*3, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)
        if not (nb_heads == 1 and patch_dim == one_head_dim):
            self.out = nn.Sequential(nn.Linear(heads_dim, patch_dim), nn.Dropout(dropout))
        else:
            self.out = nn.Identity()
    
    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'm s (h d) -> m h s d', h = self.nb_heads), qkv)
        qk = torch.matmul(q, k.transpose(-1, -2))*self.scale
        attn = self.softmax(qk)
        attn = self.dropout(attn)
        attn_v = torch.matmul(attn, v)
        attn_v = rearrange(attn_v, 'm h s d -> m s (h d)')
        output = self.out(attn_v)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, patch_dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(patch_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, patch_dim),
                                nn.Dropout(dropout)
                               )
    def forward(self, x):
        output = self.ff(x)
        return output

class PreNorm(nn.Module):
    def __init__(self, dim, function):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.function = function
    
    def forward(self, x, **kwargs):
        output = self.function(self.norm(x), **kwargs)
        return output