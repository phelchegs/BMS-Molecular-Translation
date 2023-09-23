import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def subsequent_mask(dim):
    attn_shape = (1, dim, dim)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal = 1).type(torch.uint8)
    return subsequent_mask == 0

def attention(key, query, value, mask = None, dropout = None):
    dim = key.size(-1)
    scale = dim**-0.5
    query_key = torch.matmul(query, key.transpose(-2, -1))*scale
    if mask is not None:
        query_key = query_key.masked_fill(mask == 0, -1e9)
    attn = query_key.softmax(dim = -1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value)

class PreNorm(nn.Module):
    
    def __init__(self, dim, dropout, function):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.function = function
        
    def forward(self, x):
        output = x + self.dropout(self.function(self.norm(x)))
        return output

class Attention(nn.Module):
    
    def __init__(self, src_dim, token_dim, kqv_dim, nb_heads = 3, dropout = 0.0):
        super().__init__()
        self.nb_heads = nb_heads
        self.kv_linear = nn.Linear(src_dim, kqv_dim*nb_heads*2, bias = False)
        self.q_linear = nn.Linear(token_dim, kqv_dim*nb_heads, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)
        if not (nb_heads == 1 and token_dim == kqv_dim):
            self.out = nn.Sequential(nn.Linear(kqv_dim*nb_heads, token_dim), nn.Dropout(dropout))
        else:
            self.out = nn.Identity()
        
    def forward(self, src, tgt, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        q = self.q_linear(tgt)
        kv = self.kv_linear(src).chunk(2, dim = -1)
        k, v, q = map(lambda t: rearrange(t, 'm s (h d) -> m h s d', h = self.nb_heads), kv + tuple(q.unsqueeze(0)))
        attn_output = attention(k, q, v, mask = mask, dropout = self.dropout)
        attn_output = rearrange(attn_output, 'm h s d -> m s (h d)')
        attn_output = self.out(attn_output)
        return attn_output
    
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, input_dim)
                               )
    def forward(self, x):
        output = self.ff(x)
        return output

class Decoder(nn.Module):
    def __init__(self, src_dim, tgt_dim, nb_layers, nb_heads, one_head_dim, ff_dim, dropout = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(tgt_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(nb_layers):
            self.layers.append(nn.ModuleList([self.norm, 
                                              Attention(src_dim = tgt_dim, token_dim = tgt_dim, kqv_dim = one_head_dim, nb_heads = nb_heads, dropout = dropout), 
                                              self.dropout,
                                              self.norm,
                                              Attention(src_dim = src_dim, token_dim = tgt_dim, kqv_dim = one_head_dim, nb_heads = nb_heads, dropout = dropout),
                                              self.dropout,
                                              PreNorm(tgt_dim, dropout, FeedForward(input_dim = tgt_dim, hidden_dim = ff_dim, dropout = dropout))]))
    
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        for a, b, c, d, e, f, g in self.layers:
            tgt_norm = a(tgt)
            tgt_norm_attn = b(tgt_norm, tgt_norm, mask = tgt_mask)
            tgt_norm_attn = c(tgt_norm_attn) + tgt
            tgt_norm_attn_norm = d(tgt_norm_attn)
            cross_attn = e(src, tgt_norm_attn_norm, mask = src_mask)
            cross_attn = f(cross_attn) + tgt_norm_attn
            tgt = g(cross_attn)
        return tgt