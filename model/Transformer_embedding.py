import torch
from torch import nn
import math

class TGT_Embedding(nn.Module):
    def __init__(self, token_dim, vocab_dim, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, token_dim)
        self.token_dim = token_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, token_dim)
        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, token_dim, 2)*-(math.log(10000)/token_dim))
        pe[:, 0::2] = torch.sin(positions*div_term)
        pe[:, 1::2] = torch.cos(positions*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def forward(self, tgt):
        x = self.embedding(tgt)*math.sqrt(self.token_dim)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        x = self.dropout(x)
        return x
    
