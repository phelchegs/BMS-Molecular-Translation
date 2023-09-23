import torch
from torch import nn
from torch.nn.functional import log_softmax

class Generator(nn.Module):
    def __init__(self, token_dim, vocab_dim):
        super().__init__()
        self.proj = nn.Linear(token_dim, vocab_dim)
        
    def forward(self, x):
        return log_softmax(self.proj(x), dim = -1)