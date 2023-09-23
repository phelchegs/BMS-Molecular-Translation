import torch
from torch import nn
from ViT_encoder import Encoder
from Transformer_decoder import Decoder, subsequent_mask
from ViT_embedding import Embedding
from Transformer_embedding import TGT_Embedding
from Generator import Generator

class Encoderdecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embedding, tgt_embedding, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.generator = generator
        
    def encode(self, image):
        return self.encoder(self.src_embedding(image))
    
    def decode(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(src, self.tgt_embedding(tgt), src_mask, tgt_mask)
    
    def forward(self, image, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(image), tgt, src_mask, tgt_mask)
    
    
def ViT_transformer(patch_dim, encoder_layers, encoder_heads, encoder_one_head_dim, encoder_ff_dim, encoder_dropout, token_dim, decoder_layers, decoder_heads, decoder_one_head_dim, decoder_ff_dim, decoder_dropout, image_size, patch_size, channels, image_embedding_dropout, vocab_length, tgt_embedding_dropout, tgt_max):
    
    encoder = Encoder(patch_dim, encoder_layers, encoder_heads, encoder_one_head_dim, encoder_ff_dim, encoder_dropout)
    decoder = Decoder(patch_dim, token_dim, decoder_layers, decoder_heads, decoder_one_head_dim, decoder_ff_dim, decoder_dropout)
    src_embedding = Embedding(image_size, patch_size, patch_dim, channels, image_embedding_dropout)
    tgt_embedding = TGT_Embedding(token_dim, vocab_length, tgt_embedding_dropout, tgt_max)
    generator = Generator(token_dim, vocab_length)
    model = Encoderdecoder(encoder, decoder, src_embedding, tgt_embedding, generator)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
    
    
