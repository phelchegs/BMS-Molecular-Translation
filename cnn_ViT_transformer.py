import torch
from torch import nn
from ViT_encoder_cnn import Encoder
from Transformer_decoder import Decoder, subsequent_mask
from ViT_embedding import Embedding
from Transformer_embedding import TGT_Embedding
from Generator import Generator

class Encoderdecoder(nn.Module):
    def __init__(self, encoder, decoder, tgt_embedding, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.generator = generator
        
    def encode(self, image):
        return self.encoder(image)
    
    def decode(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(src, self.tgt_embedding(tgt), src_mask, tgt_mask)
    
    def forward(self, image, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(image), tgt, src_mask, tgt_mask)
    
    
def ViT_transformer(num_channels, pretrained, src_dim, token_dim, decoder_layers, decoder_heads, decoder_one_head_dim, decoder_ff_dim, decoder_dropout, vocab_length, tgt_embedding_dropout, tgt_max):
    
    encoder = Encoder(num_channels = num_channels, pretrained = pretrained)
    decoder = Decoder(src_dim, token_dim, decoder_layers, decoder_heads, decoder_one_head_dim, decoder_ff_dim, decoder_dropout)
    # src_embedding = Embedding(image_size, patch_size, patch_dim, channels, image_embedding_dropout)
    tgt_embedding = TGT_Embedding(token_dim, vocab_length, tgt_embedding_dropout, tgt_max)
    generator = Generator(token_dim, vocab_length)
    model = Encoderdecoder(encoder, decoder, tgt_embedding, generator)
    
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    for p in tgt_embedding.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    for p in generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model