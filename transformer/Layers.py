''' Define the Layers for GPT-2 '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class DecoderLayer(nn.Module):
    ''' 
    GPT-2 style decoder layer
    
    Key differences from Transformer decoder:
    1. Pre-norm architecture: Layer norm is applied BEFORE each sub-layer, not after
    2. No encoder-decoder attention: GPT-2 is decoder-only with no encoder
    3. Explicit residual connections after each sub-layer
    4. Uses epsilon=1e-5 for layer norm (smaller than Transformer's 1e-6)
    5. Support for different attention mechanisms: full, windowed, or causal windowed
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, 
                 attention_type="full", window_size=256):
        super(DecoderLayer, self).__init__()
        
        # Layer norm before self-attention (Pre-norm architecture in GPT-2)
        # Note: epsilon value is 1e-5 in GPT-2 vs 1e-6 in original Transformer
        self.slf_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Self-attention layer (same as Transformer)
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, 
            dropout=dropout, 
            attention_type=attention_type,
            window_size=window_size
        )
        
        # Layer norm before feed-forward (Pre-norm architecture in GPT-2)
        self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Feed-forward network (same component as Transformer)
        # Note: Original GPT-2 uses GELU activation instead of ReLU
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, activation='gelu')
        
        # Removed encoder-decoder attention that was present in original Transformer

    def forward(self, dec_input, slf_attn_mask=None):
        '''
        GPT-2 forward pass
        
        Differences from Transformer:
        1. No encoder_output parameter - GPT-2 doesn't use encoder-decoder attention
        2. Layer norm before attention, not after (Pre-norm vs Post-norm)
        3. Explicit residual connections
        4. Returns only self-attention weights (no encoder-decoder attention)
        '''
        # Store input for residual connection
        residual = dec_input
        
        # Layer normalization before self-attention (Pre-norm)
        normalized = self.slf_attn_layer_norm(dec_input)
        
        # Self-attention
        attn_output, slf_attn = self.slf_attn(
            normalized, normalized, normalized, mask=slf_attn_mask)
        
        # First residual connection (explicit in GPT-2)
        dec_output = residual + attn_output
        
        # Store output for second residual connection
        residual = dec_output
        
        # Layer normalization before feed-forward (Pre-norm)
        normalized = self.ffn_layer_norm(dec_output)
        
        # Feed-forward network
        ffn_output = self.pos_ffn(normalized)
        
        # Second residual connection (explicit in GPT-2)
        dec_output = residual + ffn_output
        
        # Only return self-attention, no encoder-decoder attention as in original Transformer
        return dec_output, slf_attn


# For compatibility with original code, keep the original DecoderLayer as an alternative
class TransformerDecoderLayer(nn.Module):
    ''' Original Transformer decoder layer with three sub-layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
