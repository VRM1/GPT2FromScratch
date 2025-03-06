''' Define the sublayers for GPT-2 '''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    # This class is used in both Transformer and GPT-2 with no changes

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


def gelu(x):
    """
    GELU activation function used in GPT-2
    Implementation of the gelu activation function by Hendrycks et al. (2016)
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, activation='relu'):
        """
        Args:
            d_in: Input dimension
            d_hid: Hidden dimension
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
                        Original Transformer uses 'relu'
                        GPT-2 uses 'gelu'
        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        # Note: in original Transformer, layer_norm is applied at the end
        # In GPT-2's pre-norm architecture, layer_norm is applied before this function
        # So the layer_norm here actually doesn't run in GPT-2 implementation

        residual = x

        # Use GELU for GPT-2, ReLU for Transformer
        if self.activation == 'gelu':
            x = self.w_2(gelu(self.w_1(x)))
        else:  # default to relu
            x = self.w_2(F.relu(self.w_1(x)))
            
        x = self.dropout(x)
        x += residual

        # This layer norm is only used in the original Transformer (post-norm)
        # For GPT-2, this operation is moved to the DecoderLayer
        x = self.layer_norm(x)

        return x


class GPT2PositionwiseFeedForward(nn.Module):
    ''' 
    GPT-2 specific feed-forward implementation
    
    Key differences from Transformer:
    1. Uses GELU activation instead of ReLU
    2. No layer norm (handled in DecoderLayer)
    3. No residual connection (handled in DecoderLayer)
    '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply GELU activation (GPT-2 specific)
        x = self.w_2(gelu(self.w_1(x)))
        x = self.dropout(x)
        
        # No residual connection or layer norm here (handled in DecoderLayer)
        return x
