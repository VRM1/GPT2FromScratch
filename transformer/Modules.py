''' Define the Modules '''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class WindowedScaledDotProductAttention(nn.Module):
    """
    Windowed Scaled Dot-Product Attention
    
    Limits attention to a window of fixed size around each token,
    reducing computational complexity from O(n²) to O(n·w) where
    w is the window size.
    """

    def __init__(self, temperature, window_size=256, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.window_size = window_size
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [batch_size, n_heads, seq_len, d_k]
        mask: [batch_size, 1, seq_len, seq_len] or [batch_size, n_heads, seq_len, seq_len]
        
        returns: [batch_size, n_heads, seq_len, d_v], attention weights
        """
        batch_size, n_heads, len_q, d_k = q.size()
        len_k = k.size(2)
        
        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature  # [batch, n_heads, len_q, len_k]
        
        # Apply windowed attention
        if len_k > self.window_size:
            # Create a windowed mask - each position can only attend to nearby positions
            # within the window_size (half on each side, or less at the edges)
            window_mask = torch.zeros_like(attn)
            
            # For each position, allow attention to window_size//2 positions on each side
            half_window = self.window_size // 2
            
            for i in range(len_q):
                # Calculate the valid window range
                start_idx = max(0, i - half_window)
                end_idx = min(len_k, i + half_window + 1)
                
                # Allow attention only within this window
                window_mask[:, :, i, start_idx:end_idx] = 1.0
            
            # Apply the window mask (set attention scores outside window to -inf)
            attn = attn.masked_fill(window_mask == 0, -1e9)
        
        # Apply original attention mask if provided (e.g., for padding or causal attention)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Apply attention weights to values
        output = torch.matmul(attn, v)
        
        return output, attn


class CausalWindowedAttention(nn.Module):
    """
    Causal Windowed Attention
    
    Combines causal masking (only attending to previous tokens) with
    windowed attention (limiting to a fixed window size) for efficient
    autoregressive modeling.
    """
    
    def __init__(self, temperature, window_size=256, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.window_size = window_size
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [batch_size, n_heads, seq_len, d_k]
        mask: [batch_size, 1, seq_len, seq_len] or [batch_size, n_heads, seq_len, seq_len]
        
        returns: [batch_size, n_heads, seq_len, d_v], attention weights
        """
        batch_size, n_heads, len_q, d_k = q.size()
        len_k = k.size(2)
        
        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature  # [batch, n_heads, len_q, len_k]
        
        # Create a causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(len_q, len_k, device=q.device)).unsqueeze(0).unsqueeze(0)
        
        # Apply windowed attention within the causal constraint
        if self.window_size < len_k:
            # Create a windowed version of the causal mask
            window_mask = torch.zeros_like(causal_mask)
            
            for i in range(len_q):
                # Calculate the start of the window (max of 0 or position - window_size)
                start_idx = max(0, i - self.window_size + 1)
                # End of window is the current position (due to causal constraint)
                end_idx = i + 1
                
                # Allow attention only within this window
                window_mask[:, :, i, start_idx:end_idx] = 1.0
            
            # Use the windowed causal mask
            causal_mask = window_mask
        
        # Apply causal and window masks
        attn = attn.masked_fill(causal_mask == 0, -1e9)
        
        # Apply original attention mask if provided (e.g., for padding)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Apply attention weights to values
        output = torch.matmul(attn, v)
        
        return output, attn
