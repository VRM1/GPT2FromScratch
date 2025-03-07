''' Define the GPT-2 model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer.Layers import DecoderLayer


class GPT2Embedding(nn.Module):
    """
    GPT-2 embedding which is made up of token and positional embeddings.
    
    Differences from Transformer:
    1. Uses learned positional embeddings instead of sinusoidal
    2. No separate encoder/decoder embeddings
    """
    
    def __init__(self, vocab_size, max_position_embeddings, d_model, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings


class GPT2(nn.Module):
    """
    GPT-2 language model based on the "Language Models are Unsupervised Multitask Learners" paper.
    
    Key differences from Transformer:
    1. Decoder-only architecture (no encoder)
    2. Uses learned positional embeddings
    3. Uses Layer Normalization before each sub-layer (pre-norm)
    4. Adds a final Layer Normalization at the end
    5. Weight tying between input embedding and output projection
    6. Support for different attention mechanisms: full, windowed, or causal windowed
    """
    
    def __init__(
            self, vocab_size, pad_idx, 
            d_model=768, n_layers=12, n_head=12, d_k=64, d_v=64,
            d_inner=3072, dropout=0.1, max_position_embeddings=1024,
            attention_type="full", window_size=256):
        
        super().__init__()
        
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.attention_type = attention_type
        self.window_size = window_size
        
        # Embedding layer (token + position embeddings)
        self.embedding = GPT2Embedding(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            d_model=d_model,
            dropout=dropout
        )
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, 
                dropout=dropout,
                attention_type=attention_type,
                window_size=window_size
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (unique to GPT-2, not in original Transformer decoder)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying between token embeddings and final linear layer
        self.lm_head.weight = self.embedding.token_embeddings.weight
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights with normal distribution (GPT-2 specific)
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
                
    def get_subsequent_mask(self, seq):
        """For masking out the subsequent info - autoregressive attention mask."""
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask
        
    def forward(self, input_ids, return_attentions=False):
        # Create attention mask (same as in original Transformer decoder)
        slf_attn_mask = self.get_subsequent_mask(input_ids)
        
        # Add padding mask if needed
        if self.pad_idx is not None:
            pad_mask = (input_ids != self.pad_idx).unsqueeze(-2)
            slf_attn_mask = slf_attn_mask & pad_mask
        
        # Get embeddings
        dec_output = self.embedding(input_ids)
        
        all_attentions = [] if return_attentions else None
        
        # Process through decoder layers
        for layer in self.layers:
            # Note: DecoderLayer doesn't need enc_output in GPT-2
            dec_output, slf_attn = layer(dec_output, slf_attn_mask=slf_attn_mask)
            if return_attentions:
                all_attentions.append(slf_attn)
        
        # Final layer norm (not in original Transformer)
        dec_output = self.final_layer_norm(dec_output)
        
        # Language modeling head
        lm_logits = self.lm_head(dec_output)
        
        if return_attentions:
            return lm_logits, all_attentions
        
        return lm_logits
    
    def generate(self, input_ids, max_length, temperature=1.0, top_k=0, top_p=0.9):
        """
        Auto-regressive text generation using the model
        """
        batch_size = input_ids.shape[0]
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get model predictions for the next token
            with torch.no_grad():
                outputs = self.forward(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Add generated token to input
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        return input_ids
