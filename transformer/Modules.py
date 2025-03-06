import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

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

        '''
        What does dim=-1 means? basically it is applying softmax along the last dimension. however is it row wise or column wise? Here is the xplanation from GPT:
        
        You have a tensor of shape `[256, 8, 32, 32]`. Let's break it down:

        - The last two dimensions of this tensor are `32 x 32`. So, when you're applying `F.softmax(attn, dim=-1)`, it will be applied **along the last dimension** of each `32 x 32` slice.

        In a `32 x 32` matrix, consider:
        - **Rows** are along dimension 2 (32 elements).
        - **Columns** are along dimension 3 (also 32 elements).

        ### Softmax in this case:
        When you apply `F.softmax(attn, dim=-1)`, it applies softmax along the **last dimension**, i.e., **across the columns** for each row. In other words, for each element in the `256 x 8 x 32` slice, softmax normalizes the values along each row of the `32` elements (across the last dimension).

        #### Example:
        For each slice of the shape `[32, 32]`, it will apply softmax across the `32` columns for each of the `32` rows.

        In conclusion: 
        - `dim=-1` means the softmax will be applied **across the columns**, treating each row independently.
        '''

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
