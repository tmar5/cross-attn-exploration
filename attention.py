import torch
from torch import nn
from torch.nn import functional as F
import math
import os
import numpy as np
import threading

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, w_qkv_bias=True, w_o_bias=True):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.w_qkv = nn.Linear(d_embed, 3 * d_embed, bias=w_qkv_bias)
        self.w_o = nn.Linear(d_embed, d_embed, bias=w_o_bias)

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (batch_size, seq_length (h*w), d_embed)

        n, seq_length, d_embed = x.shape

        heads_shape =  n, seq_length, self.n_heads, self.d_head

        q, k, v = self.w_qkv(x).chunk(3, dim=-1)

        
        # (batch_size, seq_length, d_embed) -> (batch_size, seq_length, n_heads, d_head) -> (batch_size, n_heads, seq_length, d_head)
        q = q.view(heads_shape).transpose(1, 2)
        k = k.view(heads_shape).transpose(1, 2)
        v = v.view(heads_shape).transpose(1, 2)

        # (batch_size, n_heads, seq_length, seq_length)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Create a mask that is a triangular superior matrix (with 0 on diagonal)
            mask = torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            # put the future values to -inf to make it causal
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= self.d_head ** 0.5

        weight = F.softmax(weight, dim=-1)

        # (batch_size, n_heads, seq_length, d_head)
        attention = weight @ v

        # (batch_size, seq_length, n_heads, d_head)
        attention = attention.transpose(1, 2).contiguous()

        # (batch_size, seq_length, d_embed)
        attention = attention.view((n, seq_length, d_embed))

        # (batch_size, seq_length, d_embed)
        attention = self.w_o(attention)

        return attention


class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, w_qkv_bias=True, w_o_bias=True):
        super().__init__()


        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
        self.w_q = nn.Linear(d_embed, d_embed, bias=w_qkv_bias)
        self.w_k = nn.Linear(d_cross, d_embed, bias=w_qkv_bias)
        self.w_v = nn.Linear(d_cross, d_embed, bias=w_qkv_bias)

        self.w_o = nn.Linear(d_embed, d_embed, bias=w_o_bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor,  weight=None, word=None, coeff_attn=1, th=0.3) -> torch.Tensor:
        # x: (latent): (batch_size, seq_length_q, d_embed)
        # y: (context): (batch_size, seq_length_kv, d_cross) = (batch_size, 77, 768)

        n, seq_length, d_embed = x.shape
        heads_shape =  (n, -1, self.n_heads, self.d_head)

        q = self.w_q(x)
        k = self.w_k(y)
        v = self.w_v(y)

        # (batch_size, seq_length, d_embed) -> (batch_size, seq_length, n_heads, d_head) -> (batch_size, n_heads, seq_length, d_head)
        q = q.view(heads_shape).transpose(1, 2)
        k = k.view(heads_shape).transpose(1, 2)
        v = v.view(heads_shape).transpose(1, 2)
        
        if weight is None:
            # (batch_size, n_heads, seq_length_q, seq_length_kv)
            weight = q @ k.transpose(-1, -2)

            weight /= self.d_head ** 0.5
    
            weight = F.softmax(weight, dim=-1)
        elif word is not None:
            for j, w in enumerate(word):
                max_weight = weight[:,:,:,w].max()
                threshold = th * max_weight
                mask = weight[:,:,:,w] > threshold
                weight[:,:,:,w][mask] *= coeff_attn[j]
                # weight[:,:,:,w][~mask] *= 0
        # (batch_size, n_heads, seq_length, d_head)
        attention = weight @ v

        # (batch_size, seq_length, n_heads, d_head)
        attention = attention.transpose(1, 2).contiguous()

        # (batch_size, seq_length, d_embed)
        attention = attention.view((n, seq_length, d_embed))

        # (batch_size, seq_length, d_embed)
        attention = self.w_o(attention)

        return attention, weight
            
