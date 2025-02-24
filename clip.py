import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_embed: int, seq_length: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_embed)

        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, d_embed))

    def forward(self, x):
        # x: (batch_size, seq_length)

        # (batch_size, seq_length) -> (batch_size, seq_length, d_embed)
        x = self.token_embedding(x)

        x += self.pos_embedding

        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_heads:int, d_embed: int):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_embed)

        self.attention = SelfAttention(n_heads, d_embed)

        self.layer_norm_2 = nn.LayerNorm(d_embed)

        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)

        self.linear_2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_length, d_embed)

        residue = x

        ## self-attention layer 
        x = self.layer_norm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residue

        residue = x
        
        ## feed-forward layer
        x = self.layer_norm_2(x)

        # (batch_size, seq_length, d_embed) -> (batch_size, seq_length, 4 * d_embed)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGELU

        # (batch_size, seq_length, 4 * d_embed) -> (batch_size, seq_length, d_embed)
        x = self.linear_2(x)

        x += residue

        return x

class CLIP(nn.Module):

    def __init__(self, vocab_size: int, d_embed: int, seq_length: int):
        super().__init__()

        self.embedding = CLIPEmbedding(vocab_size, d_embed, seq_length)

        self.layers = nn.ModuleList([
            CLIPLayer(12, d_embed) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(d_embed)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # x: (batch_size, seq_length) (tokens)

        x = x.type(torch.long)

        # (batch_size, seq_length) -> (batch_size, seq_length, d_embed)
        state = self.embedding(x)

        for layer in self.layers:
            state = layer(state)

        state = self.layer_norm(state)

        return state

