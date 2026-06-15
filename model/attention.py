import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        keys = keys.reshape(batch_size, num_tokens, self.n_heads, self.head_dim)
        queries = queries.reshape(batch_size, num_tokens, self.n_heads, self.head_dim)
        values = values.reshape(batch_size, num_tokens, self.n_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = (queries @ keys.transpose(2, 3)) * self.scale
        mask = self.mask[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ values
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        return self.out_proj(context)
