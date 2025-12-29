"""GPT2 implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
import math

@dataclass
class GPTConfig:
    context_length: int = 1024    # max context / sequence length
    vocab_size: int = 50257    # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    num_layers: int = 12
    embd_size: int = 768    # embedding dim
    num_heads: int = 12


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embd_size % config.num_heads == 0

        self.c_attn = nn.Linear(config.embd_size, 3 * config.embd_size)
        self.c_proj = nn.Linear(config.embd_size, config.embd_size)
        self.num_heads = config.num_heads
        self.head_dim = config.embd_size // config.num_heads
        self.embed_dim = config.embd_size
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x) # (B, T, 3C)
        q, k, v = qkv.split(self.embed_dim, dim=-1) # (B, T, C) * 3
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, n_heads, T, head_dim)

        # attn = q @ k.transpose(-2, -1) # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        # attn = attn / math.sqrt(self.head_dim)
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # out = attn @ v # (B, n_heads, T, head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        out = self.c_proj(out) # (B, T, C)
        return out
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_size, 4 * config.embd_size)
        self.c_proj = nn.Linear(4 * config.embd_size, config.embd_size)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x
    
class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embd_size)
        self.ln2 = nn.LayerNorm(config.embd_size)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embd_size), # word embedding
            wpe=nn.Embedding(config.context_length, config.embd_size), # position embedding
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.num_layers)]), # transformer blocks
            ln_f=nn.LayerNorm(config.embd_size) # final layer norm
        ))
        self.lm_head = nn.Linear(config.embd_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos = pos.unsqueeze(0).expand_as(idx)
        pos_emb = self.transformer.wpe(pos)
        emb = self.transformer.wte(idx)
        x = emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    