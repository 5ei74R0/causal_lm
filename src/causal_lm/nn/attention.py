from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor as Tn
from torch import nn

from .norm import RMSNorm


TnBxLxD: TypeAlias = Tn
TnBxHx1xE: TypeAlias = Tn
TnBxHxLxL: TypeAlias = Tn


class KVCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('k', None)  # TnBxHxLxE
        self.register_buffer('v', None)  # TnBxHxLxE
        self.n_cached = 0

    def _allocate(self):
        assert self.k is not None
        self.k_cache = torch.cat([self.k_cache, torch.zeros_like(self.k)], dim=-2)
        self.v_cache = torch.cat([self.v_cache, torch.zeros_like(self.v)], dim=-2)

    def add(self, k_t: TnBxHx1xE, v_t: TnBxHx1xE):
        if self.k is None:
            assert self.v is None
            self.k_cache = k_t
            self.v_cache = v_t
            return None
        if self.n_cached >= self.k.size()[-2]:
            self._allocate()
        self.k_cache[: ,: , self.n_cached+1, :] = k_t
        self.v_cache[: ,: , self.n_cached+1, :] = v_t


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.head_dim = hidden_dim // n_heads
        self.n_heads = n_heads
        self.hidden_dim = self.head_dim * self.n_heads

        # Employ Multi Query Attention (MQA) from PaLM & Gemma
        # Employ dense projection w/o bias from PaLM
        # PaLM (https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf)
        # Gemma (https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # For stable & fast convergence
        # QK Normalization (https://arxiv.org/abs/2302.05442)
        self.pre_norm = RMSNorm(self.hidden_dim)
        self.qk_norm = RMSNorm(self.head_dim)

        # KV Cache for fast inference
        self.kv_cache = KVCache()

    def _forward_with_cache(self, x: TnBxLxD, get_attn_map: bool = False) -> Tn:
        x = self.pre_norm(x)
        q: Tn = self.q_proj(x).view(*x.shape[:-1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k: Tn = self.kv_cache.k
        v: Tn = self.kv_cache.v
        q = self.qk_norm(q)
        q = self.qk_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_map: TnBxHxLxL | None = None if not get_attn_map else q @ k.transpose(-2, -1)
        return x + out, attn_map

    def forward(self, x: TnBxLxD, get_attn_map: bool = False) -> tuple[TnBxLxD, TnBxHxLxL | None]:
        x = self.pre_norm(x)
        # MQA
        q: Tn = self.q_proj(x).view(*x.shape[:-1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k: Tn = self.k_proj(x.view(*x.shape[:-1], self.n_heads, self.head_dim)).permute(0, 2, 1, 3)
        v: Tn = self.v_proj(x.view(*x.shape[:-1], self.n_heads, self.head_dim)).permute(0, 2, 1, 3)
        q = self.qk_norm(q)
        k = self.qk_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.permute(0, 2, 1, 3).contiguous().view(*x.shape)
        attn_map: TnBxHxLxL | None = None if not get_attn_map else q @ k.transpose(-2, -1)
        return x + out, attn_map
