from typing import TypeAlias

from torch import Tensor as Tn
from torch import nn

import causal_lm


TnBxL: TypeAlias = Tn
TnBxLxD: TypeAlias = Tn


class CausalTransformerNoPE(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, n_layers: int, vocab_size: int) -> None:
        super().__init__()
        _layers = []
        for _ in range(n_layers):
            _layers.append(causal_lm.nn.Attention(hidden_dim, n_heads))
            _layers.append(causal_lm.nn.GatedMLP(hidden_dim))
        self.layers = nn.ModuleList(_layers)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.embed_norm = causal_lm.nn.RMSNorm(hidden_dim)

    def forward(self, x: TnBxL, verbose: bool = False) -> tuple[TnBxLxD, list[Tn] | None, list[Tn] | None]:
        x = self.embed(x)
        x = self.embed_norm(x)
        all_hidden = []
        all_attn = []
        for layer in self.layers:
            x, attn_matrix = layer(x)
            if verbose:
                all_hidden.append(x.clone().detach())
                all_attn.append(attn_matrix.clone().detach())
        logits = self.head(x)
        return logits, all_hidden if verbose else None, all_attn if verbose else None
