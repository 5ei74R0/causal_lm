from torch import Tensor as Tn
from torch import nn

from .norm import RMSNorm


class GatedMLP(nn.Module):
    def __init__(self, hidden_dim: Tn) -> None:
        super().__init__()

        # Employ dense projection w/o bias from PaLM
        # Use GLU: LLaMA, Gemma, etc.
        # PaLM (https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf)
        # Gemma (https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)
        # LLaMA (https://arxiv.org/pdf/2302.13971.pdf)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 8 // 3, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 8 // 3, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 8 // 3, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.pre_norm = RMSNorm(hidden_dim)

    def forward(self, x: Tn) -> tuple[Tn, None]:
        x = self.pre_norm(x)
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return x + down, None
