import torch
from torch import Tensor as Tn
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, hidden: Tn):
        in_dtype = hidden.dtype
        hidden = hidden.to(torch.float32)  # cast to f32 for numerical stability
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(in_dtype)
