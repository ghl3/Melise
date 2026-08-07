"""RMSNorm."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Per token, divide each D-vector by its RMS, then multiply by a learned
    per-dim scale. No mean subtraction, no bias. Standard in modern
    transformers (Llama, DeepSeek, Mistral, Qwen, Gemma, Kimi).

    Computes the normalization in fp32 for numerical stability when the
    model dtype is bf16, then casts back to the input dtype.

    Also used with a non-D shape: KDA applies it head-wise over each
    head's value vector (weight shape (d_v,), broadcast across heads).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (self.weight * (x32 / rms)).to(in_dtype)
