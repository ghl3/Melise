"""
Piece 11: Quantization.

THE BANDWIDTH PROBLEM, RESTATED. Decode is bandwidth-bound: each step
streams every weight matrix and the full KV cache from DRAM to compute
one output token. The compute units wait while bytes move.

THE FIX. Store numbers in fewer bits.
  - fp16 → int8:  half the bytes  → up to 2× faster decode
  - fp16 → int4:  quarter bytes   → up to 4× faster decode
  - fp16 → fp8:   half the bytes, more dynamic range than int8

Same model, smaller memory footprint, less DRAM bandwidth per step. On
hardware with matching low-bit matmul kernels (H100 fp8, modern int8),
you save *compute* too — but at decode the bandwidth win is the bigger
deal because compute was idle anyway.

THE COST. Quantization is lossy. Each fp16 weight gets rounded to one of
2^bits values. The model's outputs shift slightly. With careful schemes
(per-channel or per-group scales, calibration, GPTQ/AWQ), the quality
loss is small — often unmeasurable on benchmarks. With naive schemes
(per-tensor scale, no calibration), it can be significant.

This file:
  1. Implements simple weight-only int8 quantization from scratch.
  2. Quantizes every linear in a small model and runs inference.
  3. Compares output to the fp32 reference and measures memory savings.
  4. Discusses what production uses (int4, group-wise, calibrated).
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------- Quantization core: int8 with per-output-channel symmetric scales -------

def quantize_int8_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8 quantization.

    For weight matrices stored as (out_features, in_features), one scale
    per *row* (output channel). Symmetric means the int range is [-127, 127]
    and zero stays at zero — no zero-point offset needed.

    Returns (q_weight: int8, scale: float32 of shape (out_features,)).
    """
    abs_max = w.abs().amax(dim=-1, keepdim=True)              # (out, 1)
    scale = abs_max.clamp(min=1e-8) / 127.0                    # avoid divide-by-zero
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1)


def dequantize_int8_per_channel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct fp32 weight from int8 storage."""
    return q.float() * scale.unsqueeze(-1)


class Int8Linear(nn.Module):
    """A drop-in replacement for nn.Linear (bias=False) with int8 weights.

    Weights stored as int8 + per-row scale. At forward time we dequantize
    on the fly and do an fp32 matmul. On hardware with int8 matmul (NVIDIA
    Tensor Cores, Apple AMX in some paths), production code does the matmul
    in int8 directly — bigger speedup, same memory savings. We do the
    simple version: storage is int8, compute is fp32.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("q_weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.zeros(out_features))

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        # Match the source's device — the quantized module's freshly-registered
        # buffers default to CPU otherwise.
        layer = cls(linear.in_features, linear.out_features).to(linear.weight.device)
        q, scale = quantize_int8_per_channel(linear.weight.data)
        layer.q_weight.copy_(q)
        layer.scale.copy_(scale)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on the fly. In production this is fused with the matmul.
        w = self.q_weight.float() * self.scale.unsqueeze(-1)
        return x @ w.T


def quantize_model_(model: nn.Module) -> int:
    """Replace every nn.Linear in `model` with Int8Linear in place. Returns
    the number of layers replaced."""
    count = 0
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and child.bias is None:
            setattr(model, name, Int8Linear.from_linear(child))
            count += 1
        else:
            count += quantize_model_(child)
    return count


# ------- A minimal model to quantize (no positional encoding for clarity) -------

@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
    def forward(self, x):
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, H, Dh).transpose(1, 2)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        scores = scores.masked_fill(~self.mask[:S, :S], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, h, bias=False)
        self.down = nn.Linear(h, cfg.d_model, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
    def forward(self, x):
        a = self.attn(self.norm1(x))
        m = self.mlp(self.norm2(x + a))
        return x + a + m


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    def forward(self, ids):
        x = self.token_embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.final_norm(x))


# ------- Demo -------

torch.manual_seed(0)
device = torch.device("mps")
cfg = Config()

# Reference: full-precision fp32 model.
model_fp = TransformerLM(cfg).to(device).eval()

# Quantized: deep-copy and replace every Linear with Int8Linear.
import copy
model_q = copy.deepcopy(model_fp)
n_replaced = quantize_model_(model_q)
print(f"replaced {n_replaced} Linear layers with Int8Linear")

# Run both on the same input.
prompt = b"hello world"
ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
with torch.no_grad():
    y_fp = model_fp(ids)
    y_q = model_q(ids)

# Output difference.
diff_max = (y_fp - y_q).abs().max().item()
diff_mean = (y_fp - y_q).abs().mean().item()
print(f"\noutput agreement on prompt {prompt!r}:")
print(f"  max  |fp32 - int8|  = {diff_max:.4e}")
print(f"  mean |fp32 - int8|  = {diff_mean:.4e}")
print(f"  fp32 logit range:    [{y_fp.min().item():+.3f}, {y_fp.max().item():+.3f}]")

# Greedy next-token agreement.
next_fp = y_fp[0, -1].argmax().item()
next_q = y_q[0, -1].argmax().item()
print(f"  greedy next byte:   fp32={next_fp}  int8={next_q}  {'✓ same' if next_fp == next_q else '✗ different'}")

# ------- Memory comparison -------

def model_weight_bytes(model):
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.numel() * module.weight.element_size()
        elif isinstance(module, Int8Linear):
            total += module.q_weight.numel() * module.q_weight.element_size()
            total += module.scale.numel() * module.scale.element_size()
        elif isinstance(module, nn.Embedding):
            total += module.weight.numel() * module.weight.element_size()
    return total

bytes_fp = model_weight_bytes(model_fp)
bytes_q = model_weight_bytes(model_q)
print(f"\nweight + embedding storage:")
print(f"  fp32:  {bytes_fp / 1024:>8.1f} KB")
print(f"  int8:  {bytes_q / 1024:>8.1f} KB")
print(f"  ratio: {bytes_fp / bytes_q:.2f}× smaller")

# Per-Linear bytes-saved breakdown.
print(f"\nper-Linear bytes saved (linear weights only, embeddings unchanged):")
fp_lin = sum(p.numel() * 4 for n, p in model_fp.named_parameters() if "_proj" in n or n.endswith(".up.weight") or n.endswith(".down.weight") or n == "lm_head.weight")
q_lin = sum(buf.numel() for n, buf in model_q.named_buffers() if "q_weight" in n)
q_scales = sum(buf.numel() * 4 for n, buf in model_q.named_buffers() if "scale" in n and "q_weight" not in n)
print(f"  fp32 linear weights:   {fp_lin / 1024:>7.1f} KB")
print(f"  int8 linear weights:   {q_lin / 1024:>7.1f} KB  ({fp_lin // q_lin}× smaller)")
print(f"  scale tables overhead: {q_scales / 1024:>7.1f} KB  (negligible)")
