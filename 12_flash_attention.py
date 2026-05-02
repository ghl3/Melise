"""
Piece 10: FlashAttention via PyTorch's `scaled_dot_product_attention` (SDPA).

THE PROBLEM with our manual attention.

  scores = Q @ K^T / sqrt(Dh)           # shape (B, H, S, S)  — DRAM write
  scores = scores.masked_fill(...)      # DRAM read + DRAM write
  weights = softmax(scores)             # DRAM read + DRAM write
  out = weights @ V                     # DRAM read + DRAM write

The (B, H, S, S) attention matrix is the dominant intermediate tensor at
long context. For S=4k that's 256 MB at fp32; at S=32k it's 16 GB. Each
of our four operations reads or writes this tensor through DRAM, so we
pay 4× the bandwidth cost of just the math — and at S~16k+, this
intermediate doesn't fit in DRAM at all.

FLASHATTENTION (Dao et al., 2022). Restructure the math so the
attention matrix never materializes. Tile Q, K, V; compute attention
tile-by-tile; use online softmax to combine tiles into the running
output. The (B, H, S, S) tensor exists only ephemerally on-chip, never
gets written to DRAM, never has to fit in DRAM.

  Naive:           memory ~ S²    bandwidth dominated by S² traffic
  FlashAttention:  memory ~ S     bandwidth dominated by Q, K, V traffic

Same math. Same compute count. Vastly less DRAM traffic.

WE DON'T WRITE IT OURSELVES. FlashAttention is hand-tuned CUDA (and
analogous Metal/Triton variants on other backends). Instead, PyTorch's
`F.scaled_dot_product_attention` is a frontend that dispatches to the
best available kernel — FlashAttention on CUDA, MPS-fused attention on
Apple Silicon, a math fallback elsewhere. This file shows:
  - the equivalence: SDPA produces the same output as our manual code
  - the speedup: wall-clock comparison at various sequence lengths
  - the memory math that makes long-context inference feasible
"""

import time

import torch
import torch.nn.functional as F


def time_op(fn, n_warmup: int = 5, n_iters: int = 20) -> float:
    """Run fn() repeatedly and return the average per-call wall-clock time."""
    for _ in range(n_warmup):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / n_iters


# --- Three implementations of the same attention math ---

def manual_attention(q, k, v, bool_mask):
    """Hand-written: full (B, H, S, S) scores tensor in DRAM."""
    Dh = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
    scores = scores.masked_fill(~bool_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def sdpa_with_mask(q, k, v, bool_mask):
    """PyTorch's fused attention kernel with an explicit additive mask."""
    # SDPA wants an additive mask: 0 = allowed, -inf = forbidden.
    additive_mask = torch.where(bool_mask, 0.0, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask)


def sdpa_causal(q, k, v):
    """SDPA's fastest path: pass `is_causal=True` and skip the mask tensor entirely.
    The kernel knows to ignore the upper triangle without ever materializing a mask.
    Only valid when Q's positions and K's positions have the same length and are aligned.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# --- Equivalence check ---

torch.manual_seed(0)
device = torch.device("mps")
B, H, S, Dh = 1, 8, 512, 64

q = torch.randn(B, H, S, Dh, device=device)
k = torch.randn(B, H, S, Dh, device=device)
v = torch.randn(B, H, S, Dh, device=device)
bool_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))

with torch.no_grad():
    y_manual = manual_attention(q, k, v, bool_mask)
    y_sdpa_m = sdpa_with_mask(q, k, v, bool_mask)
    y_sdpa_c = sdpa_causal(q, k, v)

print("equivalence (random Q, K, V):")
print(f"  max |manual - sdpa(mask)|   : {(y_manual - y_sdpa_m).abs().max().item():.2e}")
print(f"  max |manual - sdpa(is_causal)|: {(y_manual - y_sdpa_c).abs().max().item():.2e}")

# --- Wall-clock comparison across sequence lengths ---

print("\nwall-clock per forward pass (B=1, H=8, Dh=64):")
print(f"  {'S':>6}  {'manual':>10}  {'sdpa(mask)':>12}  {'sdpa(is_causal)':>16}  {'speedup vs manual':>18}")
for S in [128, 512, 1024, 2048, 4096]:
    q = torch.randn(B, H, S, Dh, device=device)
    k = torch.randn(B, H, S, Dh, device=device)
    v = torch.randn(B, H, S, Dh, device=device)
    bool_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))

    t_manual = time_op(lambda: manual_attention(q, k, v, bool_mask))
    t_sdpa_m = time_op(lambda: sdpa_with_mask(q, k, v, bool_mask))
    t_sdpa_c = time_op(lambda: sdpa_causal(q, k, v))

    speedup = t_manual / t_sdpa_c
    print(f"  {S:>6}  {t_manual * 1000:>8.2f} ms  {t_sdpa_m * 1000:>10.2f} ms  {t_sdpa_c * 1000:>14.2f} ms  {speedup:>16.2f}x")

# --- The asymptotic memory savings ---

print("\nthe attention scores tensor (B, H, S, S) at fp32:")
print(f"  {'S':>8}  {'naive memory':>20}  {'FlashAttention memory':>25}")
for S in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    naive_bytes = B * H * S * S * 4
    if naive_bytes >= 1e9:
        sz = f"{naive_bytes / 1e9:.1f} GB"
    elif naive_bytes >= 1e6:
        sz = f"{naive_bytes / 1e6:.1f} MB"
    else:
        sz = f"{naive_bytes / 1024:.1f} KB"
    print(f"  {S:>8}  {sz:>20}  {'never materialized':>25}")
