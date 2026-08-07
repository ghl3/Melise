"""Verify the fla chunkwise KDA kernel against the reference scan.

Requires CUDA + flash-linear-attention; prints SKIP elsewhere. Run:

    .venv/bin/python tests/test_kda_kernel_equivalence.py

Compares, for several shapes:
  - forward outputs of KimiDeltaAttention with the kernel path vs the
    scan path (same weights, same input)
  - gradients w.r.t. the input and every parameter

The kernel computes the same recurrence in a chunk-parallel form with
different summation order (and internal precision), so we check against
a tolerance rather than bit equality.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from transformer.attention import kda as kda_mod
from transformer.attention.kda import KimiDeltaAttention


def run_case(B, S, d_model, n_heads, device):
    torch.manual_seed(0)
    m = KimiDeltaAttention(d_model, n_heads, decay_rank=16).to(device)
    x = torch.randn(B, S, d_model, device=device, requires_grad=True)

    def fwd_bwd(force_scan: bool):
        os.environ["KDA_FORCE_SCAN"] = "1" if force_scan else "0"
        m.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        out = m(x)
        out.square().mean().backward()
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
        return out.detach().clone(), x.grad.detach().clone(), grads

    out_k, xg_k, gr_k = fwd_bwd(force_scan=False)
    out_s, xg_s, gr_s = fwd_bwd(force_scan=True)
    os.environ.pop("KDA_FORCE_SCAN", None)

    def rel(a, b):
        return ((a - b).abs().max() / (b.abs().max() + 1e-8)).item()

    worst = max(
        [rel(out_k, out_s), rel(xg_k, xg_s)]
        + [rel(gr_k[n], gr_s[n]) for n in gr_k]
    )
    print(f"  B={B} S={S} d={d_model} H={n_heads}: worst relative diff {worst:.2e}")
    return worst


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0
    if kda_mod.chunk_kda is None:
        print("SKIP: flash-linear-attention not installed")
        return 0

    device = torch.device("cuda")
    tol = 5e-3
    worst = max(
        run_case(2, 64, 64, 2, device),
        run_case(2, 256, 128, 4, device),   # multi-chunk sequence
        run_case(1, 100, 64, 2, device),    # length not a chunk multiple
    )
    if worst > tol:
        print(f"FAIL: worst relative diff {worst:.2e} > tol {tol:.0e}")
        return 1
    print(f"PASS: kernel matches scan within {tol:.0e} (worst {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
