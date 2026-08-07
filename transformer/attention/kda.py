"""Kimi Delta Attention (KDA) — Kimi Linear / Kimi K3.

Linear attention: instead of comparing each query against all past keys
(O(T²)), each head maintains a fixed-size associative memory
S ∈ R^{d_k × d_v} that maps keys to values, updated once per token. The
update is the *delta rule* with a channel-wise forget gate (K3 Eq. 1):

    S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
    o_t = S_tᵀ q_t

Read it in three steps:
  1. Diag(α_t) S_{t−1} — decay: each of the d_k key channels forgets at
     its own input-dependent rate α ∈ (0,1). This per-channel (not
     per-head-scalar) gate is KDA's core refinement over Gated DeltaNet,
     and it is what carries positional information in the Kimi hybrid —
     no RoPE anywhere.
  2. (I − β k kᵀ) · — delta rule: partially erase what the memory
     currently returns for key k_t (β ∈ (0,1) is the write strength) ...
  3. + β k vᵀ — ... and write the new association k_t → v_t in its place.

Input parameterization (K3 Eq. 2): q/k/v each go through a depthwise
causal short convolution (kernel 4) then Swish; q and k are L2-normalized.
β is a per-head sigmoid. The decay logit z is a low-rank projection with
a per-channel bias.

Lower-bounded decay (K3 Eq. 5, new vs Kimi Linear): the per-step
log-decay is g = g_min · sigmoid(e^{A_h} · z) ∈ (g_min, 0), so
α = e^g never drops below e^{g_min} ≈ 6.7e-3 (g_min = −5). K3 introduced
this to keep cumulative-decay ratios inside the bf16 range so its
chunkwise kernel can run on dense tensor cores.

Output (K3 Eq. 6): head-wise RMSNorm on the memory readout, then a
full-rank sigmoid gate (input-dependent), then the output projection —
y = W_o[sigmoid(W_g x) ⊙ RMSNorm(o)].

This implementation runs the recurrence as an explicit sequential scan
over time, vectorized across batch and heads, in fp32. Production
implementations (FLA, Kimi's kernels) use a chunkwise-parallel form (K3
Eq. 3–4) that is mathematically equivalent but tensor-core friendly; the
scan is the honest reference version, so prefer modest seq_len when
training KDA layers.

The inference cache is O(1) in sequence length: the state S plus the last
kernel−1 conv inputs. No KV buffer, no position bookkeeping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm import RMSNorm
from .cache import KDALayerCache, ModelCache


class KimiDeltaAttention(nn.Module):
    """One KDA layer: short-conv q/k/v, channel-wise gated delta rule, gated output."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        layer_idx: int = 0,
        head_dim: int | None = None,
        conv_kernel: int = 4,
        decay_rank: int = 32,
        g_min: float = -5.0,
    ):
        super().__init__()
        if head_dim is None:
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            head_dim = d_model // n_heads
        H, Dk, Dv = n_heads, head_dim, head_dim
        self.n_heads, self.d_k, self.d_v = H, Dk, Dv
        self.conv_kernel = conv_kernel
        self.g_min = g_min
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(d_model, H * Dk, bias=False)
        self.k_proj = nn.Linear(d_model, H * Dk, bias=False)
        self.v_proj = nn.Linear(d_model, H * Dv, bias=False)

        # Depthwise causal short convolutions (kernel 4), one channel each,
        # à la Mamba: cheap local mixing before the recurrence.
        K = conv_kernel
        self.q_conv = nn.Conv1d(H * Dk, H * Dk, K, groups=H * Dk, bias=False)
        self.k_conv = nn.Conv1d(H * Dk, H * Dk, K, groups=H * Dk, bias=False)
        self.v_conv = nn.Conv1d(H * Dv, H * Dv, K, groups=H * Dv, bias=False)

        # Delta-rule write strength β: one sigmoid scalar per head.
        self.beta_proj = nn.Linear(d_model, H, bias=False)

        # Fine-grained decay logits z: low-rank projection (shared down, up to
        # one logit per key channel) plus a per-channel bias.
        self.a_down = nn.Linear(d_model, decay_rank, bias=False)
        self.a_up = nn.Linear(decay_rank, H * Dk, bias=False)
        # Bias init spreads retention timescales across channels, from fast
        # (α ≈ 0.1) to slow (α ≈ 0.99) — the same idea as Mamba's dt-bias init.
        self.a_bias = nn.Parameter(torch.empty(H * Dk).uniform_(-6.0, -0.5))
        # Per-head log-scale on the decay logit, A_h = 0 at init (K3 §2.1.1).
        self.a_log_scale = nn.Parameter(torch.zeros(H))

        # Output: head-wise RMSNorm (weight shared across heads), full-rank
        # sigmoid gate, output projection.
        self.out_norm = RMSNorm(Dv)
        self.gate_proj = nn.Linear(d_model, H * Dv, bias=False)
        self.o_proj = nn.Linear(H * Dv, d_model, bias=False)

    def make_cache(self, batch_size: int, device: torch.device) -> KDALayerCache:
        return KDALayerCache(
            batch_size, self.n_heads, self.d_k, self.d_v, self.conv_kernel,
            device, self.q_proj.weight.dtype,
        )

    # ---- pieces ----

    def _short_conv(
        self,
        conv: nn.Conv1d,
        x: torch.Tensor,
        tail: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal depthwise conv over time.

        x: (B, S, C). `tail` is the previous (kernel−1)-step window from the
        cache (None during training → zeros, i.e. left zero-padding).
        Returns (conv output (B, S, C), new tail (B, C, kernel−1)).
        """
        B, S, C = x.shape
        x = x.transpose(1, 2)                                     # (B, C, S)
        if tail is None:
            tail = x.new_zeros(B, C, self.conv_kernel - 1)
        full = torch.cat([tail, x], dim=-1)                       # (B, C, S + K − 1)
        out = conv(full)                                          # (B, C, S) — no padding
        new_tail = full[:, :, full.shape[-1] - (self.conv_kernel - 1) :]
        return out.transpose(1, 2), new_tail

    def _decay(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel retention factor α ∈ (e^{g_min}, 1), shape (B, S, H, Dk)."""
        B, S, _ = x.shape
        z = (self.a_up(self.a_down(x)) + self.a_bias).float()     # (B, S, H·Dk)
        z = z.view(B, S, self.n_heads, self.d_k)
        scale = self.a_log_scale.exp().view(1, 1, self.n_heads, 1)
        g = self.g_min * torch.sigmoid(scale * z)                 # (g_min, 0)
        return g.exp()

    @staticmethod
    def _scan(
        q: torch.Tensor,      # (B, H, S, Dk)  L2-normalized
        k: torch.Tensor,      # (B, H, S, Dk)  L2-normalized
        v: torch.Tensor,      # (B, H, S, Dv)
        alpha: torch.Tensor,  # (B, H, S, Dk)  per-channel decay
        beta: torch.Tensor,   # (B, H, S)      write strength
        state: torch.Tensor,  # (B, H, Dk, Dv) incoming memory
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential delta-rule scan. All inputs fp32. Returns (outputs, final state)."""
        outs = []
        S_mem = state
        for t in range(q.shape[2]):
            a_t = alpha[:, :, t].unsqueeze(-1)                    # (B, H, Dk, 1)
            k_t = k[:, :, t].unsqueeze(-1)                        # (B, H, Dk, 1)
            v_t = v[:, :, t].unsqueeze(-2)                        # (B, H, 1, Dv)
            b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)       # (B, H, 1, 1)

            S_mem = S_mem * a_t                                   # decay each key channel
            pred = (k_t * S_mem).sum(dim=-2, keepdim=True)        # k_tᵀ S — current readout
            S_mem = S_mem + b_t * k_t * (v_t - pred)              # erase old, write new
            q_t = q[:, :, t].unsqueeze(-1)                        # (B, H, Dk, 1)
            outs.append((q_t * S_mem).sum(dim=-2))                # S_tᵀ q_t → (B, H, Dv)
        return torch.stack(outs, dim=2), S_mem                    # (B, H, S, Dv)

    # ---- forward ----

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H, Dk, Dv = self.n_heads, self.d_k, self.d_v
        layer: KDALayerCache | None = (
            kv_cache.layers[self.layer_idx] if kv_cache is not None else None
        )

        # q/k/v: project → causal short conv → Swish; L2-normalize q and k so
        # the delta rule's erase step is a proper projection along k.
        q, tail_q = self._short_conv(self.q_conv, self.q_proj(x), layer.conv_q if layer else None)
        k, tail_k = self._short_conv(self.k_conv, self.k_proj(x), layer.conv_k if layer else None)
        v, tail_v = self._short_conv(self.v_conv, self.v_proj(x), layer.conv_v if layer else None)

        def heads(t: torch.Tensor, d: int) -> torch.Tensor:
            return t.view(B, S, H, d).transpose(1, 2).float()     # (B, H, S, d)

        q = F.normalize(heads(F.silu(q), Dk), dim=-1, eps=1e-6)
        k = F.normalize(heads(F.silu(k), Dk), dim=-1, eps=1e-6)
        v = heads(F.silu(v), Dv)

        alpha = self._decay(x).transpose(1, 2)                    # (B, H, S, Dk) fp32
        beta = torch.sigmoid(self.beta_proj(x).float()).transpose(1, 2)  # (B, H, S)

        state = (
            layer.state
            if layer is not None
            else torch.zeros(B, H, Dk, Dv, device=x.device, dtype=torch.float32)
        )
        out, final_state = self._scan(q, k, v, alpha, beta, state)
        if layer is not None:
            layer.state = final_state
            layer.conv_q, layer.conv_k, layer.conv_v = tail_q, tail_k, tail_v

        # Head-wise RMSNorm, input-dependent sigmoid gate, output projection.
        out = self.out_norm(out).to(x.dtype)                      # (B, H, S, Dv)
        out = out.transpose(1, 2).reshape(B, S, H * Dv)
        out = torch.sigmoid(self.gate_proj(x)) * out
        return self.o_proj(out)
