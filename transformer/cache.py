"""KV cache for autoregressive inference."""

import torch

from .config import Config


class KVCache:
    """Pre-allocated K and V buffers, mutated in place during decoding.

    Shape per side: (n_layers, B, n_kv_heads, max_seq_len, head_dim).
    Note that n_kv_heads (not n_heads) is used — the shrinkage is exactly
    the GQA cache reduction.

    Lifecycle:
        write(layer, k, v)   — append new K, V at the next free slot
        read(layer, length)  — return K, V slices for [0, length)
        advance(s)           — move the length pointer forward by s,
                               called once after all layers have written
    """

    def __init__(self, cfg: Config, batch_size: int, device: torch.device):
        self.cfg = cfg
        self.k = torch.zeros(
            cfg.n_layers, batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim,
            device=device, dtype=cfg.dtype,
        )
        self.v = torch.zeros_like(self.k)
        self.length = 0

    def write(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        s = new_k.shape[-2]
        self.k[layer_idx, :, :, self.length : self.length + s] = new_k
        self.v[layer_idx, :, :, self.length : self.length + s] = new_v

    def read(self, layer_idx: int, total_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k[layer_idx, :, :, :total_length],
            self.v[layer_idx, :, :, :total_length],
        )

    def advance(self, s: int) -> None:
        self.length += s
