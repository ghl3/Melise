"""Rotary Position Embedding (RoPE).

Encodes position by rotating Q and K vectors by an angle proportional to
their position. Pure functions, no learned parameters. Applied inside
attention to Q and K only — V is left unrotated.
"""

import torch


def precompute_rotary(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables for RoPE.

    Returns:
        cos: (max_seq_len, head_dim / 2)
        sin: (max_seq_len, head_dim / 2)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a tensor of head vectors.

    Treats consecutive pairs of the last dim as 2D points and rotates each
    pair by the angle implied by `cos`/`sin`.

    Args:
        x: shape (..., S, head_dim) — typically (B, H, S, head_dim)
        cos, sin: shape (S, head_dim / 2)

    Returns: x rotated, same shape and dtype.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)
