"""Model architectures — one file per model, each laying out its stack
explicitly from the shared component library:

    base.py     — GQA + RoPE + Mixtral-style MoE (the repo's original;
                  existing checkpoints load into this)
    vanilla.py  — MHA + RoPE + dense SwiGLU (the GPT/Llama baseline)
    deepseek.py — MLA (decoupled RoPE) + sigmoid-routed MoE with bias
                  balancing, first layer dense
    kimi3.py    — Kimi K3: 3:1 KDA / Gated-MLA hybrid (NoPE), Attention
                  Residuals, Stable LatentMoE, SiTU-GLU

`MODELS` maps a name to (config class, model class) — used by
scripts/train.py --preset. `build_model(cfg)` picks the model class from
a config instance — used to rebuild a model from a checkpoint.
"""

from ..config import Config
from .base import TransformerLM
from .deepseek import DeepSeekConfig, DeepSeekLM
from .kimi3 import Kimi3Config, Kimi3LM, kimi3_large, kimi3_medium, kimi3_small
from .vanilla import VanillaConfig, VanillaLM

# name → (config factory, model class). A factory is either the config class
# itself or a function returning a config instance (size presets).
MODELS = {
    "base": (Config, TransformerLM),
    "vanilla": (VanillaConfig, VanillaLM),
    "deepseek": (DeepSeekConfig, DeepSeekLM),
    "kimi3": (Kimi3Config, Kimi3LM),          # alias for kimi3-small
    "kimi3-small": (kimi3_small, Kimi3LM),
    "kimi3-medium": (kimi3_medium, Kimi3LM),
    "kimi3-large": (kimi3_large, Kimi3LM),
}


def build_model(cfg):
    """Instantiate the right model class for a config instance (e.g. one
    loaded from a checkpoint)."""
    for config_cls, model_cls in MODELS.values():
        if type(cfg) is config_cls:
            return model_cls(cfg)
    raise TypeError(f"no model registered for config type {type(cfg).__name__}")


__all__ = [
    "MODELS",
    "build_model",
    "Config",
    "DeepSeekConfig",
    "DeepSeekLM",
    "Kimi3Config",
    "Kimi3LM",
    "TransformerLM",
    "VanillaConfig",
    "VanillaLM",
]
