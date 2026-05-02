# transformer-learning

A self-contained, incremental walkthrough of how modern transformer language
models work end-to-end — from the first GPU allocation to a full Mixtral-style
MoE block with GQA, RoPE, FlashAttention, and bf16 storage.

Each numbered file is a runnable demo focused on one concept. Files build on
each other conceptually but most are self-contained code (you can read them
in any order, though the numbering is the order they were intended in).

The model is small enough to run on a Mac's MPS GPU. Outputs are gibberish
because the weights are random — the goal is to see the architecture and
inference pipeline, not to train a usable language model.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy
```

Run any file directly:

```bash
.venv/bin/python 01_hello_gpu.py
```

## File guide

| File | Topic |
|------|-------|
| `01_hello_gpu.py` | The CPU/GPU boundary. `torch.device("mps")`, `.to(device)`, sync points. |
| `02_memory.py` | PyTorch's caching allocator. `current_allocated_memory` vs `driver_allocated_memory`, `empty_cache`. |
| `03_embedding_and_head.py` | The frame of an LM: embedding + LM head, no transformer layers in between. |
| `04_attention.py` | Causal multi-head self-attention from scratch. Q/K/V projections, scaled dot-product, causal mask, softmax. |
| `05_block.py` | A full transformer block: pre-norm, attention, MLP, two residual additions. |
| `06_full_model.py` | Stack of blocks + token + positional embeddings + final norm + LM head. End-to-end forward pass. |
| `07_kv_cache.py` | The KV cache. Pre-allocated buffers, in-place writes, prefill vs decode equivalence. |
| `08_inference.py` | Real autoregressive generation. Prefill once, decode in a loop, time both phases. |
| `09_sharding.py` | Tensor parallelism — column-parallel `qkv_proj`, row-parallel `o_proj`, simulated `all_reduce`. |
| `10_mla_moe.py` | DeepSeek-style block: Multi-Latent Attention (compressed KV cache) + Mixture of Experts (sparse routing). |
| `11_rope.py` | Rotary Position Embedding. Replaces additive `pos_embed`. Position-as-rotation; relative-offset property. |
| `12_flash_attention.py` | `F.scaled_dot_product_attention`. Equivalence to manual attention; the asymptotic memory story. |
| `13_quantization.py` | Per-channel int8 weight quantization from scratch. Memory savings, output drift, what production does. |
| `14_final_model.py` | All of the above integrated: GQA + RoPE + MoE (Mixtral-style + shared expert) + SDPA + bf16. |

## Final model architecture (file 14)

```
Architecture
  d_model           512
  n_heads            16    (Q heads)
  n_kv_heads          4    (GQA, 4:1 grouping)
  head_dim           32
  n_layers            4
  positional         RoPE
  norm               RMSNorm (pre-norm, no biases)
  block structure    sequential (Pre-RMSNorm + residual)

MLP (MoE)
  routed experts      8
  top_k               2
  expert_hidden    1024    (SwiGLU inside)
  shared expert       1    (always-on, hidden=1024)

Engineering
  attention kernel   torch.nn.functional.scaled_dot_product_attention
  precision          bf16 throughout
  inference          prefill + decode with KV cache
```

~60M parameters total, ~22M active per token. Architecturally a Mixtral-class
MoE with a DeepSeek-style shared expert.

## What's missing vs. production

This project covers the *architecture* and *inference loop*. Production stacks
add (none of which are implemented here):

- Training (random weights here, never trained)
- Tokenization (raw bytes used; production uses BPE / SentencePiece)
- Quantization for deployment (int4/int8 with calibration — file 13 shows the
  naive version)
- Multi-device parallelism (TP within node, EP across nodes — file 9 simulates
  TP on a single device)
- Continuous batching, prefix caching, paged attention
- Custom kernels for routing / all-to-all (DeepEP, etc.)
- Speculative decoding, beam search, advanced sampling

The architectural skeleton stays the same; everything in production wraps
around it.
