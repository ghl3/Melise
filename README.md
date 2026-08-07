# transformer-learning

A proof-of-concept transformer language model on Apple Silicon. Two parts:

- **`tutorial/`** — single-file, self-contained demos walking from "Hello GPU"
  through a full Mixtral-style MoE transformer. Each numbered file is a
  runnable lesson focused on one concept. See [`tutorial/README.md`](tutorial/README.md)
  for the file guide.
- **`transformer/`** — a proper Python package built up from what the tutorial
  developed: a library of interchangeable components, plus one file per model
  architecture assembled from them.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy
```

## Running the tutorial

```bash
.venv/bin/python tutorial/01_hello_gpu.py
.venv/bin/python tutorial/02_memory.py
# ... etc
```

## The package

Components (each file documents the technique it implements):

```
transformer/
  attention/   CausalSelfAttention (GQA + RoPE)   — Llama-style grouped-query attention
               MultiLatentAttention (MLA)         — DeepSeek latent-KV attention;
                                                    decoupled-RoPE or NoPE + output gate (Kimi)
               KimiDeltaAttention (KDA)           — Kimi linear attention (gated delta rule)
               cache.py                           — per-layer caches: K/V, latent, recurrent state
  ffn/         GatedMLP                           — dense SwiGLU / SiTU-GLU
               MoE                                — Mixtral-style softmax top-k + shared expert
               DeepSeekMoE                        — sigmoid routing + aux-loss-free bias balancing
               StableLatentMoE                    — Kimi K3 latent experts + Quantile Balancing
  norm/        RMSNorm
  residual/    AttentionResiduals                 — Kimi K3 depth attention over layer outputs
  models/      base | vanilla | deepseek | kimi3  — one explicit architecture per file
```

Models (see `transformer/models/`, one self-documenting file each):

| preset     | architecture                                                            |
| ---------- | ----------------------------------------------------------------------- |
| `base`     | GQA + RoPE + Mixtral-style MoE (the repo's original; old checkpoints)   |
| `vanilla`  | MHA + RoPE + dense SwiGLU — the GPT/Llama baseline                      |
| `deepseek` | DeepSeek-V3 in miniature: MLA + sigmoid-routed MoE, first layer dense   |
| `kimi3`    | Kimi K3 in miniature (arXiv:2607.24653): 3:1 KDA/Gated-MLA hybrid, NoPE, Attention Residuals, Stable LatentMoE, SiTU-GLU |

## Training and sampling

```bash
# Train (checkpoints/, metrics, auto-named run dirs — see scripts/train.py -h)
.venv/bin/python scripts/train.py --steps 5000
.venv/bin/python scripts/train.py --preset kimi3 --seq-len 256 --steps 5000

# Sample from the latest checkpoint (architecture is recovered from the checkpoint)
.venv/bin/python scripts/sample.py --checkpoint checkpoints/<run>/latest.pt --temperature 0.8

# Train on the whole data/ directory with a mixture config (sampling weight =
# byte size × per-file multiplier; see configs/mix-downweight-wiki.json)
.venv/bin/python scripts/train.py --data-mix configs/mix-downweight-wiki.json --steps 5000

# Watch a run live (loss/bpb, LR, grad norm, val, per-layer MoE expert load, samples)
.venv/bin/tensorboard --logdir checkpoints/<run>/tb
```

Training is exactly resumable: checkpoints embed the model config, optimizer
state, RNG streams, best-val tracking, and token count, and are written
atomically — `--resume <ckpt>` continues the identical run (Ctrl-C saves an
`interrupted.pt` first).

Note: the kimi3 preset's KDA layers run a reference sequential scan (the
chunkwise kernel from the paper is not implemented), so training speed drops
with `--seq-len`; 128–256 is comfortable on MPS.

## Tests

```bash
.venv/bin/python tests/test_transformer.py
```

Covers all four architectures: shapes/gradients, causality,
incremental-decode vs full-forward equivalence, MoE balancing updates, KDA
decay bounds, and pre-refactor checkpoint compatibility.
