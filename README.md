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

## Training

Three stages, each a thin script over the transformer package; run dirs
live at `checkpoints/<stage>/<run>` and mirror to
`gs://<bucket>/runs/<stage>/<run>`:

```bash
# 1. Pretrain (next-byte LM on the corpus mixture; sampling weight =
#    byte size × per-file multiplier; see configs/mix-downweight-wiki.json)
.venv/bin/python scripts/pretrain.py --preset kimi3 --seq-len 256 \
    --data-mix configs/mix-downweight-wiki.json --steps 5000

# 2. SFT (assistant-masked loss on byte-template chat data; build the data
#    with scripts/prep_chat_data.py + scripts/gen_task_sft.py)
.venv/bin/python scripts/sft.py --init checkpoints/pretrain/<run>/best.pt --steps 3000

# 3. GRPO on verifiable rewards (transformer/rl/ — tasks, rollouts, loss)
.venv/bin/python scripts/grpo.py --init checkpoints/sft/<run>/best.pt --steps 200

# Sample from any checkpoint (architecture is recovered from the checkpoint)
.venv/bin/python scripts/sample.py --checkpoint checkpoints/<stage>/<run>/latest.pt --temperature 0.8

# Exact bpb on the reserved enwik8 test slice for every best.pt
.venv/bin/python scripts/eval_checkpoint.py

# Watch all runs live, grouped by stage (loss/bpb, val, MoE load, RL reward…)
.venv/bin/tensorboard --logdir checkpoints
```

Training is exactly resumable: checkpoints embed the model config, optimizer
state, RNG streams, best-val tracking, and token count, and are written
atomically — `--resume <ckpt>` continues the identical run (Ctrl-C saves an
`interrupted.pt` first).

Note: the kimi3 preset's KDA layers use fla's chunkwise Triton kernel on
CUDA and fall back to a reference sequential scan elsewhere (MPS/CPU), so
non-CUDA training speed drops with `--seq-len`; 128–256 is comfortable on
MPS. Real SFT/GRPO runs belong on the CUDA VM.

## Tests

```bash
.venv/bin/python tests/test_transformer.py
```

Covers all four architectures: shapes/gradients, causality,
incremental-decode vs full-forward equivalence, MoE balancing updates, KDA
decay bounds, and pre-refactor checkpoint compatibility.
