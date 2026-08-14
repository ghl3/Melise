# Papers

Local copies of every paper the architecture, training code, or run
docs lean on. Filenames are `<slug>-<arxiv-id>.pdf`; grab updates from
`https://arxiv.org/abs/<id>`.

## Core architecture

| file | paper | where it shows up here |
|---|---|---|
| `kimi-k3-2607.24653.pdf` | **Kimi K3: Open Frontier Intelligence** | The blueprint. `transformer/models/kimi3.py` is K3 in miniature: 3:1 KDA/Gated-MLA hybrid (NoPE), Attention Residuals (§2.2), Stable LatentMoE, dense first FFN, shape rules used to size kimi3-medium. |
| `kimi-linear-kda-2510.26692.pdf` | **Kimi Linear: An Expressive, Efficient Attention Architecture** | Source of KDA (Kimi Delta Attention), the linear-attention layer carrying position in kimi3; the `flash-linear-attention` `chunk_kda` kernel used on the VM implements it. |
| `deepseek-v2-mla-2405.04434.pdf` | **DeepSeek-V2** | MLA (multi-head latent attention, decoupled RoPE) — `transformer/models/deepseek.py`, and the Gated-MLA layers in kimi3. |
| `deepseek-v3-2412.19437.pdf` | **DeepSeek-V3 Technical Report** | Sigmoid routing + the dense-first-layer convention; the V3 MoE recipe our `ffn/deepseek_moe.py` follows. |
| `deepseekmoe-shared-experts-2401.06066.pdf` | **DeepSeekMoE** | Shared-expert + fine-grained routed-expert design (kimi3 runs 2 shared experts). |
| `auxloss-free-moe-balancing-2408.15664.pdf` | **Auxiliary-Loss-Free Load Balancing for MoE** | The per-expert bias balancing (no aux loss) implemented in `ffn/deepseek_moe.py`. |
| `attention-is-all-you-need-1706.03762.pdf` | **Attention Is All You Need** | The transformer. `transformer/models/vanilla.py` and everything since. |

## Training / scaling

| file | paper | where it shows up here |
|---|---|---|
| `deepseekmath-grpo-2402.03300.pdf` | **DeepSeekMath** | GRPO — `transformer/rl/grpo.py` cites it directly; the RLVR stage of the pipeline. |
| `chinchilla-compute-optimal-2203.15556.pdf` | **Training Compute-Optimal Large Language Models** | The ~20 tok/param rule used to size every generation's step budget (gen-3: 145k steps ≈ 20.6 tok/param). |
| `lm-saturation-softmax-bottleneck-godey-2404.07647.pdf` | **Why do small LMs underperform? (softmax bottleneck saturation)** | Head-rank (d/V) saturation argument; informs the bpe16k deferral and the lm_head SVD check in `docs/runs/gen4-ideas.md`. |

Adjacent but not copied here: Kimi K2 technical report (kimi3.py
mentions it in passing), Mixtral (expert-specialization analysis
discussed during gen-3), FineWeb datasets paper (data provenance —
see `docs/DATASETS.md`).
