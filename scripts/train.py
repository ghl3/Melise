"""Train a model preset on a byte-level text corpus.

Examples:

    First run (auto-generated run name in checkpoints/):
        .venv/bin/python scripts/train.py --steps 500

    Train the Kimi K3 miniature (KDA runs a sequential scan — prefer a
    modest --seq-len):
        .venv/bin/python scripts/train.py --preset kimi3 --seq-len 256 --steps 2000

    Custom run name:
        .venv/bin/python scripts/train.py --run-name my-experiment --steps 5000

    Resume training (uses the same directory; model, optimizer, RNG
    streams, and best-val tracking all continue where they left off):
        .venv/bin/python scripts/train.py \\
            --resume checkpoints/calm-river-20260503-141522/latest.pt --steps 10000

    Watch a run in TensorBoard:
        .venv/bin/tensorboard --logdir checkpoints/<run>/tb

Each run writes to its own subdirectory under checkpoints/ containing:

    step_NNNN.pt    rolling checkpoints (oldest pruned to --keep-last)
    latest.pt       symlink to most recent
    best.pt         full checkpoint, replaced whenever val_loss hits a new low
    interrupted.pt  saved on Ctrl-C if training is interrupted
    run.json        manifest: args, config, datasets, start time
    metrics.jsonl   append-only event log (steps / evals / saves / samples)
    train.log       full stdout, mirrored from console
    tb/             TensorBoard event files

What gets tracked (metrics.jsonl and TensorBoard):

    train/loss, train/bpb   cross-entropy in nats and bits-per-byte
    train/lr                the scheduled learning rate
    train/grad_norm         pre-clip global gradient norm
    train/tok_per_sec, train/tokens_seen, train/mem_gb
    val/loss, val/bpb, val/best
    params/global_norm      global L2 norm of all weights (on evals)
    moe/L*_max_load         per-MoE-layer max expert load fraction
    moe/L*_entropy          per-MoE-layer normalized routing entropy
    moe/L*_bias_span        balancing-bias spread (DeepSeek/Kimi routers)

Checkpoints embed the model config, optimizer state, RNG streams (CPU +
device), best-val tracking, and cumulative token count, so a resumed run
is bit-for-bit the run that would have happened without the interruption.
Checkpoint writes are atomic (tmp file + rename), so Ctrl-C can never
leave a truncated checkpoint behind.

Training runs in fp32 for stability; trained weights can be cast to bf16
at inference time. Run from the project root.
"""

import argparse
import json
import math
import random
import signal
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

# Make the `transformer` package importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from transformer import MODELS, build_model, generate
from transformer.ffn import DeepSeekMoE, MoE, StableLatentMoE

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed — script still works
    SummaryWriter = None

LN2 = math.log(2.0)

# ---------- Auto-naming ----------

_ADJECTIVES = [
    "calm",
    "wild",
    "swift",
    "bright",
    "silver",
    "golden",
    "stormy",
    "gentle",
    "crisp",
    "frozen",
    "scarlet",
    "dusky",
    "frosty",
    "sunny",
    "quiet",
    "fierce",
    "eager",
    "jolly",
    "hidden",
    "crystal",
    "crimson",
    "ember",
    "azure",
    "twilight",
    "rugged",
]
_NOUNS = [
    "river",
    "mountain",
    "valley",
    "harbor",
    "glacier",
    "cypress",
    "sparrow",
    "otter",
    "wolf",
    "falcon",
    "brook",
    "meadow",
    "canyon",
    "lighthouse",
    "garden",
    "forest",
    "owl",
    "comet",
    "nova",
    "dawn",
    "willow",
    "raven",
    "tide",
    "summit",
    "dell",
]


def fmt_params(n: int) -> str:
    """17,234,567 → '17M'; 1,234,567,890 → '1.2B'."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    return f"{round(n / 1e6)}M"


def generate_run_name(preset: str, n_params: int) -> str:
    """Return e.g. 'kimi3-medium-66M-calm-river-20260503-141522'.

    The name leads with the architecture preset and parameter count so
    checkpoint directories (and TensorBoard run labels) identify the
    model at a glance. Uses a separate time-seeded RNG so the chosen name
    is independent of --seed (otherwise two default-seed runs started in
    the same second would collide).
    """
    rng = random.Random()  # seeded from os.urandom
    adj = rng.choice(_ADJECTIVES)
    noun = rng.choice(_NOUNS)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{preset}-{fmt_params(n_params)}-{adj}-{noun}-{ts}"


# ---------- Tee writer (stdout -> console + file) ----------


class Tee:
    """File-like object that writes to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------- ETA formatting ----------


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) / 60)
    return f"{h}h{m:02d}m"


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("model")
    g.add_argument(
        "--preset",
        type=str,
        default="base",
        choices=sorted(MODELS),
        help="Architecture to train: base (GQA + MoE), vanilla (MHA + dense), "
        "deepseek (MLA + DeepSeekMoE), kimi3 (KDA/MLA hybrid + AttnRes + LatentMoE). "
        "Ignored when resuming — the checkpoint's config wins",
    )

    g = p.add_argument_group("training")
    g.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Total training steps (target — counts steps from any resumed checkpoint)",
    )
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--seq-len", type=int, default=512, help="Sequence length per batch")
    g.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    g.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("cosine", "constant"),
        help="cosine: linear warmup then cosine decay to --min-lr-frac * lr "
        "(schedule is a pure function of step, so it resumes exactly; "
        "note it is shaped by --steps, so extending --steps on resume "
        "reshapes the tail)",
    )
    g.add_argument(
        "--warmup-frac",
        type=float,
        default=0.01,
        help="Fraction of --steps spent in linear warmup (cosine schedule)",
    )
    g.add_argument(
        "--min-lr-frac",
        type=float,
        default=0.1,
        help="Final LR as a fraction of --lr (cosine schedule)",
    )
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")

    g = p.add_argument_group("logging & evaluation")
    g.add_argument(
        "--log-every", type=int, default=25, help="Log train metrics every N steps"
    )
    g.add_argument(
        "--eval-every", type=int, default=100, help="Run val eval every N steps"
    )
    g.add_argument(
        "--eval-batches",
        type=int,
        default=16,
        help="Number of val batches averaged per eval",
    )
    g.add_argument(
        "--sample-every", type=int, default=100, help="Generate a sample every N steps"
    )
    g.add_argument("--sample-tokens", type=int, default=200)
    g.add_argument("--sample-prompt", type=str, default="ROMEO:\n")
    g.add_argument("--no-sample", action="store_true")
    g.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard event files (tb/ subdir)",
    )

    g = p.add_argument_group("checkpointing")
    g.add_argument("--save-every", type=int, default=100)
    g.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run output directory. Default: checkpoints/{run-name}/",
    )
    g.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Auto-generated if omitted (e.g. 'calm-river-20260503-141522')",
    )
    g.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from (uses its parent dir as --out). "
        "The model is rebuilt from the checkpoint's own config",
    )
    g.add_argument(
        "--keep-last",
        type=int,
        default=5,
        help="Keep only N most recent checkpoints (best.pt's target is protected)",
    )

    g = p.add_argument_group("data")
    g.add_argument(
        "--data",
        action="append",
        type=Path,
        default=None,
        help="Path to a training corpus. Repeat for mixture training.",
    )
    g.add_argument(
        "--data-weights",
        type=str,
        default=None,
        help="Comma-separated sampling weights. Default: weight by byte size.",
    )
    g.add_argument(
        "--data-mix",
        type=Path,
        default=None,
        help="JSON mixture config: {\"include\": <glob or list of globs>, "
        "\"multipliers\": {<path or filename>: <mult>, ...}}. Sampling weight "
        "is byte_size × multiplier (default multiplier 1.0, i.e. naive "
        "byte-size weighting). Mutually exclusive with --data/--data-weights.",
    )
    g.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="Legacy --data runs only: hold out the last fraction of every "
        "file for validation. --data-mix runs use the config's per-file "
        "\"splits\" instead (files without one train on 100%% of their bytes).",
    )

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="mps")

    return p.parse_args()


def resolve_run_dir(args: argparse.Namespace, preset: str, n_params: int) -> Path:
    """Pick the output directory based on --out / --resume / --run-name."""
    if args.out is not None:
        return args.out
    if args.resume is not None:
        return args.resume.parent.resolve()
    name = args.run_name or generate_run_name(preset, n_params)
    return (PROJECT_ROOT / "checkpoints" / name).resolve()


# ---------- LR schedule ----------


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Learning rate for a (1-indexed) step. Pure function of step, so a
    resumed run lands on exactly the same schedule."""
    if args.lr_schedule == "constant":
        return args.lr
    warmup = max(int(args.warmup_frac * args.steps), 1)
    min_lr = args.min_lr_frac * args.lr
    if step <= warmup:
        return args.lr * step / warmup
    t = min((step - warmup) / max(args.steps - warmup, 1), 1.0)
    return min_lr + 0.5 * (args.lr - min_lr) * (1.0 + math.cos(math.pi * t))


# ---------- RNG state (for exact resume) ----------


def rng_state(device: torch.device) -> dict:
    """Snapshot every RNG stream training draws from: CPU (mixture-dataset
    multinomial) and the device generator (batch start offsets on MPS/CUDA)."""
    state = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict | None, device: torch.device) -> bool:
    if not state:
        return False
    try:
        torch.set_rng_state(state["cpu"])
        if device.type == "mps" and "mps" in state:
            torch.mps.set_rng_state(state["mps"])
        elif device.type == "cuda" and "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], device)
        return True
    except Exception as e:  # torch-version mismatch etc. — not fatal
        print(f"  (could not restore RNG state: {e}; batch stream restarts from --seed)")
        return False


# ---------- Data ----------


def load_data(paths, device, splits):
    """Load each corpus as a uint8 tensor (1 byte per token). Batches are
    cast to int64 on the fly in get_batch — storing the corpora themselves
    as int64 would be 8× the memory (WikiText-103 alone would be ~4.3 GB).

    `splits` maps path → (train_frac, val_frac). Files without an entry
    train on 100% of their bytes and contribute nothing to validation.
    Any remainder after train+val (e.g. enwik8's canonical last-5% test
    split) is NEVER loaded here — it stays untouched for offline evals.
    """
    train_list, byte_counts = [], []
    val_list, val_bytes = [], []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/download_data.py first."
            )
        raw = path.read_bytes()
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
        train_frac, val_frac = splits.get(path, (1.0, 0.0))
        n = data.shape[0]
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
        train_list.append(data[:train_end])
        if val_end > train_end:
            val_list.append(data[train_end:val_end])
            val_bytes.append(val_end - train_end)
        byte_counts.append(len(raw))
    return train_list, val_list, val_bytes, byte_counts


def get_batch(datasets, weights, batch_size, seq_len):
    if len(datasets) == 1:
        d = datasets[0]
        starts = torch.randint(
            0, d.shape[0] - seq_len - 1, (batch_size,), device=d.device
        )
        inputs = torch.stack([d[s : s + seq_len] for s in starts])
        targets = torch.stack([d[s + 1 : s + 1 + seq_len] for s in starts])
        return inputs.long(), targets.long()

    chosen = torch.multinomial(weights, batch_size, replacement=True).tolist()
    inputs, targets = [], []
    for d_idx in chosen:
        d = datasets[d_idx]
        s = int(torch.randint(0, d.shape[0] - seq_len - 1, (1,)).item())
        inputs.append(d[s : s + seq_len])
        targets.append(d[s + 1 : s + 1 + seq_len])
    return torch.stack(inputs).long(), torch.stack(targets).long()


def parse_weights(weights_str, byte_counts):
    if weights_str is None:
        return torch.tensor([float(b) for b in byte_counts])
    weights = [float(w) for w in weights_str.split(",")]
    if len(weights) != len(byte_counts):
        raise SystemExit(
            f"--data-weights has {len(weights)} entries; --data has {len(byte_counts)}."
        )
    return torch.tensor(weights)


def load_data_mix(mix_path):
    """Read a mixture config: which files to train on, per-file
    sampling-weight multipliers, and per-file train/val/test splits.

        {"include": "data/*.txt",                     # glob or list of globs
         "multipliers": {"data/enwik8.txt": 0.1},     # default 1.0
         "splits": {                                  # default: 100% train
            "data/enwik8.txt": {"train": 0.90, "val": 0.05, "test": 0.05}}}

    Files without a "splits" entry train on all of their bytes. The test
    fraction is reserved — train.py never loads it (offline evals do).

    Returns (paths, multipliers, splits) with multipliers aligned to
    paths and splits keyed by path. Keys may be project-relative paths or
    bare filenames.
    """
    mix = json.loads(mix_path.read_text())
    includes = mix.get("include", "data/*.txt")
    if isinstance(includes, str):
        includes = [includes]
    paths = sorted({p for g in includes for p in PROJECT_ROOT.glob(g)})
    if not paths:
        raise SystemExit(f"--data-mix: include patterns {includes} matched no files")

    def match(table, matched):
        """Look up a per-file entry by relative path or filename."""
        def lookup(p):
            rel = str(p.relative_to(PROJECT_ROOT))
            for key in (rel, p.name):
                if key in table:
                    matched.add(key)
                    return table[key]
            return None
        return lookup

    raw_mults = dict(mix.get("multipliers", {}))
    raw_splits = dict(mix.get("splits", {}))
    matched = set()
    mult_of = match(raw_mults, matched)
    split_of = match(raw_splits, matched)

    mults, splits = [], {}
    for p in paths:
        m = mult_of(p)
        mults.append(1.0 if m is None else float(m))
        s = split_of(p)
        if s is not None:
            train, val = float(s.get("train", 1.0)), float(s.get("val", 0.0))
            test = float(s.get("test", 0.0))
            if train + val + test > 1.0 + 1e-9:
                raise SystemExit(f"--data-mix: splits for {p.name} sum to > 1")
            splits[p] = (train, val)

    unmatched = (set(raw_mults) | set(raw_splits)) - matched
    if unmatched:
        raise SystemExit(
            f"--data-mix: keys match no included file: {sorted(unmatched)}"
        )
    return paths, mults, splits


# ---------- MoE routing monitor ----------


class MoEMonitor:
    """Watch expert load balance without touching the model code.

    Registers a forward-pre-hook on every MoE layer. The hooks are inert
    except on logging steps, when each one re-runs the layer's (tiny)
    router on the incoming activations and records per-expert load
    fractions. Cheap — one d×E matmul per MoE layer per logged step.

    Reported per layer:
        max_load  — worst expert's share of routings (1/E is perfectly
                    balanced; ~1.0 means router collapse)
        entropy   — routing entropy normalized to [0, 1] (1 = uniform)
        bias_span — max−min of the balancing bias, for routers that have
                    one (DeepSeekMoE sign-update, StableLatentMoE QB)
    """

    def __init__(self, model: torch.nn.Module):
        self.enabled = False
        self.loads: dict[str, torch.Tensor] = {}
        self.bias_spans: dict[str, float] = {}
        self.names: list[str] = []
        for path, mod in model.named_modules():
            if isinstance(mod, (MoE, DeepSeekMoE, StableLatentMoE)):
                name = f"L{len(self.names)}"
                self.names.append(name)
                mod.register_forward_pre_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, inputs):
            if not self.enabled:
                return
            with torch.no_grad():
                x = inputs[0].reshape(-1, inputs[0].shape[-1])
                if isinstance(module, MoE):
                    scores = module.router(x).float()
                    idx = scores.topk(module.top_k, dim=-1).indices
                else:  # sigmoid routers select through their balancing bias
                    scores = torch.sigmoid(module.router(x).float())
                    idx = (scores + module.route_bias).topk(module.top_k, dim=-1).indices
                n_experts = module.router.out_features
                counts = torch.bincount(idx.reshape(-1), minlength=n_experts).float()
                self.loads[name] = (counts / counts.sum()).cpu()
                if hasattr(module, "route_bias"):
                    b = module.route_bias
                    self.bias_spans[name] = float((b.max() - b.min()).item())

        return hook

    def summary(self) -> dict:
        """Aggregates for metrics.jsonl (per-layer detail goes to TensorBoard)."""
        if not self.loads:
            return {}
        max_loads = [float(l.max()) for l in self.loads.values()]
        entropies = [self.entropy(l) for l in self.loads.values()]
        return {
            "moe_max_load": max(max_loads),
            "moe_mean_entropy": sum(entropies) / len(entropies),
        }

    @staticmethod
    def entropy(load: torch.Tensor) -> float:
        p = load[load > 0]
        h = -(p * p.log()).sum().item()
        return h / math.log(len(load)) if len(load) > 1 else 1.0


# ---------- Eval & sampling ----------


@torch.no_grad()
def eval_loss(model, val_datasets, weights, batch_size, seq_len, n_batches):
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(n_batches):
            inputs, targets = get_batch(val_datasets, weights, batch_size, seq_len)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, model.cfg.vocab_size), targets.view(-1)
            )
            total += loss.item()
        return total / n_batches
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def sample_text(model, device, prompt, n_tokens):
    was_training = model.training
    model.eval()
    try:
        ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
        out_ids = generate(model, ids, max_new_tokens=n_tokens)
        out_bytes = bytes(b if 0 <= b < 256 else 0x3F for b in out_ids)
        full = prompt + out_bytes
        return full.decode("utf-8", errors="replace")
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def global_param_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def device_mem_gb(device: torch.device) -> float | None:
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e9
    return None


# ---------- Checkpointing ----------


def save_checkpoint(path, model, optimizer, step, cfg, *, preset, best_val,
                    best_step, tokens_seen, device):
    """Atomic write: serialize to a tmp file in the same directory, then
    rename over the target. A Ctrl-C mid-save can never leave a truncated
    checkpoint at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "preset": preset,
        "best_val": best_val,
        "best_step": best_step,
        "tokens_seen": tokens_seen,
        "rng": rng_state(device),
        "torch_version": torch.__version__,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def update_symlink(out_dir, name, target):
    """(re)point `out_dir/name` at `target` (relative filename)."""
    link = out_dir / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)


def prune_old_checkpoints(out_dir, keep_last):
    """Delete all but the most recent `keep_last` step_*.pt files.

    `best.pt` is a regular file (not a symlink to a step_*.pt), so pruning
    step files doesn't risk deleting it.
    """
    if keep_last <= 0:
        return []
    ckpts = sorted(
        out_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if len(ckpts) <= keep_last:
        return []
    to_delete = ckpts[:-keep_last]
    for p in to_delete:
        p.unlink()
    return to_delete


def recover_best_from_metrics(metrics_path):
    """Scan an existing metrics.jsonl for the best val seen so far.

    Fallback for resuming checkpoints from before best_val was stored in
    the checkpoint itself. Returns (best_val, best_step) or (inf, None).
    """
    best_val = float("inf")
    best_step = None
    if not metrics_path.exists():
        return best_val, best_step
    with open(metrics_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "eval":
                v = ev.get("val_loss")
                if v is not None and v < best_val:
                    best_val = v
                    best_step = ev.get("step")
    return best_val, best_step


# ---------- Metrics + run manifest ----------


def write_run_manifest(path, args, cfg, preset, n_params, byte_counts, weights):
    """One-shot snapshot of the run setup. Written at start; never updated."""
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": path.parent.name,
        "preset": preset,
        "n_params": n_params,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "config": asdict(cfg) if is_dataclass(cfg) else {},
        "config_dtype": str(cfg.dtype),
        "datasets": [
            {"path": str(p), "bytes": int(b), "weight": float(w)}
            for p, b, w in zip(args.data, byte_counts, weights.tolist())
        ],
    }
    # cfg.dtype is a torch.dtype; not JSON-serializable. Strip from inner dict.
    manifest["config"].pop("dtype", None)
    path.write_text(json.dumps(manifest, indent=2, default=str))


def open_metrics_log(path):
    """Open the metrics JSONL file in append mode, line-buffered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", buffering=1)


def emit(metrics_f, **fields):
    """Append one JSONL event. Always includes a timestamp."""
    fields.setdefault("time", datetime.now().isoformat(timespec="seconds"))
    metrics_f.write(json.dumps(fields) + "\n")


# ---------- Main ----------


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # TF32 matmuls on Ampere+ — big speedup for fp32 training, negligible
    # precision cost. No-op on non-CUDA devices.
    torch.set_float32_matmul_precision("high")

    # Build the model BEFORE resolving the run dir — auto-generated run
    # names lead with the preset and parameter count. On a fresh run the
    # preset decides the architecture; on resume the checkpoint's embedded
    # config is authoritative, so a wrong --preset/--seq-len flag can't
    # silently build a mismatched model. Notes are printed once the
    # train.log tee is open so they're captured in the log.
    notes = []
    ckpt = None
    if args.resume is not None:
        notes.append(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        preset = ckpt.get("preset", args.preset)
        if preset != args.preset and "preset" in ckpt:
            notes.append(f"  note: checkpoint was trained with --preset {preset}; using that")
        if cfg.max_seq_len != args.seq_len:
            notes.append(
                f"  note: checkpoint config has max_seq_len={cfg.max_seq_len}; "
                f"overriding --seq-len {args.seq_len}"
            )
            args.seq_len = cfg.max_seq_len
        model = build_model(cfg).to(device)
    else:
        preset = args.preset
        config_cls, model_cls = MODELS[preset]
        cfg = config_cls(dtype=torch.float32, max_seq_len=args.seq_len)
        model = model_cls(cfg).to(device)

    out_dir = resolve_run_dir(args, preset, model.num_parameters())
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out = out_dir

    # Mirror stdout to a per-run train.log. Closed in the finally block.
    log_file = open(out_dir / "train.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = open_metrics_log(metrics_path)

    for note in notes:
        print(note)
    print(
        f"run dir: {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}"
    )
    print(
        f"model:   {preset} ({type(model).__name__}), {model.num_parameters():,} params, "
        f"dtype={cfg.dtype}, device={device}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Restore training state from the checkpoint: weights, optimizer
    # moments, step counter, best-val tracking, token count, RNG streams.
    start_step = 0
    tokens_seen = 0
    best_val = float("inf")
    best_step = None
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", start_step * args.batch_size * args.seq_len))
        if "best_val" in ckpt and ckpt["best_val"] is not None:
            best_val, best_step = ckpt["best_val"], ckpt.get("best_step")
        else:  # pre-refactor checkpoint — reconstruct from the event log
            best_val, best_step = recover_best_from_metrics(metrics_path)
        if restore_rng_state(ckpt.get("rng"), device):
            print(f"resumed at step {start_step} (RNG streams restored)")
        else:
            print(f"resumed at step {start_step}")
        if best_step is not None:
            print(f"best so far: val_loss={best_val:.4f} at step {best_step}")
        del ckpt  # free the 100s-of-MB state dict copy

    if start_step >= args.steps:
        print(f"nothing to do — start_step ({start_step}) >= --steps ({args.steps}).")
        return

    mix_mults = None
    splits = None
    if args.data_mix is not None:
        if args.data or args.data_weights:
            raise SystemExit("--data-mix is mutually exclusive with --data/--data-weights")
        args.data, mix_mults, splits = load_data_mix(args.data_mix)
        print(f"data mix: {args.data_mix} ({len(args.data)} files)")
    if not args.data:
        args.data = [PROJECT_ROOT / "data" / "tinyshakespeare.txt"]
    if splits is None:
        # Legacy --data path: hold out the last val_frac of every file.
        splits = {p: (1.0 - args.val_frac, args.val_frac) for p in args.data}

    train_data, val_data, val_bytes, byte_counts = load_data(args.data, device, splits)
    if mix_mults is not None:
        weights = torch.tensor([b * m for b, m in zip(byte_counts, mix_mults)])
    else:
        weights = parse_weights(args.data_weights, byte_counts)
    norm_weights = weights / weights.sum()
    # Val batches are drawn from the val slices only, weighted by their size.
    val_weights = torch.tensor([float(b) for b in val_bytes]) if val_bytes else None
    if val_weights is None:
        print("note: no file defines a val split — skipping evals and best.pt tracking")

    print(f"datasets ({len(args.data)}):")
    for path, n_bytes, w in zip(args.data, byte_counts, norm_weights.tolist()):
        rel = (
            path.relative_to(PROJECT_ROOT)
            if path.is_absolute() and path.is_relative_to(PROJECT_ROOT)
            else path
        )
        train_frac, val_frac = splits.get(path, (1.0, 0.0))
        note = ""
        if (train_frac, val_frac) != (1.0, 0.0):
            test_frac = 1.0 - train_frac - val_frac
            note = f"  [train {train_frac:.0%} / val {val_frac:.0%}" + (
                f" / test {test_frac:.0%} reserved]" if test_frac > 1e-9 else "]"
            )
        print(f"  {str(rel):<42}  {n_bytes:>10,} bytes  ({w * 100:>5.1f}% sampling){note}")

    # Run manifest (only on first start; preserves original on resume).
    manifest_path = out_dir / "run.json"
    if not manifest_path.exists():
        write_run_manifest(
            manifest_path, args, cfg, preset, model.num_parameters(),
            byte_counts, norm_weights,
        )

    # TensorBoard writer. Resumed runs append to the same tb/ dir; steps
    # continue from where they left off, so curves join up seamlessly.
    writer = None
    if not args.no_tensorboard:
        if SummaryWriter is None:
            print("tensorboard not installed — skipping (pip install tensorboard)")
        else:
            writer = SummaryWriter(log_dir=str(out_dir / "tb"))
            if start_step == 0:
                config_md = json.dumps(
                    {k: str(v) for k, v in asdict(cfg).items()}, indent=2
                )
                writer.add_text("run/config", f"```\n{config_md}\n```", 0)
                writer.add_text("run/preset", preset, 0)

    monitor = MoEMonitor(model)
    if monitor.names:
        print(f"tracking {len(monitor.names)} MoE layers for load balance")

    print(
        f"training from step {start_step + 1} to {args.steps} "
        f"(batch={args.batch_size}, seq_len={args.seq_len}, lr={args.lr}, "
        f"schedule={args.lr_schedule})"
    )
    print()

    emit(
        metrics_f,
        event="start",
        step=start_step,
        preset=preset,
        n_params=model.num_parameters(),
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        resumed=args.resume is not None,
    )

    # Set up SIGINT (Ctrl-C) handler. We don't exit immediately — we set a
    # flag and let the next iteration of the loop save cleanly and then exit.
    interrupted = {"flag": False}

    def sigint_handler(_signum, _frame):
        interrupted["flag"] = True
        print("\n[SIGINT] will save and exit at next step boundary...")

    signal.signal(signal.SIGINT, sigint_handler)

    def full_save(path, step):
        save_checkpoint(
            path, model, optimizer, step, cfg,
            preset=preset, best_val=best_val, best_step=best_step,
            tokens_seen=tokens_seen, device=device,
        )

    model.train()
    t_start = time.perf_counter()
    n_done = 0
    last_step_done = start_step

    try:
        for step in range(start_step + 1, args.steps + 1):
            if interrupted["flag"]:
                break

            lr = lr_at(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            is_log_step = step == start_step + 1 or step % args.log_every == 0
            monitor.enabled = is_log_step

            inputs, targets = get_batch(
                train_data, weights, args.batch_size, args.seq_len
            )
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            monitor.enabled = False
            n_done += 1
            last_step_done = step
            tokens_seen += args.batch_size * args.seq_len

            if is_log_step:
                elapsed = time.perf_counter() - t_start
                tps = n_done * args.batch_size * args.seq_len / elapsed
                steps_left = args.steps - step
                eta = (elapsed / n_done) * steps_left
                loss_val = float(loss.item())
                gnorm = float(grad_norm.item())
                mem = device_mem_gb(device)
                print(
                    f"step {step:>5}/{args.steps}  train_loss={loss_val:.4f}  "
                    f"bpb={loss_val / LN2:.3f}  grad_norm={gnorm:.2f}  "
                    f"lr={lr:.2e}  ({tps:>6.0f} tok/s)  ETA {fmt_eta(eta)}"
                )
                emit(
                    metrics_f,
                    event="step",
                    step=step,
                    train_loss=loss_val,
                    bpb=loss_val / LN2,
                    lr=lr,
                    grad_norm=gnorm,
                    tok_per_sec=tps,
                    tokens_seen=tokens_seen,
                    eta_s=eta,
                    **monitor.summary(),
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, step)
                    writer.add_scalar("train/bpb", loss_val / LN2, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", gnorm, step)
                    writer.add_scalar("train/tok_per_sec", tps, step)
                    writer.add_scalar("train/tokens_seen", tokens_seen, step)
                    if mem is not None:
                        writer.add_scalar("train/mem_gb", mem, step)
                    for name, load in monitor.loads.items():
                        writer.add_scalar(f"moe/{name}_max_load", float(load.max()), step)
                        writer.add_scalar(f"moe/{name}_entropy", monitor.entropy(load), step)
                    for name, span in monitor.bias_spans.items():
                        writer.add_scalar(f"moe/{name}_bias_span", span, step)

            if step % args.eval_every == 0 and val_weights is not None:
                val = eval_loss(
                    model,
                    val_data,
                    val_weights,
                    args.batch_size,
                    args.seq_len,
                    args.eval_batches,
                )
                is_best = val < best_val
                if is_best:
                    best_val = val
                    best_step = step
                    # Save best.pt immediately at the eval that produced the
                    # new best — no eval-vs-save alignment gap.
                    full_save(out_dir / "best.pt", step)
                pnorm = global_param_norm(model)
                print(
                    f"        val_loss={val:.4f}  val_bpb={val / LN2:.3f}"
                    + ("  ← best" if is_best else "")
                )
                emit(
                    metrics_f,
                    event="eval",
                    step=step,
                    val_loss=val,
                    val_bpb=val / LN2,
                    tokens_seen=tokens_seen,
                    param_norm=pnorm,
                    is_best=is_best,
                )
                if writer is not None:
                    writer.add_scalar("val/loss", val, step)
                    writer.add_scalar("val/bpb", val / LN2, step)
                    writer.add_scalar("val/best", best_val, step)
                    writer.add_scalar("params/global_norm", pnorm, step)
                    writer.flush()

            if step % args.sample_every == 0 and not args.no_sample:
                text = sample_text(
                    model, device, args.sample_prompt.encode(), args.sample_tokens
                ).rstrip()
                print("---")
                print(text)
                print("---")
                emit(
                    metrics_f,
                    event="sample",
                    step=step,
                    prompt=args.sample_prompt,
                    text=text,
                )
                if writer is not None:
                    writer.add_text("samples", f"```\n{text}\n```", step)

            if step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step}.pt"
                full_save(ckpt_path, step)
                update_symlink(out_dir, "latest.pt", ckpt_path)
                pruned = prune_old_checkpoints(out_dir, args.keep_last)
                msg = f"        saved {ckpt_path.name}  (latest -> {ckpt_path.name})"
                if pruned:
                    msg += f"  pruned {len(pruned)}"
                print(msg)
                emit(
                    metrics_f,
                    event="save",
                    step=step,
                    path=ckpt_path.name,
                    pruned=len(pruned),
                )
    finally:
        # Always: save an interrupted/final checkpoint so we never lose work.
        elapsed = time.perf_counter() - t_start
        if interrupted["flag"]:
            ckpt_path = out_dir / "interrupted.pt"
            full_save(ckpt_path, last_step_done)
            update_symlink(out_dir, "latest.pt", ckpt_path)
            print(f"\ninterrupted at step {last_step_done} after {elapsed:.1f}s")
            print(f"  saved {ckpt_path.name}; resume with --resume {ckpt_path}")
            emit(metrics_f, event="interrupted", step=last_step_done, elapsed_s=elapsed)
        elif n_done > 0:
            print(
                f"\ndone — {n_done} steps in {elapsed:.1f}s "
                f"({n_done / elapsed:.2f} steps/s)"
            )
            # Avoid a duplicate save when the loop's final iteration already
            # saved this step (i.e. last_step_done % save_every == 0).
            final = out_dir / f"step_{last_step_done}.pt"
            if not final.exists():
                full_save(final, last_step_done)
                update_symlink(out_dir, "latest.pt", final)
                prune_old_checkpoints(out_dir, args.keep_last)
            print(f"final checkpoint: {final.name}")
            if best_step is not None:
                print(f"best val_loss={best_val:.4f} at step {best_step} (best.pt)")
            emit(
                metrics_f,
                event="end",
                step=last_step_done,
                elapsed_s=elapsed,
                tokens_seen=tokens_seen,
                best_val=best_val if best_step else None,
                best_step=best_step,
            )
        if writer is not None:
            writer.close()
        metrics_f.close()
        # Restore stdout/stderr and close the log file.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
