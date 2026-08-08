"""Run management shared by the stage scripts (pretrain/sft/grpo).

Everything here is about *running and recording* training — naming,
logging, checkpoint files, bucket mirroring, LR scheduling, RNG capture,
routing monitors. Model and training-objective logic lives in the
transformer package; scripts compose the two.

Layout convention (local and bucket mirror):

    checkpoints/<stage>/<run-name>/     stage ∈ pretrain | sft | rlvr
    gs://<bucket>/runs/<stage>/<run-name>/

Run names lead with the stage (sft-/rlvr-) or preset so a run dir is
self-identifying even when copied out of its stage directory.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from transformer.ffn import DeepSeekMoE, MoE, StableLatentMoE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Auto-naming ----------

_ADJECTIVES = [
    "calm", "wild", "swift", "bright", "silver", "golden", "stormy",
    "gentle", "crisp", "frozen", "scarlet", "dusky", "frosty", "sunny",
    "quiet", "fierce", "eager", "jolly", "hidden", "crystal", "crimson",
    "ember", "azure", "twilight", "rugged",
]
_NOUNS = [
    "river", "mountain", "valley", "harbor", "glacier", "cypress",
    "sparrow", "otter", "wolf", "falcon", "brook", "meadow", "canyon",
    "lighthouse", "garden", "forest", "owl", "comet", "nova", "dawn",
    "willow", "raven", "tide", "summit", "dell",
]


def fmt_params(n: int) -> str:
    """17,234,567 → '17M'; 1,234,567,890 → '1.2B'."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    return f"{round(n / 1e6)}M"


def fresh_identity(preset: str, n_params: int) -> str:
    """Mint the stable lineage core for a NEW model, e.g.
    'kimi3-small-17M-scarlet-harbor'. This exact string threads through
    every later training generation (sft-<identity>-<ts>,
    rlvr-<identity>-<ts>) and is stored in every checkpoint the chain
    saves. Uses a separate time-seeded RNG so the name is independent of
    --seed."""
    rng = random.Random()  # seeded from os.urandom
    return (f"{preset}-{fmt_params(n_params)}-"
            f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}")


def stamp(identity: str) -> str:
    """identity -> run name: append a launch timestamp. Timestamps
    distinguish repeat runs of the same lineage; exact parentage lives in
    checkpoint metadata ('lineage') and run.json ('init')."""
    return f"{identity}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def generate_run_name(preset: str, n_params: int) -> str:
    """Fresh identity + timestamp, e.g. 'kimi3-medium-66M-calm-river-20260503-141522'."""
    return stamp(fresh_identity(preset, n_params))


_TS_SUFFIX = re.compile(r"-\d{8}-\d{6}$")
_STAGE_PREFIX = re.compile(r"^(sft|rlvr|grpo)-")


def strip_run_identity(run_name: str) -> str:
    """Identity from a run/dir name (drop stage prefix + timestamp).
    Legacy fallback only — checkpoints saved since 2026-08-08 carry
    'identity' in their payload, which is authoritative."""
    return _TS_SUFFIX.sub("", _STAGE_PREFIX.sub("", run_name))


def checkpoint_identity(payload: dict, ckpt_path: Path) -> str:
    """The lineage identity of a loaded checkpoint: the embedded
    'identity' field, or (for pre-metadata checkpoints) parsed from the
    checkpoint's run dir name."""
    return payload.get("identity") or strip_run_identity(
        ckpt_path.resolve().parent.name)


def derive_run_name(stage: str, payload: dict, ckpt_path: Path) -> str:
    """Run name for a derived stage: same identity as the checkpoint it
    initializes from, so lineage reads directly off TensorBoard and
    checkpoint paths:

        pretrain/kimi3-small-17M-scarlet-harbor-<ts>
          -> sft/sft-kimi3-small-17M-scarlet-harbor-<new ts>
          -> rlvr/rlvr-kimi3-small-17M-scarlet-harbor-<new ts>
    """
    return f"{stage}-{stamp(checkpoint_identity(payload, ckpt_path))}"


# ---------- Console/formatting ----------


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


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) / 60)
    return f"{h}h{m:02d}m"


# ---------- LR schedule ----------


def lr_at(step: int, args) -> float:
    """Learning rate for a (1-indexed) step. Pure function of step, so a
    resumed run lands on exactly the same schedule. Reads args.lr,
    args.lr_schedule, args.warmup_frac, args.min_lr_frac, args.steps."""
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


# ---------- Checkpointing ----------


def save_checkpoint(path, model, optimizer, step, cfg, *, preset, best_val,
                    best_step, tokens_seen, device, run_name=None,
                    identity=None, stage=None, lineage=None):
    """Atomic write: serialize to a tmp file in the same directory, then
    rename over the target. A Ctrl-C mid-save can never leave a truncated
    checkpoint at `path`.

    Provenance metadata rides in the payload so downstream stages never
    have to parse file names: `identity` is the stable lineage core
    ('kimi3-small-17M-scarlet-harbor'), `run_name`/`stage` identify the
    run that wrote this checkpoint, and `lineage` lists run names from
    the pretrain root down to this run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "preset": preset,
        "run_name": run_name,
        "identity": identity,
        "stage": stage,
        "lineage": lineage,
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


# ---------- Metrics log ----------


def open_metrics_log(path):
    """Open the metrics JSONL file in append mode, line-buffered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", buffering=1)


def emit(metrics_f, **fields):
    """Append one JSONL event. Always includes a timestamp."""
    fields.setdefault("time", datetime.now().isoformat(timespec="seconds"))
    metrics_f.write(json.dumps(fields) + "\n")


# ---------- Bucket sync ----------


class BucketSync:
    """Mirror the run dir to gs://<bucket>/runs/<stage>/<run-name>.

    The bucket is the canonical home of runs: checkpoints, TensorBoard
    events, metrics.jsonl, train.log all survive the VM, and any
    TensorBoard host can pull every machine's runs into one view.

    kick() launches a detached `gcloud storage rsync` (training never
    blocks on uploads; if the previous sync is still running it's skipped
    — the next kick catches up). Deletions are mirrored so --keep-last
    pruning applies in the bucket too. finalize() runs one last blocking
    sync so the final checkpoint always lands.

    latest.pt is excluded: it's a symlink to a step file whose bytes are
    already uploaded, and rsync would re-upload the full copy each save.
    """

    def __init__(self, out_dir: Path, enabled: bool, stage: str):
        self.out_dir = out_dir
        self.dest = None
        self.proc = None
        gcs_config = PROJECT_ROOT / "configs" / "gcs.json"
        if enabled and gcs_config.exists() and shutil.which("gcloud"):
            cfg = json.loads(gcs_config.read_text())
            self.dest = f"gs://{cfg['bucket']}/runs/{stage}/{out_dir.name}"

    def _cmd(self) -> list[str]:
        return [
            "gcloud", "storage", "rsync", "-r",
            "--delete-unmatched-destination-objects",
            "-x", r"^latest\.pt$|.*\.tmp$",
            str(self.out_dir), self.dest,
        ]

    def kick(self) -> None:
        if self.dest is None:
            return
        if self.proc is not None and self.proc.poll() is None:
            return  # previous sync still uploading; next kick catches up
        self.proc = subprocess.Popen(
            self._cmd(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def finalize(self) -> None:
        if self.dest is None:
            return
        if self.proc is not None:
            self.proc.wait()
        subprocess.run(self._cmd(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------- Model instrumentation ----------


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
        """Aggregates for quick scanning of metrics.jsonl."""
        if not self.loads:
            return {}
        max_loads = [float(l.max()) for l in self.loads.values()]
        entropies = [self.entropy(l) for l in self.loads.values()]
        return {
            "moe_max_load": max(max_loads),
            "moe_mean_entropy": sum(entropies) / len(entropies),
        }

    def detail(self) -> dict | None:
        """Per-layer stats for metrics.jsonl — metrics.jsonl is the
        authoritative record, so it must carry everything TensorBoard
        shows (rebuild_tb.py replays this into moe/L*_ scalars)."""
        if not self.loads:
            return None
        return {
            name: {
                "max_load": float(load.max()),
                "entropy": self.entropy(load),
                **({"bias_span": self.bias_spans[name]}
                   if name in self.bias_spans else {}),
            }
            for name, load in self.loads.items()
        }

    @staticmethod
    def entropy(load: torch.Tensor) -> float:
        p = load[load > 0]
        h = -(p * p.log()).sum().item()
        return h / math.log(len(load)) if len(load) > 1 else 1.0
