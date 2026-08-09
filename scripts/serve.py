"""Chat inference server: streams completions from trained checkpoints.

    .venv/bin/python scripts/serve.py                       # newest best.pt per stage
    .venv/bin/python scripts/serve.py --model demo=checkpoints/rlvr/<run>/best.pt
    SERVE_TOKEN=secret .venv/bin/python scripts/serve.py    # require bearer auth

One process, batch-1 decoding, models loaded once at startup. Each
checkpoint self-describes its architecture and tokenizer (the config is
embedded), so byte and BPE models coexist behind one endpoint.

    GET  /healthz     "ok" once models are loaded (Cloud Run startup probe)
    GET  /v1/models   registry: name, stage, params, tokenizer, context
    POST /v1/chat     {model?, messages, temperature?, top_k?, max_tokens?}
                      -> SSE stream: {"delta": text}* then {"done": stats}

Generation applies the chat template (transformer.chat), stops at the
end-turn id, and streams UTF-8-safely: a token can end mid-multibyte-
character, so trailing replacement chars are held back until the next
token (or the end) resolves them. Conversations longer than the model's
context drop oldest turns first; the reply always keeps the full budget.

Auth: if SERVE_TOKEN is set, /v1/* requires "Authorization: Bearer
<token>" — the Vercel proxy holds the token; browsers never see it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anyio
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from transformer.chat import sanitize
from transformer.models import build_model
from transformer.tokenizer import load_tokenizer

STAGES = ("rlvr", "sft", "pretrain")  # discovery order: most chat-tuned first

app = FastAPI(title="flora inference worker")
MODELS: dict[str, dict] = {}  # name -> {model, tok, cfg, stage, params}
DEVICE = "cpu"
MAX_NEW_CAP = 512
MAX_SECONDS = 120.0  # wall-clock cap per generation
_busy = threading.Semaphore(1)  # batch-1 server: one generation at a time


# ---------- loading ----------

def load_checkpoint(name: str, path: Path, device: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt.pop("optimizer", None)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = load_tokenizer(getattr(cfg, "tokenizer", "bytes"))
    params = sum(p.numel() for p in model.parameters())
    stage = path.resolve().parent.parent.name
    print(f"  {name}: {params:,} params, tokenizer={tok.name} "
          f"(vocab {tok.vocab_size}), ctx {cfg.max_seq_len}, from {path}")
    return {"model": model, "tok": tok, "cfg": cfg, "stage": stage,
            "params": params}


def discover_models() -> dict[str, Path]:
    """Newest finished run per stage, most chat-capable stage first."""
    found = {}
    for stage in STAGES:
        runs = sorted((PROJECT_ROOT / "checkpoints" / stage).glob("*/best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            found[runs[0].parent.name] = runs[0]
    return found


# ---------- prompt assembly ----------

def conversation_ids(messages: list[dict], tok, ctx: int, reserve: int):
    """Chat-template token ids ending with the assistant marker. Drops
    oldest turns (never the final user turn) until the prompt leaves
    `reserve` tokens of reply budget inside `ctx`."""
    turns = [(m["role"], tok.encode(sanitize(m["content"]).strip()))
             for m in messages]
    dropped = 0
    while True:
        ids = []
        for role, content in turns:
            ids.append(tok.user_id if role == "user" else tok.assistant_id)
            ids.extend(content)
            ids.append(tok.end_turn_id)
        ids.append(tok.assistant_id)
        if len(ids) <= ctx - reserve or len(turns) <= 1:
            return ids, dropped
        turns.pop(0)
        dropped += 1


# ---------- streaming generation ----------

class StreamDecoder:
    """Incremental detokenizer that never emits a char it might retract:
    trailing U+FFFD from a mid-character token boundary is held back
    until later tokens (or flush) settle it."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, tid: int) -> str:
        self.ids.append(tid)
        full = self.tok.decode(self.ids)
        keep = len(full)
        while keep > self.sent and keep > len(full) - 4 \
                and full[keep - 1] == "�":
            keep -= 1
        out, self.sent = full[self.sent:keep], keep
        return out

    def flush(self) -> str:
        full = self.tok.decode(self.ids)
        out, self.sent = full[self.sent:], len(full)
        return out


def generate(entry: dict, ids: list[int], max_new: int, temperature: float,
             top_k: int):
    """Yield token ids one at a time; stops after end_turn or max_new."""
    model, tok = entry["model"], entry["tok"]
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        cache = model.new_cache(1, DEVICE)
        logits = model(x, kv_cache=cache)[:, -1]
        for _ in range(max_new):
            if temperature <= 0:
                tid = int(logits.argmax(dim=-1))
            else:
                scaled = logits / temperature
                if top_k > 0:
                    kth = torch.topk(scaled, min(top_k, scaled.shape[-1]))[0][:, -1]
                    scaled = scaled.masked_fill(scaled < kth, -torch.inf)
                tid = int(torch.multinomial(F.softmax(scaled, dim=-1), 1))
            yield tid
            if tid == tok.end_turn_id:
                return
            logits = model(torch.tensor([[tid]], dtype=torch.long,
                                        device=DEVICE), kv_cache=cache)[:, -1]


# ---------- HTTP ----------

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8192)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0, le=4096)
    max_tokens: int = Field(default=256, ge=1)


def check_auth(request: Request):
    token = os.environ.get("SERVE_TOKEN")
    if token and request.headers.get("authorization") != f"Bearer {token}":
        raise HTTPException(401, "bad or missing bearer token")


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok" if MODELS else PlainTextResponse("loading", status_code=503)


@app.get("/v1/models", dependencies=[Depends(check_auth)])
def models():
    return [
        {"name": name, "stage": e["stage"], "params": e["params"],
         "tokenizer": e["tok"].name, "vocab_size": e["tok"].vocab_size,
         "max_seq_len": e["cfg"].max_seq_len, "default": i == 0}
        for i, (name, e) in enumerate(MODELS.items())
    ]


@app.post("/v1/chat", dependencies=[Depends(check_auth)])
def chat(req: ChatRequest):
    name = req.model or next(iter(MODELS))
    entry = MODELS.get(name)
    if entry is None:
        raise HTTPException(404, f"unknown model {name!r}")
    if req.messages[-1].role != "user":
        raise HTTPException(400, "last message must be from the user")
    max_new = min(req.max_tokens, MAX_NEW_CAP)
    ids, dropped = conversation_ids(
        [m.model_dump() for m in req.messages], entry["tok"],
        entry["cfg"].max_seq_len, reserve=max_new)
    max_new = min(max_new, entry["cfg"].max_seq_len - len(ids))
    if max_new < 1:
        raise HTTPException(400, "conversation does not fit the model context")

    def event(obj) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    _DONE = object()

    async def stream():
        # Async generator + per-token thread hops: on client disconnect
        # starlette CANCELS this coroutine at an await, so finally runs
        # and the semaphore is released. (A plain sync generator gets
        # abandoned *suspended* on disconnect — finally never runs and
        # the busy lock is held forever. Learned the hard way.)
        if not await anyio.to_thread.run_sync(lambda: _busy.acquire(timeout=15)):
            yield event({"error": "server busy — one generation at a time"})
            return
        gen = generate(entry, ids, max_new, req.temperature, req.top_k)
        try:
            dec = StreamDecoder(entry["tok"])
            n_out, stopped, truncated = 0, False, False
            t0 = time.perf_counter()
            while True:
                tid = await anyio.to_thread.run_sync(
                    lambda: next(gen, _DONE), abandon_on_cancel=False)
                if tid is _DONE:
                    break
                n_out += 1
                stopped = tid == entry["tok"].end_turn_id
                if delta := dec.push(tid):
                    yield event({"delta": delta})
                if time.perf_counter() - t0 > MAX_SECONDS:
                    truncated = True
                    break
            if tail := dec.flush():
                yield event({"delta": tail})
            dt = time.perf_counter() - t0
            yield event({"done": {
                "model": name, "prompt_tokens": len(ids), "tokens": n_out,
                "tok_per_s": round(n_out / dt, 1), "stopped": stopped,
                "truncated": truncated, "dropped_turns": dropped}})
        except Exception as exc:  # surface, don't hang the stream
            yield event({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            gen.close()
            _busy.release()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------- entrypoint ----------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", metavar="NAME=PATH",
                   help="checkpoint to serve (repeatable); default: newest "
                        "best.pt per stage under checkpoints/")
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    p.add_argument("--max-new", type=int, default=512,
                   help="hard cap on tokens per completion")
    p.add_argument("--max-seconds", type=float, default=120.0,
                   help="wall-clock cap per generation")
    args = p.parse_args()

    global DEVICE, MAX_NEW_CAP, MAX_SECONDS
    DEVICE, MAX_NEW_CAP, MAX_SECONDS = args.device, args.max_new, args.max_seconds

    if args.model:
        wanted = {}
        for spec in args.model:
            name, _, path = spec.partition("=")
            if not path:
                p.error(f"--model expects NAME=PATH, got {spec!r}")
            wanted[name] = Path(path)
    else:
        wanted = discover_models()
    if not wanted:
        raise SystemExit("no checkpoints found — pass --model NAME=PATH")

    print(f"loading {len(wanted)} model(s) on {DEVICE}:")
    for name, path in wanted.items():
        MODELS[name] = load_checkpoint(name, path, DEVICE)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
