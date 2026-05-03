# transformer-learning

A proof-of-concept transformer language model on Apple Silicon. Two parts:

- **`tutorial/`** — single-file, self-contained demos walking from "Hello GPU"
  through a full Mixtral-style MoE transformer. Each numbered file is a
  runnable lesson focused on one concept. See [`tutorial/README.md`](tutorial/README.md)
  for the file guide.
- **`transformer/`** — a proper Python package being built up from what the
  tutorial developed. Reusable model components, organized into modules.

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

## Using the package

(Under construction.)
