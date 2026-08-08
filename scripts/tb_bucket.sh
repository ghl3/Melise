#!/bin/bash
# TensorBoard over the bucket — view every run (all stages, all machines)
# with no VM required:
#
#     bash scripts/tb_bucket.sh                 # everything
#     bash scripts/tb_bucket.sh runs/sft        # one stage (faster scan)
#
# Needs: pip install gcsfs (in the venv), one-time
# `gcloud auth application-default login`. SSL_CERT_FILE points gcsfs's
# aiohttp at certifi's CA bundle — macOS framework Python ships without
# system certs (same fix scripts/hf_util.py applies internally).
set -euo pipefail
cd "$(dirname "$0")/.."
BUCKET=$(python3 -c "import json; print(json.load(open('configs/gcs.json'))['bucket'])")
PREFIX="${1:-runs}"
export SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')
exec .venv/bin/tensorboard --logdir "gs://$BUCKET/$PREFIX" --port "${TB_PORT:-6006}"
