# Chat inference worker (scripts/serve.py) — CPU-only, Cloud Run ready.
#
# The image deliberately skips CUDA, triton, and flash-linear-attention:
# the KDA kernel import is guarded (transformer/attention/kda.py), and
# batch-1 decoding of these models is comfortably interactive on CPU.
# CPU wheels keep the image ~1 GB instead of ~3+, which is most of the
# cold-start time on scale-to-zero Cloud Run.
#
# Checkpoints are baked in at build time. Stage them first:
#
#     mkdir -p serve_models/rlvr/<run>
#     cp checkpoints/rlvr/<run>/best.pt serve_models/rlvr/<run>/best.pt
#     docker build -t chat-worker .
#
# serve_models/ mirrors the checkpoints/<stage>/<run>/best.pt layout, so
# serve.py's stage discovery works unchanged inside the container.
#
# Deploy:
#     gcloud run deploy chat-worker --source . --region us-central1 \
#         --cpu 2 --memory 2Gi --concurrency 1 --min-instances 0 \
#         --timeout 300 --cpu-boost --set-env-vars SERVE_TOKEN=<secret>

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" tokenizers

COPY transformer/ transformer/
# Every trained tokenizer ships: checkpoints self-describe which one
# they need (gen-2 = bpe4k, gen-3+ = bpe8k; a missing artifact crashes
# model load at startup).
COPY configs/tokenizer-*.json configs/
COPY scripts/serve.py scripts/serve.py
COPY serve_models/ checkpoints/

# Cloud Run sets PORT (serve.py reads it); 8080 for local docker run.
ENV PORT=8080
EXPOSE 8080

CMD ["python", "scripts/serve.py"]
