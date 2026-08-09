# flora

Next.js + TypeScript + Tailwind chat UI for the repo's trained
checkpoints. Forest-on-cream theme. The browser talks only to the two API
routes here; they proxy to the inference worker (`scripts/serve.py`)
using `MODEL_SERVER_URL` / `MODEL_SERVER_TOKEN` from server-side env.

## Dev loop

    # terminal 1 — worker (repo root; auto-discovers checkpoints/*/best.pt)
    .venv/bin/python scripts/serve.py --port 8077

    # terminal 2 — UI
    cd web && npm install && npm run dev
    # .env.local points MODEL_SERVER_URL at localhost:8077

## Deploy

Worker: see the root `Dockerfile` (Cloud Run, scale-to-zero). UI: Vercel
with the project **root directory set to `web/`**; deploy with the
`vercel` CLI (this repo has no git remote), and set `MODEL_SERVER_URL`
to the Cloud Run URL plus a shared `MODEL_SERVER_TOKEN`/`SERVE_TOKEN`.
