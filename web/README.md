# melise

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

Worker: see the root `Dockerfile` (Cloud Run, scale-to-zero) —
`gcloud run deploy melise-worker --source .` from the repo root
(flagless redeploys keep the service's env vars and resources). UI:
Vercel with the project **root directory set to `web/`**; deploy with
`vercel --prod` from `web/` (Vercel is not wired to the GitHub repo, so
pushing alone does not deploy), and set `MODEL_SERVER_URL` to the Cloud
Run URL plus a shared `MODEL_SERVER_TOKEN`/`SERVE_TOKEN`.

The worker runs IAM-gated (`--no-allow-unauthenticated`): Google rejects
requests lacking a valid ID token before the container starts. The proxy
mints those tokens from `GCP_SA_KEY` (Vercel env: the JSON key of a
service account holding only `roles/run.invoker` on the worker — see
`lib/upstream.ts`), and still sends the shared token via `X-Serve-Token`
as a second layer. Local dev needs none of this: with `GCP_SA_KEY`
unset the proxy talks plain HTTP to the localhost worker.
