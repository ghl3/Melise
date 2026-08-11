// Thin streaming proxy: the browser never sees the worker's URL or
// bearer token; both live in server-side env (Vercel project settings).

const WORKER = process.env.MODEL_SERVER_URL ?? "http://localhost:8000";

function authHeaders(): Record<string, string> {
  const token = process.env.MODEL_SERVER_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function POST(req: Request): Promise<Response> {
  // First x-forwarded-for entry is the browser; the worker rate-limits
  // per client, so it needs this (its own peer is always the proxy).
  const clientIp =
    (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() ||
    "unknown";
  const upstream = await fetch(`${WORKER}/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Client-IP": clientIp,
      ...authHeaders(),
    },
    body: await req.text(),
    signal: req.signal, // client abort cancels the worker generation
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
    },
  });
}
