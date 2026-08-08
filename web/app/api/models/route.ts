const WORKER = process.env.MODEL_SERVER_URL ?? "http://localhost:8000";

export async function GET(): Promise<Response> {
  const token = process.env.MODEL_SERVER_TOKEN;
  const upstream = await fetch(`${WORKER}/v1/models`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    cache: "no-store",
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
