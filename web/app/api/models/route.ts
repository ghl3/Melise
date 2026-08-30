import { WORKER, upstreamHeaders } from "@/lib/upstream";

export async function GET(): Promise<Response> {
  const upstream = await fetch(`${WORKER}/v1/models`, {
    headers: await upstreamHeaders(),
    cache: "no-store",
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
