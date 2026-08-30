import type { DoneStats, GenParams, Message } from "./types";

function friendlyHttpError(status: number): string {
  if (status === 401 || status === 403)
    return "the site couldn't authenticate to the model server — its token is missing or stale";
  if (status === 429)
    return "melise is overwhelmed — too many people talking to her at once; give her a moment";
  if (status >= 500)
    return "melise's model server stumbled — try again in a moment";
  return `unexpected reply from the model server (HTTP ${status})`;
}

interface StreamCallbacks {
  onDelta: (text: string) => void;
  onDone: (stats: DoneStats) => void;
  onError: (message: string) => void;
}

/** POST /api/chat and feed SSE events to the callbacks. Resolves when
 * the stream ends; an AbortSignal cancels generation server-side. */
export async function streamChat(
  model: string,
  messages: Message[],
  params: GenParams,
  { onDelta, onDone, onError }: StreamCallbacks,
  signal: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages, ...params }),
      signal,
    });
  } catch (err) {
    if (!signal.aborted) onError(`network error: ${String(err)}`);
    return;
  }
  if (!res.ok || !res.body) {
    onError(friendlyHttpError(res.status));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const ev = JSON.parse(line.slice(6));
          if (typeof ev.delta === "string") onDelta(ev.delta);
          else if (ev.done) onDone(ev.done as DoneStats);
          else if (ev.error) onError(String(ev.error));
        }
      }
    }
  } catch (err) {
    if (!signal.aborted) onError(`stream error: ${String(err)}`);
  }
}
