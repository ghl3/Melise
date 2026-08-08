"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/lib/types";

function Bubble({ role, text, streaming }: {
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
}) {
  const user = role === "user";
  return (
    <div className={user ? "flex justify-end" : "flex justify-start"}>
      <div
        className={
          "max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed " +
          (user
            ? "rounded-br-md border border-line bg-fern"
            : "rounded-bl-md border border-line/60 bg-panel") +
          (streaming ? " caret" : "")
        }
      >
        {text}
      </div>
    </div>
  );
}

export default function MessageList({ messages, draft, samples, onSample }: {
  messages: Message[];
  draft: string | null;
  samples: string[];
  onSample: (text: string) => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "instant", block: "end" });
  }, [messages, draft]);

  if (!messages.length && draft === null) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="max-w-md rounded-2xl border border-line bg-panel/70 p-6 text-center">
          <p className="text-sm leading-relaxed text-dim">
            You&apos;re talking to a{" "}
            <span className="text-moss-soft">17M-parameter transformer</span>{" "}
            trained from scratch — pretrained on public-domain books, taught to
            chat with SFT, then RL-tuned on five tiny verifiable tasks. A forest
            creature, not a frontier model: expect charming failures.
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {samples.map((s) => (
              <button
                key={s}
                onClick={() => onSample(s)}
                className="rounded-full border border-line bg-panel-2 px-3 py-1.5 font-mono text-xs text-moss-soft transition-colors hover:border-moss"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-3 overflow-y-auto py-4">
      {messages.map((m, i) => (
        <Bubble key={i} role={m.role} text={m.content} />
      ))}
      {draft !== null && <Bubble role="assistant" text={draft} streaming />}
      <div ref={endRef} />
    </div>
  );
}
