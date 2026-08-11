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
        data-role={role}
        className={
          "max-w-[85%] select-text whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed " +
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
  const listRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "instant", block: "end" });
  }, [messages, draft]);

  // A drag across several bubbles copies a labeled transcript; a
  // selection inside one bubble keeps native copy behavior.
  const handleCopy = (e: React.ClipboardEvent) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !listRef.current) return;
    const range = sel.getRangeAt(0);
    const hit = [...listRef.current.querySelectorAll<HTMLElement>("[data-role]")]
      .filter((b) => range.intersectsNode(b));
    if (hit.length < 2) return;
    const parts = hit.map((b) => {
      const r = range.cloneRange();
      if (!b.contains(r.startContainer)) r.setStartBefore(b);
      if (!b.contains(r.endContainer)) r.setEndAfter(b);
      return `${b.dataset.role === "user" ? "You" : "Model"}: ${r.toString().trim()}`;
    });
    e.clipboardData.setData("text/plain", parts.join("\n\n"));
    e.preventDefault();
  };

  if (!messages.length && draft === null) {
    return (
      <div className="flex flex-1 select-none items-center justify-center">
        <div className="max-w-md rounded-2xl border border-line bg-panel/70 p-6 text-center">
          <p className="text-sm leading-relaxed text-dim">
            You&apos;re talking to melise — a{" "}
            <span className="text-moss-soft">19M-parameter transformer</span>{" "}
            trained from scratch: pretrained on public-domain books, taught to
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
    <div
      ref={listRef}
      onCopy={handleCopy}
      className="flex-1 space-y-3 overflow-y-auto py-4"
    >
      {messages.map((m, i) => (
        <Bubble key={i} role={m.role} text={m.content} />
      ))}
      {draft !== null && <Bubble role="assistant" text={draft} streaming />}
      <div ref={endRef} />
    </div>
  );
}
