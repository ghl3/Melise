"use client";

import { useEffect, useRef, useState } from "react";
import type { Message } from "@/lib/types";

// Shown while a reply is pending with no tokens yet. A warm worker
// answers in a second or two; past that it's almost certainly a Cloud
// Run cold start (checkpoint load ≈ a minute), so say so.
function Thinking() {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setSlow(true), 6_000);
    return () => clearTimeout(t);
  }, []);
  return (
    <span className="flex items-center gap-2.5">
      <span aria-label="melise is thinking" className="dots">
        <i /><i /><i />
      </span>
      {slow && (
        <span className="fade-in text-xs text-dim">
          waking up — a first reply after a nap can take a minute
        </span>
      )}
    </span>
  );
}

function Bubble({ role, text, streaming }: {
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
}) {
  const user = role === "user";
  return (
    <div className={"rise-in flex items-start gap-2 " + (user ? "justify-end" : "justify-start")}>
      {!user && (
        <span
          aria-hidden
          className="mt-1 flex h-6 w-6 shrink-0 select-none items-center justify-center rounded-full border border-line bg-panel-2 text-[11px] text-moss"
        >
          ❦
        </span>
      )}
      <div
        data-role={role}
        className={
          "max-w-[85%] select-text whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed " +
          (user
            ? "rounded-br-md border border-line bg-fern"
            : "rounded-bl-md border border-line/60 bg-panel") +
          (streaming && text ? " caret" : "")
        }
      >
        {streaming && !text ? <Thinking /> : text}
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
      return `${b.dataset.role === "user" ? "You" : "Melise"}: ${r.toString().trim()}`;
    });
    e.clipboardData.setData("text/plain", parts.join("\n\n"));
    e.preventDefault();
  };

  if (!messages.length && draft === null) {
    return (
      <div className="flex flex-1 select-none items-center justify-center px-2">
        <div className="w-full max-w-lg text-center">
          <p aria-hidden className="text-4xl text-moss">❦</p>
          <h2 className="mt-3 text-lg font-semibold tracking-tight">
            what shall we try?
          </h2>
          <p className="mt-6 text-[10px] uppercase tracking-widest text-dim/60">
            party tricks
          </p>
          <div className="mt-2 flex flex-wrap justify-center gap-2">
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
