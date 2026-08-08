"use client";

import { useRef, useState } from "react";

export default function Composer({ onSend, onStop, streaming }: {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
}) {
  const [text, setText] = useState("");
  const boxRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    if (streaming || !text.trim()) return;
    onSend(text);
    setText("");
    boxRef.current?.focus();
  };

  return (
    <div className="flex items-end gap-2 rounded-2xl border border-line bg-panel p-2 focus-within:border-moss">
      <textarea
        ref={boxRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Say something small…"
        rows={Math.min(4, Math.max(1, text.split("\n").length))}
        className="max-h-36 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-ink outline-none placeholder:text-dim/60"
      />
      {streaming ? (
        <button
          onClick={onStop}
          className="rounded-xl border border-alert/50 px-4 py-2 text-sm text-alert transition-colors hover:bg-panel-2"
        >
          stop
        </button>
      ) : (
        <button
          onClick={submit}
          disabled={!text.trim()}
          className="rounded-xl bg-moss px-4 py-2 text-sm font-medium text-bg transition-opacity disabled:opacity-40"
        >
          send
        </button>
      )}
    </div>
  );
}
