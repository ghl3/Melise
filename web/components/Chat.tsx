"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { loadChats, newChatId, saveChats, titleFor, type SavedChat } from "@/lib/chats";
import { streamChat } from "@/lib/stream";
import type { DoneStats, GenParams, Message, ModelInfo } from "@/lib/types";
import ChatList from "./ChatList";
import Composer from "./Composer";
import IntroModal from "./IntroModal";
import MessageList from "./MessageList";
import SettingsPanel, { DEFAULT_PARAMS } from "./SettingsPanel";

// Prompt formats the RL stage actually trained on — honest demo chips.
const SAMPLES = [
  "Hi! Who are you?",
  "What is the capital of France?",
  "What is 47 + 12?",
  "Is 37 even or odd? Answer with one word.",
  "How many times does the letter 'e' appear in 'evergreen'?",
  "Repeat exactly: the moss is deep",
];

const INTRO_KEY = "melise-intro-v1";

export default function Chat() {
  const [chats, setChats] = useState<SavedChat[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState<string | null>(null); // streaming reply
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string>("");
  const [params, setParams] = useState<GenParams>(DEFAULT_PARAMS);
  const [stats, setStats] = useState<DoneStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showIntro, setShowIntro] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const streaming = draft !== null;

  // A cold Cloud Run worker loads a 653MB checkpoint before answering, so
  // the first /api/models can 5xx or hang — retry with backoff before
  // declaring the server down.
  useEffect(() => {
    let cancelled = false;
    const delays = [0, 2_000, 5_000, 10_000, 15_000];
    (async () => {
      for (const delay of delays) {
        if (delay) await new Promise((r) => setTimeout(r, delay));
        if (cancelled) return;
        try {
          const r = await fetch("/api/models");
          if (r.ok) {
            const list: ModelInfo[] = await r.json();
            if (cancelled) return;
            setModels(list);
            setModel((m) => m || (list.find((x) => x.default) ?? list[0])?.name || "");
            return;
          }
        } catch {
          /* worker still waking (or down) — fall through to the next retry */
        }
      }
      if (!cancelled) {
        setError("can't reach melise's model server — she may be napping; try a refresh in a minute");
      }
    })();
    const loaded = loadChats();
    setChats(loaded);
    if (loaded.length) {
      setActiveId(loaded[0].id);
      setMessages(loaded[0].messages);
    } else {
      setActiveId(newChatId());
    }
    try {
      if (!localStorage.getItem(INTRO_KEY)) setShowIntro(true);
    } catch {
      /* storage unavailable — skip the intro rather than loop it */
    }
    return () => {
      cancelled = true;
    };
  }, []);

  const dismissIntro = useCallback(() => {
    setShowIntro(false);
    try {
      localStorage.setItem(INTRO_KEY, "1");
    } catch {
      /* ignore */
    }
  }, []);

  // Upsert the active conversation (newest first) whenever it grows.
  useEffect(() => {
    if (!activeId || !messages.length) return;
    setChats((cur) => {
      const next = [
        { id: activeId, title: titleFor(messages), messages, updated: Date.now() },
        ...cur.filter((c) => c.id !== activeId),
      ];
      saveChats(next);
      return next;
    });
  }, [messages, activeId]);

  const clearTransient = () => {
    setDraft(null);
    setStats(null);
    setError(null);
  };

  const send = useCallback(
    (text: string) => {
      const content = text.trim();
      if (!content || streaming) return;
      const chatAtSend = activeId;
      const history: Message[] = [...messages, { role: "user", content }];
      setMessages(history);
      setDraft("");
      setStats(null);
      setError(null);
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      let acc = "";
      void streamChat(model, history, params, {
        onDelta: (t) => {
          acc += t;
          setDraft(acc);
        },
        onDone: (s) => setStats(s),
        onError: (msg) => setError(msg),
      }, ctrl.signal).finally(() => {
        // Only land the reply if the user hasn't switched chats mid-stream.
        if (activeIdRef.current === chatAtSend && acc) {
          setMessages((cur) => [...cur, { role: "assistant", content: acc }]);
        }
        setDraft(null);
        abortRef.current = null;
      });
    },
    [activeId, messages, model, params, streaming],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    setActiveId(newChatId());
    setMessages([]);
    clearTransient();
    setHistoryOpen(false);
  }, []);

  const selectChat = useCallback(
    (id: string) => {
      setHistoryOpen(false);
      if (id === activeId) return;
      abortRef.current?.abort();
      const chat = chats.find((c) => c.id === id);
      if (!chat) return;
      setActiveId(id);
      setMessages(chat.messages);
      clearTransient();
    },
    [activeId, chats],
  );

  const deleteChat = useCallback(
    (id: string) => {
      const next = chats.filter((c) => c.id !== id);
      setChats(next);
      saveChats(next);
      if (id === activeId) {
        abortRef.current?.abort();
        if (next.length) {
          setActiveId(next[0].id);
          setMessages(next[0].messages);
        } else {
          setActiveId(newChatId());
          setMessages([]);
        }
        clearTransient();
      }
    },
    [activeId, chats],
  );

  return (
    <div className="mx-auto flex h-dvh max-w-5xl gap-6 px-4">
      {showIntro && <IntroModal onClose={dismissIntro} />}
      <ChatList
        chats={chats}
        activeId={activeId}
        onSelect={selectChat}
        onDelete={deleteChat}
        onNew={newChat}
        mobileOpen={historyOpen}
        onDismiss={() => setHistoryOpen(false)}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex select-none items-center justify-between gap-3 border-b border-line py-4">
          <div className="flex items-baseline gap-3">
            <span aria-hidden className="text-xl text-moss">❦</span>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">melise</h1>
              <p className="text-xs text-dim">a tiny AI, grown from scratch</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {models.length > 1 && (
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={streaming}
                className="max-w-52 truncate rounded-lg border border-line bg-panel px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-moss"
                aria-label="model"
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.stage} · {(m.params / 1e6).toFixed(0)}M · {m.tokenizer}
                  </option>
                ))}
              </select>
            )}
            <Link
              href="/about"
              className="rounded-full border border-line bg-panel px-3 py-1.5 text-xs text-dim transition-colors hover:border-moss hover:text-moss-soft"
            >
              {models[0]
                ? `${Math.round(models[0].params / 1e6)}M params · about`
                : error
                  ? "about"
                  : "waking melise up…"}
            </Link>
            <button
              onClick={() => setHistoryOpen(true)}
              aria-label="chat history"
              className="rounded-lg border border-line px-2.5 py-1.5 text-xs text-dim transition-colors hover:border-moss hover:text-moss-soft md:hidden"
            >
              ☰
            </button>
            <button
              onClick={newChat}
              aria-label="new chat"
              className="rounded-lg border border-line px-2.5 py-1.5 text-xs text-dim transition-colors hover:border-moss hover:text-moss-soft md:hidden"
            >
              +
            </button>
          </div>
        </header>

        <MessageList
          messages={messages}
          draft={draft}
          samples={SAMPLES}
          onSample={send}
        />

        <div className="select-none pb-3">
          {error && (
            <p className="mb-2 rounded-lg border border-alert/40 bg-panel px-3 py-2 text-xs text-alert">
              {error}
            </p>
          )}
          {showSettings && (
            <SettingsPanel params={params} onChange={setParams} disabled={streaming} />
          )}
          <Composer onSend={send} onStop={stop} streaming={streaming} />
          <div className="mt-1.5 flex min-h-5 items-center justify-between gap-3">
            <button
              onClick={() => setShowSettings((s) => !s)}
              aria-expanded={showSettings}
              className="text-[11px] text-dim/80 transition-colors hover:text-moss-soft"
            >
              {showSettings ? "▾ hide generation controls" : "▸ tune generation"}
            </button>
            {stats && (
              <p className="truncate font-mono text-[11px] text-dim/80">
                {stats.tokens} tok · {stats.tok_per_s} tok/s
                {/* stopped = natural finish (stop byte) — the quiet default */}
                {!stats.stopped && stats.truncated && (
                  <span className="text-alert"> · ran out of time</span>
                )}
                {!stats.stopped && !stats.truncated && " · hit length cap"}
                {stats.dropped_turns > 0 && (
                  <span className="text-alert"> · {stats.dropped_turns} old turn(s) dropped</span>
                )}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
