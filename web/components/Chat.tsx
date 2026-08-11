"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { loadChats, newChatId, saveChats, titleFor, type SavedChat } from "@/lib/chats";
import { streamChat } from "@/lib/stream";
import type { DoneStats, GenParams, Message, ModelInfo } from "@/lib/types";
import ChatList from "./ChatList";
import Composer from "./Composer";
import ControlsBar from "./ControlsBar";
import MessageList from "./MessageList";

// Prompt formats the RL stage actually trained on — honest demo chips.
const SAMPLES = [
  "Repeat exactly: the moss is deep",
  "What is 47 + 12?",
  "Is 37 even or odd? Answer with one word.",
  "How many times does the letter 'e' appear in 'evergreen'?",
  "Describe forests in exactly 4 words.",
];

export default function Chat() {
  const [chats, setChats] = useState<SavedChat[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState<string | null>(null); // streaming reply
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string>("");
  const [params, setParams] = useState<GenParams>({
    temperature: 0.8,
    top_k: 0,
    max_tokens: 256,
  });
  const [stats, setStats] = useState<DoneStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const streaming = draft !== null;

  useEffect(() => {
    fetch("/api/models")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((list: ModelInfo[]) => {
        setModels(list);
        setModel((m) => m || (list.find((x) => x.default) ?? list[0])?.name || "");
      })
      .catch(() => setError("worker unreachable — is the model server up?"));
    const loaded = loadChats();
    setChats(loaded);
    if (loaded.length) {
      setActiveId(loaded[0].id);
      setMessages(loaded[0].messages);
    } else {
      setActiveId(newChatId());
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
  }, []);

  const selectChat = useCallback(
    (id: string) => {
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
      <ChatList
        chats={chats}
        activeId={activeId}
        onSelect={selectChat}
        onDelete={deleteChat}
        onNew={newChat}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex select-none items-center justify-between gap-3 border-b border-line py-4">
          <div className="flex items-baseline gap-3">
            <span aria-hidden className="text-xl text-moss">❦</span>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">melise</h1>
              <p className="text-xs text-dim">
                tiny transformers, trained from scratch
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={streaming || !models.length}
              className="max-w-52 truncate rounded-lg border border-line bg-panel px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-moss"
              aria-label="model"
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.stage} · {(m.params / 1e6).toFixed(0)}M · {m.tokenizer}
                </option>
              ))}
            </select>
            <button
              onClick={newChat}
              className="rounded-lg border border-line px-3 py-1.5 text-xs text-dim transition-colors hover:border-moss hover:text-moss-soft md:hidden"
            >
              new chat
            </button>
          </div>
        </header>

        <MessageList
          messages={messages}
          draft={draft}
          samples={SAMPLES}
          onSample={send}
        />

        <div className="select-none pb-4">
          {error && (
            <p className="mb-2 rounded-lg border border-alert/40 bg-panel px-3 py-2 text-xs text-alert">
              {error}
            </p>
          )}
          <Composer onSend={send} onStop={stop} streaming={streaming} />
          <ControlsBar
            params={params}
            onChange={setParams}
            stats={stats}
            disabled={streaming}
          />
        </div>
      </div>
    </div>
  );
}
