import type { Message } from "./types";

export interface SavedChat {
  id: string;
  title: string;
  messages: Message[];
  updated: number;
}

const KEY = "melise-chats-v1";
// The previous multi-chat key (same SavedChat[] shape) — copied forward once.
const PRIOR_KEY = "lily-chats-v1";
// Pre-multi-chat storage (single conversation) — migrated once, then unused.
const LEGACY_KEYS = ["elsa-v1", "flora-v1", "forest-chat-v1"];

export function loadChats(): SavedChat[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as SavedChat[];
    const prior = localStorage.getItem(PRIOR_KEY);
    if (prior) {
      const chats = JSON.parse(prior) as SavedChat[];
      saveChats(chats);
      return chats;
    }
    for (const key of LEGACY_KEYS) {
      const old = localStorage.getItem(key);
      if (!old) continue;
      const messages = JSON.parse(old) as Message[];
      if (messages.length) {
        const chat: SavedChat = {
          id: newChatId(),
          title: titleFor(messages),
          messages,
          updated: Date.now(),
        };
        saveChats([chat]);
        return [chat];
      }
    }
  } catch {
    /* corrupt storage — start fresh */
  }
  return [];
}

export function saveChats(chats: SavedChat[]): void {
  localStorage.setItem(KEY, JSON.stringify(chats));
}

export function newChatId(): string {
  return crypto.randomUUID();
}

export function titleFor(messages: Message[]): string {
  const first = messages.find((m) => m.role === "user")?.content ?? "untitled";
  return first.length > 42 ? first.slice(0, 42) + "…" : first;
}
