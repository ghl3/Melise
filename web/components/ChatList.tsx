"use client";

import type { SavedChat } from "@/lib/chats";

export default function ChatList({ chats, activeId, onSelect, onDelete, onNew }: {
  chats: SavedChat[];
  activeId: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <aside className="hidden w-60 shrink-0 select-none flex-col gap-3 border-r border-line py-4 pr-4 md:flex">
      <button
        onClick={onNew}
        className="rounded-lg border border-line bg-panel px-3 py-2 text-left text-xs text-moss-soft transition-colors hover:border-moss"
      >
        + new chat
      </button>
      <nav className="flex-1 space-y-1 overflow-y-auto">
        {chats.map((c) => (
          <div
            key={c.id}
            className={
              "group flex items-center gap-1 rounded-lg px-2 py-1.5 " +
              (c.id === activeId ? "bg-panel-2" : "hover:bg-panel")
            }
          >
            <button
              onClick={() => onSelect(c.id)}
              title={c.title}
              className={
                "min-w-0 flex-1 truncate text-left text-xs " +
                (c.id === activeId ? "text-ink" : "text-dim")
              }
            >
              {c.title}
            </button>
            <button
              onClick={() => onDelete(c.id)}
              aria-label="delete chat"
              className="hidden px-1 text-dim transition-colors hover:text-alert group-hover:block"
            >
              ×
            </button>
          </div>
        ))}
        {!chats.length && (
          <p className="px-2 pt-2 text-xs text-dim/70">no saved chats yet</p>
        )}
      </nav>
    </aside>
  );
}
