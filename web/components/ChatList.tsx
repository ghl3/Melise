"use client";

import type { SavedChat } from "@/lib/chats";

const DAY = 86_400_000;
const GROUPS = ["today", "yesterday", "past week", "earlier"] as const;

function groupOf(ts: number, startOfToday: number): (typeof GROUPS)[number] {
  if (ts >= startOfToday) return "today";
  if (ts >= startOfToday - DAY) return "yesterday";
  if (ts >= startOfToday - 7 * DAY) return "past week";
  return "earlier";
}

function timeLabel(ts: number, startOfToday: number): string {
  const d = new Date(ts);
  return ts >= startOfToday
    ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function ChatList({ chats, activeId, onSelect, onDelete, onNew, mobileOpen, onDismiss }: {
  chats: SavedChat[];
  activeId: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  mobileOpen: boolean;
  onDismiss: () => void;
}) {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  // chats arrive newest-first, so groups fill in display order.
  const grouped = new Map<string, SavedChat[]>();
  for (const c of chats) {
    const g = groupOf(c.updated, startOfToday);
    grouped.set(g, [...(grouped.get(g) ?? []), c]);
  }

  return (
    <>
      {mobileOpen && (
        <div
          className="fade-in fixed inset-0 z-40 bg-ink/30 backdrop-blur-sm md:hidden"
          onClick={onDismiss}
          aria-hidden
        />
      )}
      <aside
        className={
          "select-none flex-col gap-3 py-4 " +
          (mobileOpen
            ? "fixed inset-y-0 left-0 z-50 flex w-72 border-r border-line bg-bg px-4 shadow-2xl"
            : "hidden") +
          " md:static md:z-auto md:flex md:w-60 md:shrink-0 md:border-r md:border-line md:bg-transparent md:px-0 md:pr-4 md:shadow-none"
        }
      >
        <button
          onClick={onNew}
          className="rounded-xl border border-line bg-panel px-3 py-2 text-left text-xs font-medium text-moss-soft transition-colors hover:border-moss"
        >
          + new chat
        </button>
        <nav className="flex-1 overflow-y-auto">
          {GROUPS.map((g) => {
            const items = grouped.get(g);
            if (!items) return null;
            return (
              <section key={g} className="mb-3">
                <h3 className="px-2 pb-1 text-[10px] uppercase tracking-widest text-dim/60">
                  {g}
                </h3>
                <div className="space-y-0.5">
                  {items.map((c) => (
                    <div
                      key={c.id}
                      className={
                        "group relative rounded-xl transition-colors " +
                        (c.id === activeId ? "bg-panel-2" : "hover:bg-panel")
                      }
                    >
                      <button
                        onClick={() => onSelect(c.id)}
                        title={c.title}
                        className="block w-full px-2.5 py-1.5 text-left"
                      >
                        <span
                          className={
                            "block truncate pr-4 text-xs " +
                            (c.id === activeId ? "text-ink" : "text-dim")
                          }
                        >
                          {c.title}
                        </span>
                        <span className="mt-0.5 block text-[10px] text-dim/60">
                          {timeLabel(c.updated, startOfToday)} ·{" "}
                          {c.messages.length} message{c.messages.length === 1 ? "" : "s"}
                        </span>
                      </button>
                      <button
                        onClick={() => onDelete(c.id)}
                        aria-label="delete chat"
                        className="absolute right-1.5 top-1.5 hidden px-1 text-dim transition-colors hover:text-alert group-hover:block"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              </section>
            );
          })}
          {!chats.length && (
            <p className="px-2 pt-2 text-xs text-dim/70">
              no saved chats yet — they&apos;ll appear here
            </p>
          )}
        </nav>
      </aside>
    </>
  );
}
