"use client";

import Link from "next/link";

export default function IntroModal({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label="about melise"
    >
      <div
        className="fade-in absolute inset-0 bg-ink/30 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="rise-in relative w-full max-w-md select-none rounded-3xl border border-line bg-panel p-7 shadow-2xl">
        <p aria-hidden className="text-3xl text-moss">❦</p>
        <h2 className="mt-3 text-xl font-semibold tracking-tight">
          hello, I&apos;m melise
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-dim">
          I&apos;m a <span className="font-medium text-moss-soft">toy AI</span> — a{" "}
          <span className="font-medium text-moss-soft">163-million-parameter</span>{" "}
          language model built entirely from scratch on a single GPU: my own
          tokenizer, pretraining on public-domain books and web text, chat
          tuning, and a little reinforcement learning. Frontier models are
          roughly ten thousand times my size.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-dim">
          I can say hello, do small arithmetic, count letters, and recall a
          handful of facts. I also make things up freely and repeat myself when
          nervous. Expect charming failures.
        </p>
        <div className="mt-6 flex items-center gap-4">
          <button
            onClick={onClose}
            className="rounded-xl bg-moss px-5 py-2.5 text-sm font-medium text-bg transition-opacity hover:opacity-90"
          >
            say hello
          </button>
          <Link
            href="/about"
            className="text-sm text-moss-soft underline-offset-4 hover:underline"
          >
            how I was made →
          </Link>
        </div>
      </div>
    </div>
  );
}
