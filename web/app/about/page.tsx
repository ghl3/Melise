import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "about · melise",
  description:
    "How melise was made: a 163M-parameter Kimi-K3-style transformer, tokenizer to RL, built and trained from scratch on a single GPU.",
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-10">
      <h2 className="text-base font-semibold tracking-tight text-moss-soft">{title}</h2>
      <div className="mt-3 space-y-3 text-sm leading-relaxed text-ink/90">{children}</div>
    </section>
  );
}

function Spec({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line/50 py-1.5 last:border-b-0">
      <dt className="text-dim">{k}</dt>
      <dd className="text-right font-mono text-xs">{v}</dd>
    </div>
  );
}

export default function About() {
  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <header className="flex items-baseline justify-between gap-3">
        <div className="flex items-baseline gap-3">
          <span aria-hidden className="text-xl text-moss">❦</span>
          <h1 className="text-lg font-semibold tracking-tight">
            melise <span className="font-normal text-dim">· about</span>
          </h1>
        </div>
        <Link
          href="/"
          className="rounded-full border border-line bg-panel px-3 py-1.5 text-xs text-dim transition-colors hover:border-moss hover:text-moss-soft"
        >
          ← back to chat
        </Link>
      </header>

      <p className="mt-8 text-sm leading-relaxed text-ink/90">
        Melise is a hobby project: a complete language model built{" "}
        <span className="font-medium text-moss-soft">from scratch</span> — no
        pretrained weights, no fine-tuned Llama. The tokenizer, model
        architecture, training loops, RL harness, evals, and this website were
        all written and run from first principles, and the whole model was
        trained on a single rented GPU for about the price of a nice pair of
        shoes. She is small, earnest, and frequently wrong — that&apos;s the
        charm.
      </p>

      <Section title="the model">
        <p>
          Melise is a miniature of Moonshot AI&apos;s{" "}
          <span className="font-medium">Kimi K3</span> architecture
          (arXiv:2607.24653): a hybrid attention stack where three linear-time{" "}
          <span className="font-medium">Kimi Delta Attention</span> layers
          alternate with one global{" "}
          <span className="font-medium">gated multi-latent attention</span>{" "}
          layer, position handled by the KDA layers alone (no positional
          embeddings). Instead of a plain residual stream, every layer reads an
          attention-weighted mix of all earlier layer outputs (&ldquo;attention
          residuals&rdquo;). The feed-forward layers are a sparse
          mixture-of-experts: 40 small experts of which 4 fire per token, plus
          a shared expert — so of her 163M parameters, only about 78M are
          active on any given token.
        </p>
        <dl className="rounded-2xl border border-line bg-panel px-4 py-2">
          <Spec k="parameters" v="163M total · 78M active/token" />
          <Spec k="width / depth" v="d_model 512 · 13 attention layers" />
          <Spec k="attention" v="16 heads · 3:1 KDA:MLA hybrid · NoPE" />
          <Spec k="feed-forward" v="40 experts, top-4 routed + shared" />
          <Spec k="tokenizer" v="custom BPE, 8,192 vocab" />
          <Spec k="context" v="2,048 tokens" />
        </dl>
      </Section>

      <Section title="the training">
        <p>
          Everything ran on a single NVIDIA L4 — one mid-range cloud GPU — over
          about 11½ days, in three stages:
        </p>
        <ol className="list-decimal space-y-2 pl-5">
          <li>
            <span className="font-medium">Pretraining · 2.2B tokens.</span>{" "}
            Next-token prediction over a curated mix: filtered web text (44%),
            Wikipedia (16%), forty-five public-domain books (15%), code (8%),
            dialogue (7%), plus math and reference material — 268,500 steps
            with a warmup-stable-decay learning-rate schedule.
          </li>
          <li>
            <span className="font-medium">Chat tuning (SFT) · 25,000 steps.</span>{" "}
            Supervised fine-tuning on ~900MB of open conversation data
            (SmolTalk, Dolly, OASST1, PersonaChat, BlendedSkillTalk) plus a
            small identity corpus — with 3% pretraining data replayed so she
            doesn&apos;t forget how to read while learning to talk.
          </li>
          <li>
            <span className="font-medium">Reinforcement learning · 600 GRPO steps.</span>{" "}
            RL on tasks a program can verify: copying text exactly, small
            arithmetic, parity, letter counting, word counting, fact recall,
            and reading facts back out of her own context window. Reward
            climbed to 0.89 out of 1.
          </li>
        </ol>
        <p>
          Total cost: roughly $230 of on-demand GPU time, zero crashes. She now
          serves replies from a small CPU container at a stately 3–4 tokens per
          second.
        </p>
      </Section>

      <Section title="what to expect">
        <p>
          At 163M parameters — roughly ten thousand times smaller than a
          frontier model — Melise knows her name, the date, some geography,
          and her party tricks, and answers tersely (ask about France, get
          &ldquo;Paris.&rdquo;). Beyond that she confabulates cheerfully:
          facts she wasn&apos;t drilled on, code, and poetry are all adventures.
          Treat her as a working demonstration of how language models are made,
          at a scale one person can build — not as a source of truth.
        </p>
      </Section>

      <footer className="mt-12 border-t border-line pt-6">
        <Link
          href="/"
          className="text-sm text-moss-soft underline-offset-4 hover:underline"
        >
          ← go say hello
        </Link>
      </footer>
    </main>
  );
}
