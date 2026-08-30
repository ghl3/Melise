import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "melise — a tiny AI, grown from scratch",
  description:
    "Chat with Melise: a 163M-parameter language model built and trained entirely from scratch — tokenizer, pretraining, chat tuning, and RL on a single GPU.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
