import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "elsa",
  description:
    "elsa — chat with tiny transformers trained from scratch (transformer-learning).",
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
