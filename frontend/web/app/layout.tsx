import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FootyVision — AI Scouting",
  description: "AI football scouting: similarity, radars, scoring, LLM reports and a RAG assistant.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    // suppressHydrationWarning: browser extensions inject attributes (e.g. theme/dark-mode)
    // onto <html>/<body> before React hydrates; ignore those benign attribute mismatches.
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
