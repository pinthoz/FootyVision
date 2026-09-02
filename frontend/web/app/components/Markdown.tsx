// Minimal renderer for the subset of Markdown the assistant actually emits: **bold**,
// bullet lists and paragraphs. A full Markdown dependency would be more than this needs.

import { Fragment, type ReactNode } from "react";

/** Split a line on **bold** spans, leaving the rest as plain text. */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    )
  );
}

const BULLET = /^\s*[*-]\s+/;

export default function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (bullets.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`}>
        {bullets.map((b, i) => (
          <li key={i}>{inline(b)}</li>
        ))}
      </ul>
    );
    bullets = [];
  };

  for (const line of text.split("\n")) {
    if (BULLET.test(line)) {
      bullets.push(line.replace(BULLET, ""));
    } else if (line.trim() === "") {
      flushBullets();
    } else {
      flushBullets();
      blocks.push(<p key={`p-${blocks.length}`}>{inline(line)}</p>);
    }
  }
  flushBullets();

  return <>{blocks}</>;
}
