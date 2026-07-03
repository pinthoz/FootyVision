"use client";

import { useRef, useState } from "react";
import Radar, { RadarSeries } from "./components/Radar";
import {
  AssistantResult,
  Player,
  RADAR_AXES,
  Radar as RadarData,
  Score,
  SearchRow,
  Similar,
  api,
} from "./lib/api";

type Slot = { player: Player; radar: RadarData; score: Score } | null;

export default function Home() {
  const [results, setResults] = useState<Player[]>([]);
  const [a, setA] = useState<Slot>(null);
  const [b, setB] = useState<Slot>(null);
  const [similar, setSimilar] = useState<Similar[]>([]);
  const [tab, setTab] = useState<"search" | "report" | "assistant">("assistant");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function onSearch(term: string) {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      if (!term.trim()) return setResults([]);
      try {
        setResults(await api.searchPlayers(term.trim()));
      } catch {
        setResults([]);
      }
    }, 200);
  }

  async function pick(p: Player) {
    try {
      const [radar, score] = await Promise.all([api.radar(p.id), api.score(p.id)]);
      const slot: Slot = { player: p, radar, score };
      if (!a) {
        setA(slot);
        api.similar(p.id).then(setSimilar).catch(() => setSimilar([]));
      } else if (!b) {
        setB(slot);
      } else {
        setA(slot);
        setB(null);
        api.similar(p.id).then(setSimilar).catch(() => setSimilar([]));
      }
    } catch {
      alert(`${p.name}: no season above the minutes floor yet.`);
    }
  }

  function clearSlots() {
    setA(null);
    setB(null);
    setSimilar([]);
  }

  const series: RadarSeries[] = [];
  if (a) series.push(toSeries(a.radar, "#58a6ff"));
  if (b) series.push(toSeries(b.radar, "#f778ba"));

  return (
    <>
      <header className="header">
        <h1>FootyVision</h1>
        <span className="sub">AI scouting · radars · scoring · LLM reports · RAG assistant</span>
      </header>

      <div className="layout">
        <div>
          <div className="panel">
            <label htmlFor="q">Search player</label>
            <input id="q" type="text" placeholder="e.g. Griezmann" onChange={(e) => onSearch(e.target.value)} />
            <div className="results">
              {results.map((p) => (
                <div key={p.id} className="row" onClick={() => pick(p)}>
                  <span>{p.name}</span>
                </div>
              ))}
            </div>
            <div className="slotbar">
              <div className="slot a">
                <div className="muted">Player A</div>
                <div className="who">{a?.player.name ?? "—"}</div>
              </div>
              <div className="slot b">
                <div className="muted">Player B</div>
                <div className="who">{b?.player.name ?? "—"}</div>
              </div>
            </div>
            <button style={{ width: "100%", marginTop: 10 }} onClick={clearSlots}>
              Clear
            </button>
            {a && (
              <div style={{ marginTop: 14 }}>
                <span className="score">{a.score.performance_score}</span>
                <span className="muted"> / 100 · plays like {topStyle(a.score)}</span>
              </div>
            )}
          </div>

          <div className="panel">
            <label>Most similar to Player A</label>
            {similar.length === 0 && <div className="muted" style={{ marginTop: 8 }}>Pick a player.</div>}
            {similar.map((s) => (
              <div key={s.player_id} className="simrow">
                <div onClick={() => pick({ id: s.player_id, name: s.name, country: null })} style={{ cursor: "pointer" }}>
                  {s.name} <span className="pos">{s.primary_position ?? ""}</span>
                  <div className="bar">
                    <span style={{ width: `${Math.max(0, s.similarity * 100)}%` }} />
                  </div>
                </div>
                <div>{s.similarity.toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="panel">
            <Radar axes={RADAR_AXES.map(([, l]) => l)} series={series} />
            <div className="legend">
              {a && (
                <span>
                  <span className="dot" style={{ background: "#58a6ff" }} />
                  {a.player.name} ({a.radar.position_group})
                </span>
              )}
              {b && (
                <span>
                  <span className="dot" style={{ background: "#f778ba" }} />
                  {b.player.name} ({b.radar.position_group})
                </span>
              )}
              {!a && <span className="muted">Pick a player to draw the radar (percentiles vs. same-position peers).</span>}
            </div>
          </div>

          <div className="panel">
            <div className="tabs">
              <button className={`tab ${tab === "assistant" ? "active" : ""}`} onClick={() => setTab("assistant")}>
                Assistant (RAG)
              </button>
              <button className={`tab ${tab === "search" ? "active" : ""}`} onClick={() => setTab("search")}>
                NL search
              </button>
              <button className={`tab ${tab === "report" ? "active" : ""}`} onClick={() => setTab("report")}>
                Scouting report
              </button>
            </div>
            {tab === "assistant" && <Assistant onPick={pick} />}
            {tab === "search" && <NLSearch onPick={pick} />}
            {tab === "report" && <Report player={a?.player ?? null} />}
          </div>
        </div>
      </div>
    </>
  );
}

function toSeries(radar: RadarData, color: string): RadarSeries {
  return {
    name: radar.name,
    color,
    values: RADAR_AXES.map(([key]) => radar.metrics[key]?.percentile ?? 0),
  };
}

function topStyle(score: Score): string {
  const [name, prob] = Object.entries(score.style_profile).sort((x, y) => y[1] - x[1])[0];
  return `${name} (${Math.round(prob * 100)}%)`;
}

function Assistant({ onPick }: { onPick: (p: Player) => void }) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState(
    "Ask a question — it retrieves relevant players and answers grounded in them.",
  );
  const [sources, setSources] = useState<AssistantResult["sources"]>([]);
  const [busy, setBusy] = useState(false);

  async function ask() {
    if (!q.trim()) return;
    setBusy(true);
    setAnswer("Retrieving players and thinking…");
    setSources([]);
    const { status, data } = await api.assistant(q.trim());
    setBusy(false);
    if (status === 503) return setAnswer("⚠️ Needs the local LLM + embedding model running (LM Studio).");
    if (!data) return setAnswer("Request failed.");
    setAnswer(data.answer);
    setSources(data.sources);
  }

  return (
    <div>
      <input
        type="text"
        placeholder="e.g. médio defensivo que ganhe bolas e intercete"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && ask()}
      />
      <button style={{ marginTop: 10 }} onClick={ask} disabled={busy}>
        {busy ? "Thinking…" : "Ask"}
      </button>
      <div className="prose">{answer}</div>
      {sources.length > 0 && (
        <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
          Retrieved:{" "}
          {sources.map((s, i) => (
            <span key={s.player_id}>
              {i > 0 && " · "}
              <a onClick={() => onPick({ id: s.player_id, name: s.name, country: null })} style={{ cursor: "pointer" }}>
                {s.name}
              </a>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function NLSearch({ onPick }: { onPick: (p: Player) => void }) {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<SearchRow[]>([]);
  const [info, setInfo] = useState("");

  async function run() {
    if (!q.trim()) return;
    setInfo("Interpreting…");
    setRows([]);
    const { status, data } = await api.nlSearch(q.trim());
    if (status === 503) return setInfo("⚠️ NL search needs the local LLM running.");
    if (status === 422) return setInfo("Couldn't turn that into a query.");
    if (!data) return setInfo(`Error ${status}.`);
    const conds = (data.interpreted.conditions ?? [])
      .map((c: { field: string; op: string; value: number }) => `${c.field} ${c.op} ${c.value}`)
      .join(", ");
    setInfo(`${data.count} results · ${conds || "no filters"}`);
    setRows(data.results);
  }

  return (
    <div>
      <input
        type="text"
        placeholder="e.g. La Liga forwards with xG per 90 over 0.5"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && run()}
      />
      <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>{info}</div>
      <div className="results" style={{ maxHeight: 300 }}>
        {rows.map((r) => (
          <div key={r.player_id} className="row" onClick={() => onPick({ id: r.player_id, name: r.name, country: null })}>
            <span>{r.name}</span>
            <span className="pos">{r.competition ?? ""}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Report({ player }: { player: Player | null }) {
  const [text, setText] = useState("Pick Player A, then generate a grounded scouting report.");
  const [busy, setBusy] = useState(false);

  async function generate() {
    if (!player) return;
    setBusy(true);
    setText("Contacting local LLM…");
    const { status, data } = await api.report(player.id);
    setBusy(false);
    if (status === 503) return setText("⚠️ Local LLM not reachable. Start LM Studio / Ollama.");
    if (!data) return setText("Request failed.");
    setText(data.report);
  }

  return (
    <div>
      <button onClick={generate} disabled={!player || busy}>
        {busy ? "Generating…" : `Generate report${player ? ` — ${player.name}` : ""}`}
      </button>
      <div className="prose">{text}</div>
    </div>
  );
}
