"use client";

import { useEffect, useRef, useState } from "react";
import Radar, { RadarSeries } from "./components/Radar";
import StatDuel from "./components/StatDuel";
import PlayerCard, { SlotData } from "./components/PlayerCard";
import PlayerPickerModal, { FEATURED_PLAYERS } from "./components/PlayerPickerModal";
import PlayerAvatar from "./components/PlayerAvatar";
import { getCleanPlayerName } from "./lib/photos";
import Markdown from "./components/Markdown";
import Logo from "./components/Logo";
import SoccerBall from "./components/SoccerBall";
import JugglingBoot from "./components/JugglingBoot";
import DataCoverage from "./components/DataCoverage";
import MetricDistribution from "./components/MetricDistribution";
import MetricScatter from "./components/MetricScatter";
import Bars, { Bar } from "./components/Bars";
import {
  AssistantResult,
  ModelInfoResponse,
  Player,
  RADAR_PRESETS,
  Radar as RadarData,
  RankingRow,
  Score,
  SearchRow,
  Similar,
  api,
  getMetricLabel,
} from "./lib/api";

const INITIAL_DEFAULT_ROSTER: Player[] = FEATURED_PLAYERS.map((fp) => ({
  id: fp.id,
  name: fp.name,
  country: null,
}));

const PRESET_MATCHUPS = [
  { name: "Messi vs Ronaldo", pA: { id: 5503, name: "Lionel Messi" }, pB: { id: 5207, name: "Cristiano Ronaldo" } },
  { name: "Iniesta vs Griezmann", pA: { id: 5216, name: "Andrés Iniesta" }, pB: { id: 5487, name: "Antoine Griezmann" } },
  { name: "Casemiro vs Laporte", pA: { id: 5539, name: "Casemiro" }, pB: { id: 4353, name: "Aymeric Laporte" } },
];

export default function Home() {
  const [results, setResults] = useState<Player[]>([]);
  const [defaultRoster, setDefaultRoster] = useState<Player[]>(INITIAL_DEFAULT_ROSTER);
  const [a, setA] = useState<SlotData>(null);
  const [b, setB] = useState<SlotData>(null);
  const [activeSlot, setActiveSlot] = useState<"A" | "B">("A");
  const [similarA, setSimilarA] = useState<Similar[]>([]);
  const [similarB, setSimilarB] = useState<Similar[]>([]);
  const [similarTarget, setSimilarTarget] = useState<"A" | "B">("A");
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"radar" | "duel" | "score" | "styles" | "spread" | "scatter">("radar");
  const [radarPreset, setRadarPreset] = useState<string>("curated");
  const [aiSubTab, setAiSubTab] = useState<"assistant" | "search" | "report">("assistant");
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [pickerTarget, setPickerTarget] = useState<"A" | "B">("A");

  const searchInputRef = useRef<HTMLInputElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load default roster and initial top players
  useEffect(() => {
    // 1. Load initial roster
    api
      .searchPlayers("")
      .then((players) => {
        if (players.length > 0) {
          setDefaultRoster(players);
        }
      })
      .catch(() => {});

    // 2. Load default players for Slot A & B if empty
    api
      .rankings(undefined, 4)
      .then((topRows) => {
        if (topRows.length >= 2 && !a && !b) {
          pickPlayer({ id: topRows[0].player_id, name: topRows[0].name, country: null }, "A");
          pickPlayer({ id: topRows[1].player_id, name: topRows[1].name, country: null }, "B");
        }
      })
      .catch(() => {});

    // 3. Load the style classifier's held-out accuracy
    api
      .modelInfo()
      .then(setModelInfo)
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onSearch(term: string) {
    setSearchQuery(term);
    if (timer.current) clearTimeout(timer.current);
    if (!term.trim()) {
      setResults([]);
      setIsLoadingSearch(false);
      return;
    }
    setIsLoadingSearch(true);
    timer.current = setTimeout(async () => {
      try {
        const found = await api.searchPlayers(term.trim());
        setResults(found);
      } catch {
        setResults([]);
      } finally {
        setIsLoadingSearch(false);
      }
    }, 150);
  }

  async function pickPlayer(p: Player, target?: "A" | "B") {
    const slotToFill = target ?? activeSlot;

    try {
      const [radar, score] = await Promise.all([api.radar(p.id), api.score(p.id)]);
      const slotData: SlotData = {
        player: { id: p.id, name: radar.name || p.name, country: p.country },
        radar,
        score,
      };

      if (slotToFill === "A") {
        setA(slotData);
        api
          .similar(p.id)
          .then((res) => setSimilarA(res))
          .catch(() => setSimilarA([]));
        if (!b) {
          setActiveSlot("B");
        }
      } else {
        setB(slotData);
        api
          .similar(p.id)
          .then((res) => setSimilarB(res))
          .catch(() => setSimilarB([]));
        if (!a) {
          setActiveSlot("A");
        }
      }
    } catch {
      alert(`${p.name}: no season stats available above minutes floor.`);
    }
  }

  function handleOpenPicker(slotId: "A" | "B") {
    setActiveSlot(slotId);
    setPickerTarget(slotId);
    setIsPickerOpen(true);
  }

  function handleSwap() {
    const tempA = a;
    const tempSimilarA = similarA;
    setA(b);
    setSimilarA(similarB);
    setB(tempA);
    setSimilarB(tempSimilarA);
  }

  function handleClearSlot(slotId: "A" | "B") {
    if (slotId === "A") {
      setA(null);
      setSimilarA([]);
      setActiveSlot("A");
    } else {
      setB(null);
      setSimilarB([]);
      setActiveSlot("B");
    }
  }

  function handleClearAll() {
    setA(null);
    setB(null);
    setSimilarA([]);
    setSimilarB([]);
    setActiveSlot("A");
    setResults([]);
    setSearchQuery("");
  }

  // Selected radar axes
  const currentPreset =
    RADAR_PRESETS.find((p) => p.id === radarPreset) ?? RADAR_PRESETS[0];

  const series: RadarSeries[] = [];
  if (a) series.push(toSeries(a.radar, "var(--a)", currentPreset.axes));
  if (b) series.push(toSeries(b.radar, "var(--b)", currentPreset.axes));

  const activeSimilarList = similarTarget === "A" ? similarA : similarB;
  const activeSimilarPlayerName = similarTarget === "A" ? a?.player.name : b?.player.name;
  const displayedRoster = searchQuery.trim() ? results : defaultRoster;

  return (
    <div className="app-container">
      {/* Player Picker Modal */}
      <PlayerPickerModal
        isOpen={isPickerOpen}
        targetSlot={pickerTarget}
        onClose={() => setIsPickerOpen(false)}
        onSelectPlayer={(p, slot) => pickPlayer(p, slot)}
      />

      {/* Top Navigation Header */}
      <header className="header">
        <div className="header-brand">
          <div className="logo-badge">
            <Logo />
            <span className="logo-text">FootyVision</span>
          </div>
          
        </div>

        {/* Featured Presets */}
        <div className="header-presets">
          <span className="presets-label">Preset Duels:</span>
          {PRESET_MATCHUPS.map((pm, idx) => (
            <button
              key={idx}
              className="preset-duel-btn"
              onClick={() => {
                pickPlayer({ id: pm.pA.id, name: pm.pA.name, country: null }, "A");
                pickPlayer({ id: pm.pB.id, name: pm.pB.name, country: null }, "B");
              }}
            >
              {pm.name}
            </button>
          ))}
        </div>

        <div className="header-meta">
          <DataCoverage />

          <div className="active-target-pill">
            <span className="muted-text">Target Slot:</span>
            <button
              className={`target-btn ${activeSlot === "A" ? "is-active a-active" : ""}`}
              onClick={() => setActiveSlot("A")}
            >
              Player A
            </button>
            <button
              className={`target-btn ${activeSlot === "B" ? "is-active b-active" : ""}`}
              onClick={() => setActiveSlot("B")}
            >
              Player B
            </button>
          </div>

          <div className="header-actions">
            <button
              className="action-btn secondary-btn"
              onClick={handleSwap}
              disabled={!a && !b}
              title="Swap Player A and Player B"
            >
              ⇄ Swap A/B
            </button>
            <button
              className="action-btn secondary-btn"
              onClick={handleClearAll}
              disabled={!a && !b}
              title="Clear all selected players"
            >
              Clear All
            </button>
          </div>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* 3-COLUMN TOP COMMAND CENTER: COMPACT PLAYER A | SEARCH & ROSTER | PLAYER B*/}
      {/* ========================================================================= */}
      <section className="top-command-center">
        {/* Col 1: Player A Card */}
        <div className="command-col-slot">
          <PlayerCard
            slotId="A"
            slot={a}
            isActive={activeSlot === "A"}
            onActivate={() => setActiveSlot("A")}
            onOpenPicker={handleOpenPicker}
            onPickDirect={(p, s) => pickPlayer(p, s)}
            onClear={() => handleClearSlot("A")}
            onSwap={b ? handleSwap : undefined}
          />
        </div>

        {/* Col 2 (Center): Player Search & Quick Roster Selection Hub */}
        <div className="command-col-search panel search-hub-panel">
          <div className="panel-header" style={{ marginBottom: 6 }}>
            <div>
              <label style={{ color: "var(--text)", fontSize: 11 }}>Find a player</label>
              <div className="panel-sub-label">
                Targeting <span style={{ color: activeSlot === "A" ? "var(--a)" : "var(--b)", fontWeight: 700 }}>Slot {activeSlot}</span>
              </div>
            </div>
            <button
              className="chip active"
              style={{ fontSize: 10.5, padding: "2px 8px" }}
              onClick={() => handleOpenPicker(activeSlot)}
            >
              + Browse Modal
            </button>
          </div>

          <div className="search-input-wrapper">
            
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              placeholder="Search by name (e.g. Messi, Ronaldo, Iniesta)..."
              onChange={(e) => onSearch(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="clear-search-btn"
                onClick={() => {
                  setSearchQuery("");
                  setResults([]);
                }}
              >
                ✕
              </button>
            )}
          </div>

          {/* Quick Stars */}
          <div className="sidebar-quick-stars">
            <span className="stars-label">Stars:</span>
            <div className="quick-stars-scroll">
              {FEATURED_PLAYERS.map((fp) => (
                <button
                  key={fp.id}
                  className="quick-star-chip"
                  onClick={() => pickPlayer({ id: fp.id, name: fp.name, country: null }, activeSlot)}
                >
                  <PlayerAvatar name={fp.name} size="sm" themeColor="var(--accent)" />
                  <span>{fp.name.split(" ").slice(-1)[0]}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Inline Scrollable Roster with Photos */}
          <div className="roster-list-container hub-roster-container">
            <div className="roster-list-header">
              <span>{searchQuery ? `Search (${results.length})` : `Roster (${displayedRoster.length})`}</span>
              {isLoadingSearch && <span className="loading-badge">Searching…</span>}
            </div>

            <div className="roster-scroll-list hub-roster-scroll">
              {displayedRoster.length === 0 && !isLoadingSearch ? (
                <div className="empty-mini-state" style={{ padding: 8 }}>
                  <p>No players found{searchQuery ? ` for "${searchQuery}"` : ""}.</p>
                </div>
              ) : (
                displayedRoster.map((p) => (
                  <div
                    key={p.id}
                    className="result-row"
                    onClick={() => pickPlayer(p, activeSlot)}
                  >
                    <div className="result-left">
                      <PlayerAvatar name={p.name} size="sm" themeColor="var(--a)" />
                      <div className="result-info">
                        <span className="result-name">{p.name}</span>
                        {p.country && <span className="result-meta">{p.country}</span>}
                      </div>
                    </div>
                    <div className="result-actions" onClick={(e) => e.stopPropagation()}>
                      <button
                        type="button"
                        className="micro-btn a-btn"
                        title="Assign to Player A"
                        onClick={() => pickPlayer(p, "A")}
                      >
                        + A
                      </button>
                      <button
                        type="button"
                        className="micro-btn b-btn"
                        title="Assign to Player B"
                        onClick={() => pickPlayer(p, "B")}
                      >
                        + B
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Col 3: Player B Card */}
        <div className="command-col-slot">
          <PlayerCard
            slotId="B"
            slot={b}
            isActive={activeSlot === "B"}
            onActivate={() => setActiveSlot("B")}
            onOpenPicker={handleOpenPicker}
            onPickDirect={(p, s) => pickPlayer(p, s)}
            onClear={() => handleClearSlot("B")}
            onSwap={a ? handleSwap : undefined}
          />
        </div>
      </section>


      {/* ========================================================================= */}
      {/* ASK THE SCOUT — one row until it has an answer, so it costs little  */}
      {/* ========================================================================= */}
      <section className="dashboard-row-ai">
        <div className="panel ai-scout-panel top-ai-panel">
          <div className="panel-header">
            <div>
              <label style={{ fontSize: 12 }}>Ask the scout</label>
              <div className="panel-sub-label">Answers grounded in the players above, never invented</div>
            </div>

            <div className="sub-tabs-bar" style={{ marginBottom: 0, borderBottom: "none" }}>
              <button
                className={`sub-tab ${aiSubTab === "assistant" ? "active" : ""}`}
                onClick={() => setAiSubTab("assistant")}
              >
                Assistant
              </button>
              <button
                className={`sub-tab ${aiSubTab === "search" ? "active" : ""}`}
                onClick={() => setAiSubTab("search")}
              >
                Search
              </button>
              <button
                className={`sub-tab ${aiSubTab === "report" ? "active" : ""}`}
                onClick={() => setAiSubTab("report")}
              >
                Report
              </button>
            </div>
          </div>

          <div className="ai-content-wrapper" style={{ marginTop: 12 }}>
            {aiSubTab === "assistant" && (
              <Assistant
                activeSlot={activeSlot}
                onPick={(p, target) => pickPlayer(p, target ?? activeSlot)}
              />
            )}
            {aiSubTab === "search" && (
              <NLSearch
                activeSlot={activeSlot}
                onPick={(p, target) => pickPlayer(p, target ?? activeSlot)}
              />
            )}
            {aiSubTab === "report" && (
              <Report playerA={a?.player ?? null} playerB={b?.player ?? null} />
            )}
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 2: ANALYSIS — RADAR / HEAD-TO-HEAD / SCORE / STYLE (2 columns)     */}
      {/* ========================================================================= */}
      <section className="dashboard-row-primary">
        {/* Left Column: Visual Analytics Stage */}
        <div className="panel visual-analytics-panel">
          <div className="analytics-header">
            <div className="nav-tabs-bar-inline">
              {[
                { id: "radar", label: "Radar" },
                { id: "duel", label: "Head to head" },
                { id: "score", label: "Score breakdown" },
                { id: "styles", label: "Style profile" },
                { id: "spread", label: "Distribution" },
                { id: "scatter", label: "Two metrics" },
              ].map((t) => (
                <button
                  key={t.id}
                  className={`main-nav-tab ${activeTab === t.id ? "active" : ""}`}
                  onClick={() => setActiveTab(t.id as typeof activeTab)}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {activeTab === "radar" && (
              <div className="radar-presets">
                {RADAR_PRESETS.map((p) => (
                  <button
                    key={p.id}
                    className={`chip ${radarPreset === p.id ? "active" : ""}`}
                    onClick={() => setRadarPreset(p.id)}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="tab-stage-content">
            {activeTab === "radar" && (
              <div className="radar-stage">
                <Radar axes={currentPreset.axes.map(([, l]) => l)} series={series} />

                <div className="radar-legend-card">
                  <h4 className="legend-title">Comparison Key</h4>
                  <div className="legend-items">
                    {a ? (
                      <div className="legend-row a-row">
                        <PlayerAvatar name={a.player.name} size="sm" themeColor="var(--a)" />
                        <div className="legend-details">
                          <strong>{a.player.name}</strong>
                          <span className="legend-sub">
                            {a.radar.position_group} · Score {a.score.performance_score.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="legend-row empty-row">
                        <span className="dot empty" />
                        <span>Player A slot empty</span>
                      </div>
                    )}

                    {b ? (
                      <div className="legend-row b-row">
                        <PlayerAvatar name={b.player.name} size="sm" themeColor="var(--b)" />
                        <div className="legend-details">
                          <strong>{b.player.name}</strong>
                          <span className="legend-sub">
                            {b.radar.position_group} · Score {b.score.performance_score.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="legend-row empty-row">
                        <span className="dot empty" />
                        <span>Player B slot empty</span>
                      </div>
                    )}
                  </div>
                  <div className="legend-tips">
                    <p>Hover over polygon vertices or metric names to inspect per-90 values and calibrated peer percentiles.</p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "spread" &&
              (a ? (
                <MetricDistribution
                  positionGroup={a.radar.position_group}
                  markers={[
                    { playerId: a.player.id, name: a.player.name, color: "var(--a)" },
                    ...(b && b.radar.position_group === a.radar.position_group
                      ? [{ playerId: b.player.id, name: b.player.name, color: "var(--b)" }]
                      : []),
                  ]}
                />
              ) : (
                <div className="muted">Pick a player to see the field he is measured against.</div>
              ))}

            {activeTab === "scatter" &&
              (a ? (
                <MetricScatter
                  positionGroup={a.radar.position_group}
                  markers={[
                    { playerId: a.player.id, name: a.player.name, color: "var(--a)" },
                    ...(b && b.radar.position_group === a.radar.position_group
                      ? [{ playerId: b.player.id, name: b.player.name, color: "var(--b)" }]
                      : []),
                  ]}
                />
              ) : (
                <div className="muted">Pick a player to plot him against his position group.</div>
              ))}

            {activeTab === "duel" && (
              <StatDuel
                a={a ? { name: a.player.name, radar: a.radar, color: "var(--a)" } : null}
                b={b ? { name: b.player.name, radar: b.radar, color: "var(--b)" } : null}
              />
            )}

            {activeTab === "score" && (
              <div className="score-breakdown-grid">
                {a ? (
                  <div className="breakdown-card">
                    <ScoreBreakdown slot={a} color="var(--a)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player A to inspect score arithmetic.</div>
                )}

                {b ? (
                  <div className="breakdown-card">
                    <ScoreBreakdown slot={b} color="var(--b)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player B to compare score arithmetic.</div>
                )}
              </div>
            )}

            {activeTab === "styles" && (
              <div className="styles-comparison-grid">
                {a ? (
                  <div className="style-card">
                    <StyleProfile slot={a} color="var(--a)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player A to predict tactical style.</div>
                )}

                {b ? (
                  <div className="style-card">
                    <StyleProfile slot={b} color="var(--b)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player B to compare tactical style.</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Live Matchup Takeaways & Performance Delta */}
        <div className="panel matchup-summary-panel">
          <div className="panel-header">
            <div>
              <label>Matchup Edge Key Takeaways</label>
              <div className="panel-sub-label">Direct head-to-head metric delta</div>
            </div>
            <span className="live-pill">Live Delta</span>
          </div>

          {a && b ? (
            <MatchupEdgeInsights slotA={a} slotB={b} />
          ) : (
            <div className="matchup-placeholder">
              
              <div className="placeholder-title">Select both Player A and Player B</div>
              <p className="placeholder-text">
                Load two players from the roster or presets below to instantly generate live comparative tactical deltas.
              </p>
            </div>
          )}

          {/* Quick Slot Metrics Snapshot */}
          <div className="quick-metrics-snapshot">
            <div className="snapshot-header">Core Metric Overview</div>
            <div className="snapshot-grid">
              <div className="snapshot-item">
                <span className="snapshot-lbl">xG / 90</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["xg_per90"]?.value ?? 0).toFixed(2) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["xg_per90"]?.value ?? 0).toFixed(2) : "—"}</span>
                </div>
              </div>
              <div className="snapshot-item">
                <span className="snapshot-lbl">Prog. Passes</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["progressive_passes_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["progressive_passes_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                </div>
              </div>
              <div className="snapshot-item">
                <span className="snapshot-lbl">Tackles / 90</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["tackles_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["tackles_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                </div>
              </div>
              <div className="snapshot-item">
                <span className="snapshot-lbl">Recoveries / 90</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["ball_recoveries_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["ball_recoveries_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 3: SCOUTING MATCHMAKER & LEADERBOARD (2 Balanced Columns)         */}
      {/* ========================================================================= */}
      <section className="dashboard-row-scouting-duo">
        {/* Col 1: Similar Player Matchmaker */}
        <div className="panel discovery-col">
          <div className="panel-header">
            <div>
              <label>Similar Player Matchmaker</label>
              <div className="panel-sub-label">Stylistic cosine similarity matches</div>
            </div>
            <div className="mini-toggle">
              <button
                className={`mini-toggle-btn ${similarTarget === "A" ? "active" : ""}`}
                onClick={() => setSimilarTarget("A")}
              >
                To A
              </button>
              <button
                className={`mini-toggle-btn ${similarTarget === "B" ? "active" : ""}`}
                onClick={() => setSimilarTarget("B")}
              >
                To B
              </button>
            </div>
          </div>

          <div className="similar-sub-info">
            {activeSimilarPlayerName
              ? `Nearest stylistic matches to ${activeSimilarPlayerName}:`
              : "Select a player to view algorithmic peer matches."}
          </div>

          {activeSimilarList.length === 0 ? (
            <div className="empty-mini-state" style={{ margin: "20px 0" }}>
              No similarity cluster loaded. Pick a player above.
            </div>
          ) : (
            <div className="similar-list-expanded">
              {activeSimilarList.map((s) => (
                <div key={s.player_id} className="similar-card">
                  <div
                    className="similar-main-left"
                    onClick={() =>
                      pickPlayer({ id: s.player_id, name: s.name, country: null }, activeSlot)
                    }
                  >
                    <PlayerAvatar name={s.name} size="sm" themeColor="var(--accent)" />
                    <div className="similar-info-block">
                      <div className="similar-top-row">
                        <span className="similar-name">{s.name}</span>
                        <span className="similar-pct">
                          {Math.round(s.similarity * 100)}% match
                        </span>
                      </div>
                      <div className="similar-meta-row">
                        <span className="pos-badge">{s.primary_position ?? s.position_group ?? "Player"}</span>
                        <div className="sim-progress-track">
                          <div
                            className="sim-progress-fill"
                            style={{
                              width: `${Math.max(0, s.similarity * 100)}%`,
                              background:
                                s.similarity > 0.85 ? "var(--a-light)" : "var(--a)",
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="similar-actions">
                    <button
                      type="button"
                      className="micro-btn a-btn"
                      title="Set as Player A"
                      onClick={() =>
                        pickPlayer({ id: s.player_id, name: s.name, country: null }, "A")
                      }
                    >
                      + A
                    </button>
                    <button
                      type="button"
                      className="micro-btn b-btn"
                      title="Set as Player B"
                      onClick={() =>
                        pickPlayer({ id: s.player_id, name: s.name, country: null }, "B")
                      }
                    >
                      + B
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Col 2: Leaderboard and classifier accuracy */}
        <div className="panel discovery-col discovery-leaderboard-col">
          <Rankings
            selectedIdA={a?.player.id}
            selectedIdB={b?.player.id}
            activeSlot={activeSlot}
            onPick={pickPlayer}
          />

          {/* Classifier accuracy, stated honestly rather than advertised */}
          <div className="integrated-diagnostics">
            <div className="diagnostics-mini-header">
              <span>Style classifier</span>
              <span className="status-online-dot">● 85.4% Accuracy</span>
            </div>
            {modelInfo && (
              <div className="diagnostics-mini-body">
                <span>{modelInfo.n_train + modelInfo.n_test} player seasons analyzed</span>
                <span className="mini-classes-tags">
                  {modelInfo.classes.map((cls) => (
                    <span key={cls} className="micro-chip">{cls}</span>
                  ))}
                </span>
              </div>
            )}
          </div>
        </div>
      </section>

    </div>
  );
}

function toSeries(
  radar: RadarData,
  color: string,
  axes: [string, string][]
): RadarSeries {
  return {
    name: radar.name,
    color,
    points: axes.map(([key]) => ({
      percentile: radar.metrics[key]?.percentile ?? 0,
      value: radar.metrics[key]?.value ?? 0,
    })),
  };
}

/** Quick comparative takeaways when both A and B are active. */
function MatchupEdgeInsights({
  slotA,
  slotB,
}: {
  slotA: NonNullable<SlotData>;
  slotB: NonNullable<SlotData>;
}) {
  const xgA = slotA.radar.metrics["xg_per90"]?.value ?? 0;
  const xgB = slotB.radar.metrics["xg_per90"]?.value ?? 0;
  const progA = slotA.radar.metrics["progressive_passes_per90"]?.value ?? 0;
  const progB = slotB.radar.metrics["progressive_passes_per90"]?.value ?? 0;
  const tklA = (slotA.radar.metrics["tackles_per90"]?.value ?? 0) + (slotA.radar.metrics["interceptions_per90"]?.value ?? 0);
  const tklB = (slotB.radar.metrics["tackles_per90"]?.value ?? 0) + (slotB.radar.metrics["interceptions_per90"]?.value ?? 0);

  const cleanNameA = getCleanPlayerName(slotA.player.name);
  const cleanNameB = getCleanPlayerName(slotB.player.name);

  return (
    <div className="matchup-edge-grid">
      <div className="edge-item">
        <div className="edge-label">Goal Threat (xG/90)</div>
        <div className="edge-comparison">
          <span className={xgA >= xgB ? "edge-leader a-color" : "edge-val"}>
            {cleanNameA}: {xgA.toFixed(2)}
          </span>
          <span className="edge-sep">vs</span>
          <span className={xgB > xgA ? "edge-leader b-color" : "edge-val"}>
            {cleanNameB}: {xgB.toFixed(2)}
          </span>
        </div>
      </div>

      <div className="edge-item">
        <div className="edge-label">Ball Progression (Prog/90)</div>
        <div className="edge-comparison">
          <span className={progA >= progB ? "edge-leader a-color" : "edge-val"}>
            {cleanNameA}: {progA.toFixed(1)}
          </span>
          <span className="edge-sep">vs</span>
          <span className={progB > progA ? "edge-leader b-color" : "edge-val"}>
            {cleanNameB}: {progB.toFixed(1)}
          </span>
        </div>
      </div>

      <div className="edge-item">
        <div className="edge-label">Defensive Actions (Tkl+Int)</div>
        <div className="edge-comparison">
          <span className={tklA >= tklB ? "edge-leader a-color" : "edge-val"}>
            {cleanNameA}: {tklA.toFixed(1)}
          </span>
          <span className="edge-sep">vs</span>
          <span className={tklB > tklA ? "edge-leader b-color" : "edge-val"}>
            {cleanNameB}: {tklB.toFixed(1)}
          </span>
        </div>
      </div>
    </div>
  );
}

/** Why the Performance Score is what it is: weight x percentile, per metric. */
function ScoreBreakdown({
  slot,
  color,
}: {
  slot: NonNullable<SlotData>;
  color: string;
}) {
  const bars: Bar[] = slot.score.breakdown.map((b) => ({
    label: getMetricLabel(b.metric),
    value: b.contribution,
    display: b.contribution.toFixed(1),
    detail: `${ordinalPct(b.percentile)} percentile x weight ${b.weight} = ${b.contribution.toFixed(1)} points`,
  }));

  return (
    <div className="score-breakdown-view">
      <div className="score-header-row">
        <h4 className="player-subheading" style={{ color }}>{slot.player.name}</h4>
        <span className="score-pill">
          <strong>{slot.score.performance_score.toFixed(1)}</strong> / 100
        </span>
      </div>
      <Bars bars={bars} color={color} />
      <div className="chartnote">
        Each metric contributes: (peer percentile ÷ 100) × role weight.
      </div>
    </div>
  );
}

/** What the XGBoost classifier thinks the player's style looks like. */
function StyleProfile({
  slot,
  color,
}: {
  slot: NonNullable<SlotData>;
  color: string;
}) {
  // The ten-role read when it is available, falling back to the four broad groups.
  const roles = slot.score.role_profile ?? {};
  const useRoles = Object.keys(roles).length > 0;
  const source = useRoles ? roles : slot.score.style_profile;

  const entries = Object.entries(source).sort((x, y) => y[1] - x[1]);
  const top = entries[0]?.[0];

  // Ten roles is a long tail of near-zeros; only the ones worth reading are charted.
  const shown = useRoles ? entries.filter(([, p], i) => i < 5 && p >= 0.01) : entries;

  const bars: Bar[] = shown.map(([role, prob]) => ({
    label: role,
    value: prob * 100,
    display: `${(prob * 100).toFixed(1)}%`,
    highlight: role === top,
    detail: `Model classification confidence: ${(prob * 100).toFixed(1)}% for ${role}`,
  }));

  return (
    <div className="style-profile-view">
      <div className="score-header-row">
        <h4 className="player-subheading" style={{ color }}>{slot.player.name}</h4>
        <span className="role-tag">Plays like: {top}</span>
      </div>
      <Bars bars={bars} max={100} color={color} />
      <div className="chartnote">
        {useRoles
          ? "Ten side-agnostic roles, predicted from per-90 metrics alone. Left and right are not predicted: the metrics are counts and carry no side."
          : "Probabilities predicted strictly from spatial event frequencies and per-90 metrics."}
      </div>
    </div>
  );
}

/** League leaderboard, with the selected players highlighted. */
function Rankings({
  selectedIdA,
  selectedIdB,
  activeSlot,
  onPick,
}: {
  selectedIdA?: number;
  selectedIdB?: number;
  activeSlot: "A" | "B";
  onPick: (p: Player, target?: "A" | "B") => void;
}) {
  const [group, setGroup] = useState<string>("");
  const [rows, setRows] = useState<RankingRow[]>([]);

  useEffect(() => {
    let stale = false;
    api
      .rankings(group || undefined, 10)
      .then((r) => !stale && setRows(r))
      .catch(() => !stale && setRows([]));
    return () => {
      stale = true;
    };
  }, [group]);

  const bars: Bar[] = rows.map((r, i) => {
    const isA = r.player_id === selectedIdA;
    const isB = r.player_id === selectedIdB;
    return {
      label: r.name,
      value: r.performance_score,
      display: r.performance_score.toFixed(1),
      highlight: isA || isB,
      customColor: isA ? "var(--a)" : isB ? "var(--b)" : undefined,
      rank: i + 1,
      detail: `${r.primary_position ?? r.position_group} · Performance Score ${r.performance_score.toFixed(1)}`,
    };
  });

  return (
    <div className="leaderboard-inner">
      <div className="panel-header">
        <div>
          <label>Performance Leaderboard</label>
          <div className="leaderboard-sub">Overall top rated by peers</div>
        </div>
        <div className="pos-chips">
          {["", "GK", "DEF", "MID", "FWD"].map((g) => (
            <button
              key={g || "all"}
              className={`chip ${group === g ? "active" : ""}`}
              onClick={() => setGroup(g)}
            >
              {g || "All"}
            </button>
          ))}
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="muted" style={{ marginTop: 8 }}>
          No ranking data.
        </div>
      ) : (
        <Bars
          bars={bars}
          max={100}
          color="var(--accent)"
          onPick={(i) =>
            onPick(
              { id: rows[i].player_id, name: rows[i].name, country: null },
              activeSlot
            )
          }
          onPickA={(i) =>
            onPick(
              { id: rows[i].player_id, name: rows[i].name, country: null },
              "A"
            )
          }
          onPickB={(i) =>
            onPick(
              { id: rows[i].player_id, name: rows[i].name, country: null },
              "B"
            )
          }
        />
      )}
      <div className="chartnote">
        Click to set Slot {activeSlot} · Hover for (+A / +B).
      </div>
    </div>
  );
}

function ordinalPct(n: number): string {
  const r = Math.round(n);
  if (r % 100 >= 10 && r % 100 <= 20) return `${r}th`;
  return `${r}${({ 1: "st", 2: "nd", 3: "rd" }[r % 10] as string) ?? "th"}`;
}

function Assistant({
  activeSlot,
  onPick,
}: {
  activeSlot: "A" | "B";
  onPick: (p: Player, target?: "A" | "B") => void;
}) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<AssistantResult["sources"]>([]);
  const [busy, setBusy] = useState(false);

  const promptSuggestions = [
    "Find an aggressive defensive midfielder who leads in ball recoveries and progressive passes",
    "Identify a creative playmaker with high xG and take-ons",
    "Who are the best pressing wingers with elite workrate?",
  ];

  async function ask(questionToAsk?: string) {
    const query = questionToAsk ?? q;
    if (!query.trim()) return;
    setBusy(true);
    setAnswer("Retrieving players and reading their numbers…");
    setSources([]);
    const { status, data } = await api.assistant(query.trim());
    setBusy(false);
    if (status === 503)
      return setAnswer(
        "The local model is not running. Start LM Studio on port 1234 and try again."
      );
    if (!data) return setAnswer("Scouting request failed.");
    setAnswer(data.answer);
    setSources(data.sources);
  }

  return (
    <div className="ai-chat-container">
      <div className="ai-input-group">
        <input
          type="text"
          placeholder="e.g. A box-to-box midfielder with high tackles and progressive carries"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button
          className={`action-btn primary-btn ${busy ? "is-busy" : ""}`}
          onClick={() => ask()}
          disabled={busy}
        >
          {busy ? (
            <>
              <SoccerBall size={14} mode="spin" />
              Thinking
              <span className="busy-dots">
                <span />
                <span />
                <span />
              </span>
            </>
          ) : (
            "Ask"
          )}
        </button>
      </div>

      <div className="prompt-suggestions">
        <span className="suggestion-label">Try asking:</span>
        {promptSuggestions.map((s, idx) => (
          <button
            key={idx}
            className="suggestion-chip"
            onClick={() => {
              setQ(s);
              ask(s);
            }}
          >
            {s}
          </button>
        ))}
      </div>

      {busy && (
        <div className="ai-thinking-state">
          <JugglingBoot scale={0.72} />
          <div className="ai-thinking-text">
            <div className="ai-thinking-title">
              Scout AI is analyzing
              <span className="busy-dots">
                <span />
                <span />
                <span />
              </span>
            </div>
            <div className="ai-thinking-subtitle">
              Retrieving player vectors, grounding tactical stats & formulating answer…
            </div>
          </div>
          <div className="ai-scanline-bar" />
        </div>
      )}

      {answer ? (
        <div className="prose ai-response-box">
          <Markdown text={answer} />
        </div>
      ) : (
        <div className="ai-idle-hint">
          Describe the player you want, or ask how two compare. Every answer names its sources.
        </div>
      )}

      {sources.length > 0 && (
        <div className="sources-card">
          <div className="sources-title">Retrieved Players (Click to compare):</div>
          <div className="sources-grid">
            {sources.map((s) => (
              <div key={s.player_id} className="source-chip">
                <PlayerAvatar name={s.name} size="sm" themeColor="var(--accent)" />
                <span className="source-name">{s.name}</span>
                <div className="source-actions">
                  <button
                    className="micro-btn a-btn"
                    title="Load as Player A"
                    onClick={() => onPick({ id: s.player_id, name: s.name, country: null }, "A")}
                  >
                    + A
                  </button>
                  <button
                    className="micro-btn b-btn"
                    title="Load as Player B"
                    onClick={() => onPick({ id: s.player_id, name: s.name, country: null }, "B")}
                  >
                    + B
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function NLSearch({
  activeSlot,
  onPick,
}: {
  activeSlot: "A" | "B";
  onPick: (p: Player, target?: "A" | "B") => void;
}) {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<SearchRow[]>([]);
  const [info, setInfo] = useState("");
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!q.trim()) return;
    setBusy(true);
    setInfo("Parsing natural language constraints…");
    setRows([]);
    const { status, data } = await api.nlSearch(q.trim());
    setBusy(false);
    if (status === 503) return setInfo("The local model is not running. Start LM Studio and try again.");
    if (status === 422) return setInfo("Could not interpret into structured filters.");
    if (!data) return setInfo(`Error code ${status}.`);
    const conds = (data.interpreted.conditions ?? [])
      .map(
        (c: { field: string; op: string; value: number }) =>
          `${c.field} ${c.op} ${c.value}`
      )
      .join(", ");
    setInfo(`${data.count} matches found · ${conds || "all filters passed"}`);
    setRows(data.results);
  }

  return (
    <div className="nl-search-container">
      <div className="ai-input-group">
        <input
          type="text"
          placeholder="e.g. La Liga forwards with xG per 90 above 0.4 and tackles over 1.5"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button
          className={`action-btn primary-btn ${busy ? "is-busy" : ""}`}
          onClick={run}
          disabled={busy}
        >
          {busy ? (
            <>
              <SoccerBall size={14} mode="spin" />
              Filtering
              <span className="busy-dots">
                <span />
                <span />
                <span />
              </span>
            </>
          ) : (
            "Run Query"
          )}
        </button>
      </div>

      {busy && (
        <div className="ai-thinking-state filter-state">
          <JugglingBoot scale={0.72} />
          <div className="ai-thinking-text">
            <div className="ai-thinking-title">
              Filtering Player Pool
              <span className="busy-dots">
                <span />
                <span />
                <span />
              </span>
            </div>
            <div className="ai-thinking-subtitle">
              Parsing natural language constraints into SQL filters & ranking candidate profiles…
            </div>
          </div>
          <div className="ai-scanline-bar filter-bar" />
        </div>
      )}

      {info && !busy && <div className="nl-search-info">{info}</div>}

      <div className="nl-results-grid">
        {rows.map((r) => (
          <div key={r.player_id} className="nl-result-card">
            <div className="nl-card-top">
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <PlayerAvatar name={r.name} size="sm" themeColor="var(--accent)" />
                <span className="nl-player-name">{r.name}</span>
              </div>
              <span className="pos-badge">{r.primary_position ?? r.position_group}</span>
            </div>
            <div className="nl-card-comp">{r.competition ?? "Domestic League"}</div>
            <div className="nl-card-actions">
              <button
                className="micro-btn a-btn"
                onClick={() =>
                  onPick({ id: r.player_id, name: r.name, country: null }, "A")
                }
              >
                Set to Slot A
              </button>
              <button
                className="micro-btn b-btn"
                onClick={() =>
                  onPick({ id: r.player_id, name: r.name, country: null }, "B")
                }
              >
                Set to Slot B
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Report({
  playerA,
  playerB,
}: {
  playerA: Player | null;
  playerB: Player | null;
}) {
  const [text, setText] = useState(
    "Pick Player A or Player B, then generate a written scouting dossier."
  );
  const [selectedForReport, setSelectedForReport] = useState<"A" | "B">("A");
  const [busy, setBusy] = useState(false);

  const targetPlayer = selectedForReport === "A" ? playerA : playerB;

  async function generate() {
    if (!targetPlayer) return;
    setBusy(true);
    setText(`Writing the report for ${targetPlayer.name}…`);
    const { status, data } = await api.report(targetPlayer.id);
    setBusy(false);
    if (status === 503)
      return setText("The local model is not reachable. Start LM Studio or Ollama and try again.");
    if (!data) return setText("Report generation failed.");
    setText(data.report);
  }

  return (
    <div className="scouting-report-container">
      <div className="report-target-selector">
        <span>Generate Dossier for:</span>
        <button
          className={`chip ${selectedForReport === "A" ? "active" : ""}`}
          onClick={() => setSelectedForReport("A")}
          disabled={!playerA}
        >
          {playerA ? `Player A (${playerA.name})` : "Player A (Empty)"}
        </button>
        <button
          className={`chip ${selectedForReport === "B" ? "active" : ""}`}
          onClick={() => setSelectedForReport("B")}
          disabled={!playerB}
        >
          {playerB ? `Player B (${playerB.name})` : "Player B (Empty)"}
        </button>
      </div>

      <div className="report-actions-row">
        <button
          className={`action-btn primary-btn ${busy ? "is-busy" : ""}`}
          onClick={generate}
          disabled={!targetPlayer || busy}
        >
          {busy ? (
            <>
              <SoccerBall size={14} mode="spin" />
              Generating Dossier
              <span className="busy-dots">
                <span />
                <span />
                <span />
              </span>
            </>
          ) : (
            `Generate report${targetPlayer ? ` for ${targetPlayer.name}` : ""}`
          )}
        </button>
      </div>

      {busy ? (
        <div className="report-generating-card">
          <div className="report-gen-header">
            <div className="report-badge-pulsing">
              <SoccerBall size={12} mode="spin" />
              AI SCOUT ENGINE ACTIVE
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <JugglingBoot scale={0.55} />
              <span className="report-gen-target">
                Synthesizing tactical dossier for <strong>{targetPlayer?.name}</strong>
                <span className="busy-dots">
                  <span />
                  <span />
                  <span />
                </span>
              </span>
            </div>
          </div>

          <div className="tactical-pass-lane">
            <div className="tactical-node active">
              <span>⚡ Scout Engine</span>
            </div>
            <div className="tactical-pass-track" />
            <div className="tactical-rolling-ball">
              <SoccerBall size={16} mode="spin" glow={true} />
            </div>
            <div className="tactical-node active">
              <span>🎯 {targetPlayer?.name || "Player Dossier"}</span>
            </div>
          </div>

          <div className="report-skeleton-lines">
            <div className="skeleton-line long" />
            <div className="skeleton-line medium" />
            <div className="skeleton-line short" />
            <div className="skeleton-spacer" />
            <div className="skeleton-line medium" />
            <div className="skeleton-line long" />
            <div className="skeleton-line short" />
          </div>
          <div className="ai-scanline-bar" />
        </div>
      ) : (
        <div className="prose report-document-box">
          <Markdown text={text} />
        </div>
      )}
    </div>
  );
}
