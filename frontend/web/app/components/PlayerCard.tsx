"use client";

import { Player, Radar as RadarData, Score } from "../lib/api";
import { FEATURED_PLAYERS } from "./PlayerPickerModal";
import PlayerAvatar from "./PlayerAvatar";

export type SlotData = {
  player: Player;
  radar: RadarData;
  score: Score;
} | null;

type PlayerCardProps = {
  slotId: "A" | "B";
  slot: SlotData;
  isActive: boolean;
  onActivate: () => void;
  onOpenPicker: (slotId: "A" | "B") => void;
  onPickDirect: (p: Player, slotId: "A" | "B") => void;
  onClear: () => void;
  onSwap?: () => void;
};

export default function PlayerCard({
  slotId,
  slot,
  isActive,
  onActivate,
  onOpenPicker,
  onPickDirect,
  onClear,
  onSwap,
}: PlayerCardProps) {
  const isA = slotId === "A";
  const themeColor = isA ? "var(--a)" : "var(--b)";
  // Channels of --a / --b, for the rgba() tints CSS variables cannot be composed into.
  const themeRgb = isA ? "184, 142, 45" : "224, 96, 126";

  if (!slot) {
    const quickPicks = isA
      ? FEATURED_PLAYERS.slice(0, 3)
      : FEATURED_PLAYERS.slice(1, 4);

    return (
      <div
        className={`player-slot-card hero-card empty ${isActive ? "active-target" : ""}`}
        onClick={onActivate}
        style={{
          borderColor: isActive ? themeColor : undefined,
          boxShadow: isActive ? `0 0 20px rgba(${themeRgb}, 0.25)` : undefined,
        }}
      >
        <div className="slot-header-bar">
          <div className="slot-pill-group">
            <span
              className="slot-pill"
              style={{ backgroundColor: `rgba(${themeRgb}, 0.15)`, color: themeColor }}
            >
              Player {slotId}
            </span>
            <span className="muted-badge">Empty Slot</span>
          </div>

          <button
            type="button"
            className="action-btn primary-btn mini-btn"
            style={{ background: themeColor, color: isA ? "#000" : "#fff" }}
            onClick={(e) => {
              e.stopPropagation();
              onActivate();
              onOpenPicker(slotId);
            }}
          >
            Search
          </button>
        </div>

        <div className="empty-slot-hero">
          <div
            className="empty-hero-circle"
            style={{ borderColor: `rgba(${themeRgb}, 0.4)`, color: themeColor }}
          >
            +
          </div>
          <div className="empty-hero-title">Select Player {slotId}</div>
          <span className="compact-prompt" style={{ marginBottom: 6 }}>Quick Add Featured:</span>

          <div className="card-quick-picks">
            {quickPicks.map((qp) => (
              <button
                key={qp.id}
                type="button"
                className="quick-pick-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onPickDirect({ id: qp.id, name: qp.name, country: null }, slotId);
                }}
              >
                + {qp.name.split(" ").slice(-1)[0]}
              </button>
            ))}
          </div>
        </div>

        <div className="hero-target-bar">
          <span className={isActive ? "target-active-text" : "target-muted-text"}>
            {isActive ? "● Active Target for Assignment" : "Click card to target"}
          </span>
        </div>
      </div>
    );
  }

  const { player, radar, score } = slot;
  const topRole = Object.entries(score.style_profile).sort((x, y) => y[1] - x[1])[0];
  const styleText = topRole ? `${topRole[0]} (${Math.round(topRole[1] * 100)}%)` : "—";

  return (
    <div
      className={`player-slot-card hero-card filled ${isActive ? "active-target" : ""}`}
      onClick={onActivate}
      style={{
        borderColor: isActive ? themeColor : `rgba(${themeRgb}, 0.35)`,
        boxShadow: isActive ? `0 0 24px rgba(${themeRgb}, 0.3)` : undefined,
      }}
    >
      {/* Card Header */}
      <div className="slot-header-bar">
        <div className="slot-pill-group">
          <span
            className="slot-pill"
            style={{ backgroundColor: `rgba(${themeRgb}, 0.2)`, color: themeColor, fontWeight: 700 }}
          >
            Player {slotId}
          </span>
          <span className="position-tag">{radar.position_group}</span>
          {player.country && <span className="country-tag">{player.country}</span>}
        </div>

        <div className="slot-actions">
          {onSwap && (
            <button
              type="button"
              className="slot-action"
              title="Swap Player A and Player B"
              aria-label="Swap Player A and Player B"
              onClick={(e) => {
                e.stopPropagation();
                onSwap();
              }}
            >
              ⇄
            </button>
          )}
          <button
            type="button"
            className="slot-action"
            title={`Replace Player ${slotId}`}
            aria-label={`Replace Player ${slotId}`}
            onClick={(e) => {
              e.stopPropagation();
              onActivate();
              onOpenPicker(slotId);
            }}
          >
            ↻
          </button>
          <button
            type="button"
            className="slot-action slot-action-clear"
            title={`Remove Player ${slotId}`}
            aria-label={`Remove Player ${slotId}`}
            onClick={(e) => {
              e.stopPropagation();
              onClear();
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Identity reads across, like a team-sheet line: who, then where he played. */}
      <div className="slot-identity">
        <PlayerAvatar name={player.name} size="hero" themeColor={themeColor} />
        <div className="slot-identity-text">
          <h3 className="slot-name" title={player.name}>{player.name}</h3>
          <div className="slot-meta">
            {radar.position_group} · {Math.round(radar.minutes)} min
          </div>
        </div>
      </div>

      {/* The Performance Score is the reason to look at this card, so it is set like a
          scoreboard figure rather than as one of two equal-weight fields. */}
      <div className="slot-score">
        <span className="slot-score-value" style={{ color: themeColor }}>
          {score.performance_score.toFixed(1)}
        </span>
        <div className="slot-score-side">
          <span className="slot-score-label">Performance score</span>
          <span className="slot-score-style">Plays like {styleText}</span>
        </div>
      </div>

      <div className="hero-target-bar">
        <span className={isActive ? "target-active-text" : "target-muted-text"}>
          {isActive ? "● Picks land here" : "Click to send picks here"}
        </span>
      </div>
    </div>
  );
}
