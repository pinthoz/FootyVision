"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import {
  ModelInfoResponse,
  Player,
  Radar as RadarData,
  Score,
  getMetricLabel,
} from "../lib/api";
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
  /** The position classifier's own report card, from /talent/model-info. Passed whole
      rather than as a list of features so the copy can cite the real sample size and
      accuracy instead of numbers written into the JSX, which go stale unnoticed. */
  modelInfo?: ModelInfoResponse | null;
};

// The percentage beside "Plays like" reads like a rating unless it says otherwise: it is
// the classifier's confidence that a player's per-90 metrics look like that position
// group, not the position he is listed in. The badge opens the evidence rather than
// asserting it — the metrics that decide position, and where this player sits on each.
const STYLE_HINT = "Why this style? See the metrics that decided it.";

/** The evidence behind "Plays like X", opened as a modal like Dataset Coverage.

    Same shell as that one on purpose: the dashboard already teaches that a badge with an
    overlay behind it is how detail is reached, and a second, differently-shaped popover
    would be a new thing to learn for no reason.

    Rendered through a portal because the card carries `backdrop-filter: blur(16px)`, and
    a filtered element becomes the containing block for `position: fixed` descendants —
    the overlay would centre itself inside the card instead of the viewport. Dataset
    Coverage escapes this only because the header happens not to be filtered. */
function StyleEvidence({
  playerName,
  styleText,
  radar,
  modelInfo,
  themeColor,
  onClose,
}: {
  playerName: string;
  styleText: string;
  radar: RadarData;
  modelInfo: ModelInfoResponse;
  themeColor: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (typeof document === "undefined") return null;

  const features = modelInfo.top_features ?? [];
  const pool = modelInfo.n_train + modelInfo.n_test;
  // Bars are drawn against the strongest metric, not against 1.0: |SHAP| is an absolute
  // effect on the model's output, so its scale means nothing without a reference.
  const strongest = Math.max(...features.map((f) => f.mean_abs_shap), 0.0001);

  return createPortal(
    <div className="modal-backdrop" onClick={onClose}>
      <div className="style-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="picker-modal-header">
          <div>
            <div className="coverage-header-pill" style={{ color: themeColor }}>
              Plays like {styleText}
            </div>
            <h3 className="picker-title">{playerName}</h3>
          </div>
          <button type="button" className="picker-close-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="style-modal-body">
          <p className="why-note">
            The position classifier is an <strong>XGBoost</strong> model trained on the 17
            per-90 metrics. <strong>TreeSHAP</strong> — the exact attribution method for
            tree ensembles — measures how much each metric moves its prediction, and these
            are the five largest by mean |SHAP| over all {pool.toLocaleString()} player
            seasons. They are what decides position <em>in general</em>, not for this player.
          </p>

          <ul className="why-list why-list-head">
            <li>
              <span>Metric</span>
              <span>Weight in the model</span>
              <span className="why-pct">His rank</span>
            </li>
          </ul>

          <ul className="why-list">
            {features.map((f) => {
              const metric = radar.metrics[f.feature];
              const percentile = metric ? Math.round(metric.percentile) : null;
              return (
                <li key={f.feature}>
                  <span className="why-metric">{getMetricLabel(f.feature)}</span>
                  <span className="why-shap">
                    <span className="why-bar">
                      <span
                        style={{
                          width: `${(f.mean_abs_shap / strongest) * 100}%`,
                          background: themeColor,
                        }}
                      />
                    </span>
                    <span className="why-shap-value">{f.mean_abs_shap.toFixed(2)}</span>
                  </span>
                  <span className="why-pct">
                    {percentile === null ? "—" : `${percentile}th`}
                  </span>
                </li>
              );
            })}
          </ul>

          <p className="why-note">
            The right column is his percentile among {radar.position_group}s. A metric only
            explains <em>him</em> when both are high: one the model leans on but he ranks
            low on is a reason it looked elsewhere.
          </p>

          <div className="chartnote">
            Held-out accuracy {(modelInfo.test_accuracy * 100).toFixed(1)}% over{" "}
            {modelInfo.classes.join(" / ")}. Percentiles are within the position group,
            among players above the minutes floor.
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export default function PlayerCard({
  slotId,
  slot,
  isActive,
  onActivate,
  onOpenPicker,
  onPickDirect,
  onClear,
  onSwap,
  modelInfo,
}: PlayerCardProps) {
  const [showWhy, setShowWhy] = useState(false);
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
          <span className="slot-score-style">
            Plays like {styleText}
            {modelInfo && (modelInfo.top_features?.length ?? 0) > 0 && (
              <span className="why-anchor">
                <button
                  type="button"
                  className="info-badge"
                  aria-label={STYLE_HINT}
                  aria-expanded={showWhy}
                  title={STYLE_HINT}
                  onClick={(e) => {
                    // The whole card is a click target that switches the active slot.
                    e.stopPropagation();
                    setShowWhy((open) => !open);
                  }}
                >
                  i
                </button>
                {showWhy && (
                  <StyleEvidence
                    playerName={player.name}
                    styleText={styleText}
                    radar={radar}
                    modelInfo={modelInfo}
                    themeColor={themeColor}
                    onClose={() => setShowWhy(false)}
                  />
                )}
              </span>
            )}
          </span>
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
