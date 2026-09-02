"use client";

import { useEffect, useRef, useState } from "react";
import { Player, api } from "../lib/api";
import PlayerAvatar from "./PlayerAvatar";

export const FEATURED_PLAYERS: { id: number; name: string; position: string; desc: string }[] = [
  { id: 5503, name: "Lionel Messi", position: "FWD", desc: "Elite Playmaking & Finishing" },
  { id: 5207, name: "Cristiano Ronaldo", position: "FWD", desc: "Supreme Goalscorer & Aerial Threat" },
  { id: 5487, name: "Antoine Griezmann", position: "FWD", desc: "Dynamic Second Striker & Workrate" },
  { id: 5216, name: "Andrés Iniesta", position: "MID", desc: "Magical Progression & Vision" },
  { id: 5539, name: "Casemiro", position: "MID", desc: "Ball Winning & Defensive Anchor" },
  { id: 6394, name: "Aritz Aduriz", position: "FWD", desc: "Classic Box Striker & Heading" },
  { id: 4353, name: "Aymeric Laporte", position: "DEF", desc: "Ball-Playing Center Back" },
  { id: 4324, name: "Dani Alves", position: "DEF", desc: "Attacking Full-Back & Crossing" },
];

type PlayerPickerModalProps = {
  isOpen: boolean;
  targetSlot: "A" | "B";
  onClose: () => void;
  onSelectPlayer: (p: Player, slot: "A" | "B") => void;
};

export default function PlayerPickerModal({
  isOpen,
  targetSlot,
  onClose,
  onSelectPlayer,
}: PlayerPickerModalProps) {
  const [query, setQuery] = useState("");
  const [roster, setRoster] = useState<Player[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setIsLoading(true);
      api
        .searchPlayers(query.trim())
        .then((data) => {
          setRoster(data);
          setIsLoading(false);
        })
        .catch(() => {
          setRoster([]);
          setIsLoading(false);
        });
      setTimeout(() => inputRef.current?.focus(), 80);
    }
  }, [isOpen, query]);

  if (!isOpen) return null;

  const isA = targetSlot === "A";
  const themeColor = isA ? "var(--a)" : "var(--b)";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="picker-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="picker-modal-header">
          <div>
            <div className="picker-target-pill" style={{ background: isA ? "rgba(184, 142, 45, 0.15)" : "rgba(224, 96, 126, 0.15)", color: themeColor }}>
              Targeting: Player {targetSlot}
            </div>
            <h3 className="picker-title">Select Player for Slot {targetSlot}</h3>
          </div>
          <button className="picker-close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Search Input */}
        <div className="picker-search-bar">
          
          <input
            ref={inputRef}
            type="text"
            value={query}
            placeholder="Type player name (e.g. Messi, Ronaldo, Griezmann)..."
            onChange={(e) => setQuery(e.target.value)}
          />
          {query && (
            <button className="clear-search-btn" onClick={() => setQuery("")}>✕</button>
          )}
        </div>

        {/* Featured Quick Stars */}
        <div className="featured-stars-section">
          <span className="featured-stars-title">Featured Stars:</span>
          <div className="featured-chips-grid">
            {FEATURED_PLAYERS.map((fp) => (
              <button
                key={fp.id}
                className="featured-chip"
                onClick={() => {
                  onSelectPlayer({ id: fp.id, name: fp.name, country: null }, targetSlot);
                  onClose();
                }}
              >
                <PlayerAvatar name={fp.name} size="sm" themeColor={themeColor} />
                <span className="featured-chip-pos">{fp.position}</span>
                <span className="featured-chip-name">{fp.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Available Roster List */}
        <div className="picker-roster-section">
          <div className="roster-header">
            <span>Available Roster {roster.length > 0 ? `(${roster.length} players)` : ""}</span>
            {isLoading && <span className="loading-badge">Loading…</span>}
          </div>

          <div className="picker-roster-list">
            {roster.length === 0 && !isLoading ? (
              <div className="empty-roster-state">
                <p>No players found matching &ldquo;{query}&rdquo;.</p>
                <p className="empty-sub">Try clicking one of the featured stars above or clearing the search.</p>
              </div>
            ) : (
              roster.map((p) => (
                <div
                  key={p.id}
                  className="picker-player-row"
                  onClick={() => {
                    onSelectPlayer(p, targetSlot);
                    onClose();
                  }}
                >
                  <div className="picker-player-left">
                    <PlayerAvatar name={p.name} size="sm" themeColor={themeColor} />
                    <div className="picker-player-info">
                      <span className="picker-name">{p.name}</span>
                      {p.country && <span className="picker-country">{p.country}</span>}
                    </div>
                  </div>
                  <button
                    className="action-btn primary-btn picker-select-btn"
                    style={{ background: themeColor, color: isA ? "#000" : "#fff" }}
                  >
                    Select for {targetSlot}
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
