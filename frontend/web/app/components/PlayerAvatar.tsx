"use client";

import { useEffect, useState } from "react";
import { resolvePlayerPhoto } from "../lib/photos";

type PlayerAvatarProps = {
  name: string;
  size?: "sm" | "md" | "lg" | "xl" | "hero";
  themeColor?: string;
  className?: string;
};

export default function PlayerAvatar({
  name,
  size = "md",
  themeColor = "var(--a)",
  className = "",
}: PlayerAvatarProps) {
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(false);

    resolvePlayerPhoto(name)
      .then((url) => {
        if (active) {
          setPhotoUrl(url);
          setLoading(false);
        }
      })
      .catch(() => {
        if (active) {
          setPhotoUrl(null);
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [name]);

  const initials = getInitials(name);
  const sizeClass = `avatar-${size}`;

  if (photoUrl && !error) {
    return (
      <div className={`player-avatar-wrapper ${sizeClass} ${className}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={photoUrl}
          alt={name}
          className="player-avatar-img"
          onError={() => setError(true)}
          loading="lazy"
        />
        <div className="avatar-ring" style={{ borderColor: themeColor }} />
      </div>
    );
  }

  return (
    <div
      className={`player-avatar-wrapper ${sizeClass} monogram-fallback ${className}`}
      style={{
        background: `linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)`,
        borderColor: themeColor,
      }}
      title={name}
    >
      <span className="avatar-initials" style={{ color: themeColor }}>
        {initials}
      </span>
      {loading && <div className="avatar-pulse" />}
    </div>
  );
}

function getInitials(name: string): string {
  if (!name) return "⚽";
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
