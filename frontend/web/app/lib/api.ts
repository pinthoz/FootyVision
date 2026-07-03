// Thin typed client for the FootyVision FastAPI backend.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export type Player = { id: number; name: string; country: string | null };

export type RadarMetric = { value: number; percentile: number };
export type Radar = {
  player_id: number;
  name: string;
  position_group: string;
  minutes: number;
  metrics: Record<string, RadarMetric>;
};

export type Similar = {
  player_id: number;
  name: string;
  primary_position: string | null;
  similarity: number;
};

export type Score = {
  name: string;
  position_group: string;
  performance_score: number;
  style_profile: Record<string, number>;
  breakdown: { metric: string; percentile: number; contribution: number }[];
};

export type SearchRow = {
  player_id: number;
  name: string;
  competition: string | null;
  primary_position: string | null;
  stats: Record<string, number>;
};

export type AssistantResult = {
  answer: string;
  sources: { player_id: number; name: string }[];
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  searchPlayers: (q: string) =>
    get<Player[]>(`/players?search=${encodeURIComponent(q)}&limit=25`),
  radar: (id: number) => get<Radar>(`/players/${id}/radar`),
  similar: (id: number) =>
    get<{ results: Similar[] }>(`/players/${id}/similar?top_n=8`).then((r) => r.results),
  score: (id: number) => get<Score>(`/players/${id}/score`),
  nlSearch: (query: string) =>
    post("/search", { query }).then(async (r) => ({
      status: r.status,
      data: r.ok ? await r.json() : null,
    })),
  report: (id: number) =>
    post(`/players/${id}/report`, {}).then(async (r) => ({
      status: r.status,
      data: r.ok ? await r.json() : null,
    })),
  assistant: (question: string) =>
    post("/assistant", { question, k: 6 }).then(async (r) => ({
      status: r.status,
      data: r.ok ? ((await r.json()) as AssistantResult) : null,
    })),
};

// Curated radar axes (subset of the 17 metrics) with readable labels.
export const RADAR_AXES: [string, string][] = [
  ["xg_per90", "xG"],
  ["shots_per90", "Shots"],
  ["assists_per90", "Assists"],
  ["progressive_passes_per90", "Prog Passes"],
  ["passes_per90", "Passes"],
  ["dribbles_per90", "Dribbles"],
  ["ball_recoveries_per90", "Recoveries"],
  ["tackles_per90", "Tackles"],
  ["interceptions_per90", "Interceptions"],
];
