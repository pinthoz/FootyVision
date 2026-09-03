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
  position_group?: string;
  similarity: number;
  xg_per90?: number;
  progressive_passes_per90?: number;
  tackles_per90?: number;
  dribbles_per90?: number;
};

export type Score = {
  player_id?: number;
  name: string;
  position_group: string;
  performance_score: number;
  style_profile: Record<string, number>;
  breakdown: { metric: string; weight: number; percentile: number; contribution: number }[];
};

export type RankingRow = {
  player_id: number;
  name: string;
  competition?: string | null;
  position_group: string;
  primary_position: string | null;
  performance_score: number;
};

export type TopFeature = {
  feature: string;
  mean_abs_shap: number;
};

export type ModelInfoResponse = {
  task: string;
  classes: string[];
  test_accuracy: number;
  n_train: number;
  n_test: number;
  top_features?: TopFeature[];
};

export type DistributionPoint = { player_id: number; name: string; value: number };

export type Distribution = {
  metric: string;
  position_group: string | null;
  count: number;
  values: DistributionPoint[];
};

export type SearchRow = {
  player_id: number;
  name: string;
  competition: string | null;
  primary_position: string | null;
  position_group?: string;
  stats: Record<string, number>;
};

export type AssistantResult = {
  answer: string;
  sources: { player_id: number; name: string; score?: number }[];
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
  searchPlayers: (q: string) => {
    const term = q.trim();
    const queryParam = term ? `search=${encodeURIComponent(term)}&` : "";
    return get<Player[]>(`/players?${queryParam}with_stats=true&limit=40`);
  },
  radar: (id: number) => get<Radar>(`/players/${id}/radar`),
  similar: (id: number) =>
    get<{ results: Similar[] }>(`/players/${id}/similar?top_n=10`).then((r) => r.results),
  score: (id: number) => get<Score>(`/players/${id}/score`),
  distribution: (metric: string, positionGroup?: string) =>
    get<Distribution>(
      `/metrics/${metric}/distribution` +
        (positionGroup ? `?position_group=${positionGroup}` : "")
    ),
  rankings: (positionGroup?: string, topN = 20) =>
    get<{ results: RankingRow[] }>(
      `/rankings?top_n=${topN}` + (positionGroup ? `&position_group=${positionGroup}` : "")
    ).then((r) => r.results),
  modelInfo: () => get<ModelInfoResponse>("/talent/model-info"),
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
  coverage: () => get<Coverage>("/coverage"),
};

export type CoverageSeason = {
  competition_id: number;
  competition: string;
  country: string | null;
  season_id: number;
  season: string;
  matches: number;
  teams: number;
  players: number;
  coverage: number;
  complete: boolean;
};

export type CatalogueEntry = {
  competition_id: number;
  season_id: number;
  competition: string;
  country: string | null;
  season: string;
  matches: number;
  teams: number;
  gender: string;
  kind: string;
  complete: boolean;
  loaded: boolean;
};

export type Coverage = {
  competitions: number;
  matches: number;
  players: number;
  seasons: CoverageSeason[];
  catalogue: CatalogueEntry[];
  catalogue_verified: string;
};

export type MetricCategory = "all" | "attack" | "passing" | "defense";

export type MetricMeta = {
  key: string;
  label: string;
  category: "attack" | "passing" | "defense";
  description: string;
};

export const ALL_METRICS: MetricMeta[] = [
  // Attacking
  { key: "goals_per90", label: "Goals", category: "attack", description: "Goals scored per 90 mins" },
  { key: "xg_per90", label: "xG", category: "attack", description: "Expected goals per 90 mins" },
  { key: "shots_per90", label: "Shots", category: "attack", description: "Total shots attempted per 90 mins" },
  { key: "assists_per90", label: "Assists", category: "attack", description: "Goal assists per 90 mins" },
  { key: "dribbles_per90", label: "Dribbles", category: "attack", description: "Take-ons attempted per 90 mins" },
  { key: "dribbles_completed_per90", label: "Dribbles Done", category: "attack", description: "Successful take-ons per 90 mins" },

  // Passing & Ball Progression
  { key: "passes_per90", label: "Passes", category: "passing", description: "Passes attempted per 90 mins" },
  { key: "passes_completed_per90", label: "Passes Comp.", category: "passing", description: "Successful passes per 90 mins" },
  { key: "progressive_passes_per90", label: "Prog. Passes", category: "passing", description: "Forward progressive passes per 90 mins" },
  { key: "carries_per90", label: "Carries", category: "passing", description: "Ball carries per 90 mins" },
  { key: "progressive_carries_per90", label: "Prog. Carries", category: "passing", description: "Progressive ball carries per 90 mins" },

  // Defending & Workrate
  { key: "tackles_per90", label: "Tackles", category: "defense", description: "Tackles made per 90 mins" },
  { key: "interceptions_per90", label: "Interceptions", category: "defense", description: "Interceptions won per 90 mins" },
  { key: "blocks_per90", label: "Blocks", category: "defense", description: "Pass/shot blocks per 90 mins" },
  { key: "clearances_per90", label: "Clearances", category: "defense", description: "Clearances per 90 mins" },
  { key: "ball_recoveries_per90", label: "Recoveries", category: "defense", description: "Loose ball recoveries per 90 mins" },
  { key: "pressures_per90", label: "Pressures", category: "defense", description: "Defensive pressing events per 90 mins" },
];

export const RADAR_PRESETS: { id: string; label: string; axes: [string, string][] }[] = [
  {
    id: "curated",
    label: "Scout Core (9)",
    axes: [
      ["xg_per90", "xG"],
      ["shots_per90", "Shots"],
      ["assists_per90", "Assists"],
      ["progressive_passes_per90", "Prog Passes"],
      ["passes_per90", "Passes"],
      ["dribbles_per90", "Dribbles"],
      ["ball_recoveries_per90", "Recoveries"],
      ["tackles_per90", "Tackles"],
      ["interceptions_per90", "Interceptions"],
    ],
  },
  {
    id: "all",
    label: "All Metrics (17)",
    axes: ALL_METRICS.map((m) => [m.key, m.label]),
  },
  {
    id: "attack",
    label: "Attack & Finishing",
    axes: ALL_METRICS.filter((m) => m.category === "attack").map((m) => [m.key, m.label]),
  },
  {
    id: "passing",
    label: "Distribution & Progression",
    axes: ALL_METRICS.filter((m) => m.category === "passing").map((m) => [m.key, m.label]),
  },
  {
    id: "defense",
    label: "Defending & Pressing",
    axes: ALL_METRICS.filter((m) => m.category === "defense").map((m) => [m.key, m.label]),
  },
];

export const RADAR_AXES: [string, string][] = RADAR_PRESETS[0].axes;

export function getMetricLabel(metricKey: string): string {
  const found = ALL_METRICS.find((m) => m.key === metricKey);
  if (found) return found.label;
  return metricKey.replace(/_per90$/, "").replace(/_/g, " ");
}
