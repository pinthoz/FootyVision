"use client";

import { useEffect, useState } from "react";
import { ScatterPlot, ThemeProvider } from "@withqwerty/campos-react";
import { PITCH_CHART_THEME } from "../lib/chartTheme";
import { RADAR_AXES, api } from "../lib/api";

// Two metrics at once separates things a single ranking conflates: volume from quality.
// A striker high on shots but low on xG is taking bad ones; the radar cannot say that,
// because it shows each metric on its own axis.

type Row = { id: string; name: string; x: number; y: number };
type Marker = { playerId: number; name: string; color: string };

export default function MetricScatter({
  positionGroup,
  markers,
}: {
  positionGroup: string;
  markers: Marker[];
}) {
  const [xMetric, setXMetric] = useState("shots_per90");
  const [yMetric, setYMetric] = useState("xg_per90");
  const [rows, setRows] = useState<Row[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let stale = false;
    setRows(null);
    setFailed(false);
    Promise.all([
      api.distribution(xMetric, positionGroup),
      api.distribution(yMetric, positionGroup),
    ])
      .then(([xs, ys]) => {
        if (stale) return;
        const yById = new Map(ys.values.map((v) => [v.player_id, v.value]));
        setRows(
          xs.values
            .filter((v) => yById.has(v.player_id))
            .map((v) => ({
              id: String(v.player_id),
              name: v.name,
              x: v.value,
              y: yById.get(v.player_id) as number,
            }))
        );
      })
      .catch(() => !stale && setFailed(true));
    return () => {
      stale = true;
    };
  }, [xMetric, yMetric, positionGroup]);

  const labelOf = (metric: string) =>
    RADAR_AXES.find(([key]) => key === metric)?.[1] ?? metric;
  const colourOf = new Map(markers.map((m) => [String(m.playerId), m.color]));

  return (
    <div>
      <label>
        {labelOf(yMetric)} against {labelOf(xMetric)}, all {positionGroup}s
      </label>

      {failed && <div className="chartnote">Could not load the metrics.</div>}

      <div className="scatter-layout">
        <div className="axis-pickers">
          <MetricPicker axis="Y axis" value={yMetric} onChange={setYMetric} />
          <MetricPicker axis="X axis" value={xMetric} onChange={setXMetric} />
        </div>

        {rows && (
          <div className="chart-frame">
          <ThemeProvider value={PITCH_CHART_THEME}>
          <ScatterPlot<Row>
            points={rows}
            idKey="id"
            xKey="x"
            yKey="y"
            labelKey="name"
            xLabel={`${labelOf(xMetric)} per 90`}
            yLabel={`${labelOf(yMetric)} per 90`}
            labelStrategy="manual"
            labelIds={markers.map((m) => String(m.playerId))}
            markers={{
              fill: ({ point }) =>
                (point && colourOf.get(point.id)) ?? "rgba(233, 240, 230, 0.38)",
              radius: ({ point }) => (point && colourOf.has(point.id) ? 5 : 2.6),
            }}
            labelStyle={{
              // The label carries the same identity as its marker, so it wears the
              // same colour rather than the generic text token.
              fill: ({ label }) => colourOf.get(label.id) ?? "var(--text)",
              fontSize: 9,
            }}
          />
            </ThemeProvider>
          </div>
        )}
      </div>

      <div className="chartnote">
        {rows
          ? `${rows.length} ${positionGroup} player-seasons. Up and to the right is more of both.`
          : "Loading the field…"}
      </div>
    </div>
  );
}

function MetricPicker({
  axis,
  value,
  onChange,
}: {
  axis: string;
  value: string;
  onChange: (metric: string) => void;
}) {
  return (
    <div className="axis-picker">
      <span className="scatter-axis-tag">{axis}</span>
      {RADAR_AXES.map(([key, label]) => (
        <button
          key={key}
          className={`chip ${value === key ? "active" : ""}`}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
