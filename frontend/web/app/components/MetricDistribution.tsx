"use client";

import { useEffect, useState } from "react";
import { Beeswarm, ThemeProvider } from "@withqwerty/campos-react";
import { PITCH_CHART_THEME } from "../lib/chartTheme";
import { RADAR_AXES, api, type Distribution } from "../lib/api";

// A percentile says where a player ranks. It does not say what he is ranked against:
// the 96th percentile can be out on his own or packed in with twenty others. This plots
// every player in the position group so that shape is visible.

type Marker = { playerId: number; name: string; color: string };

export default function MetricDistribution({
  positionGroup,
  markers,
}: {
  positionGroup: string;
  markers: Marker[];
}) {
  const [metric, setMetric] = useState("xg_per90");
  const [data, setData] = useState<Distribution | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let stale = false;
    setData(null);
    setFailed(false);
    api
      .distribution(metric, positionGroup)
      .then((d) => !stale && setData(d))
      .catch(() => !stale && setFailed(true));
    return () => {
      stale = true;
    };
  }, [metric, positionGroup]);

  const label = RADAR_AXES.find(([key]) => key === metric)?.[1] ?? metric;
  const marked = new Map(markers.map((m) => [m.playerId, m]));

  return (
    <div>
      <label>Where he sits in the {positionGroup} field</label>
      <div className="filters">
        {RADAR_AXES.map(([key, axisLabel]) => (
          <button
            key={key}
            className={`chip ${metric === key ? "active" : ""}`}
            onClick={() => setMetric(key)}
          >
            {axisLabel}
          </button>
        ))}
      </div>

      {failed && <div className="chartnote">Could not load the distribution.</div>}

      {data && (
        <div className="chart-frame" style={{ marginTop: 12 }}>
          <ThemeProvider value={PITCH_CHART_THEME}>
          <Beeswarm
            groups={[
              {
                id: positionGroup,
                // The axis already names the metric and the note below carries the
                // sample size; a group label here only duplicates them, and its
                // gutter pushed the swarm out of the frame.
                label: "",
                values: data.values.map((v) => {
                  const mark = marked.get(v.player_id);
                  return {
                    id: String(v.player_id),
                    value: v.value,
                    label: v.name,
                    ...(mark
                      ? {
                          highlight: {
                            label: mark.name.split(" ").slice(-1)[0],
                            color: mark.color,
                            radius: 6,
                            stroke: "var(--panel)",
                            strokeWidth: 2,
                          },
                        }
                      : {}),
                  };
                }),
              },
            ]}
            metric={{
              label: `${label} per 90`,
              format: (v: number) => v.toFixed(2),
            }}
            populationColor={{ mode: "uniform", color: "rgba(233, 240, 230, 0.30)" }}
            dotRadius={2.6}
            layout={{ viewBoxHeight: 150, groupLabelSize: 0, outerPadding: 10 }}
            // The library paints highlight labels with currentColor and exposes no
            // per-highlight text style, so two players cannot be told apart by colour
            // inside the chart. The key below does that job instead.
            labelStrategy="none"
            maxWidth={620}
          />
          </ThemeProvider>
        </div>
      )}

      <div className="chart-key">
        {markers.map((m) => (
          <span key={m.playerId} className="chart-key-item" style={{ color: m.color }}>
            <span className="dot" style={{ background: m.color }} />
            {m.name}
          </span>
        ))}
      </div>

      <div className="chartnote">
        {data
          ? `Each dot is one of ${data.count} ${positionGroup} player-seasons. Hover a dot for the player behind it.`
          : "Loading the field…"}
      </div>
    </div>
  );
}
