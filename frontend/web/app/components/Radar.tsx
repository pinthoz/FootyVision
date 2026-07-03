"use client";

export type RadarSeries = { name: string; color: string; values: number[] };

const SIZE = 380;
const R = SIZE / 2 - 54;
const CX = SIZE / 2;
const CY = SIZE / 2;

function point(i: number, n: number, valuePct: number): [number, number] {
  const angle = -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const rr = (R * valuePct) / 100;
  return [CX + rr * Math.cos(angle), CY + rr * Math.sin(angle)];
}

export default function Radar({ axes, series }: { axes: string[]; series: RadarSeries[] }) {
  const n = axes.length;
  const rings = [25, 50, 75, 100];

  return (
    <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width="100%" style={{ maxWidth: SIZE }}>
      {/* grid rings */}
      {rings.map((ring) => (
        <polygon
          key={ring}
          points={axes
            .map((_, i) => point(i, n, ring).join(","))
            .join(" ")}
          fill="none"
          stroke="var(--border)"
          strokeWidth={1}
        />
      ))}
      {/* spokes + labels */}
      {axes.map((label, i) => {
        const [x, y] = point(i, n, 100);
        const [lx, ly] = point(i, n, 118);
        return (
          <g key={label}>
            <line x1={CX} y1={CY} x2={x} y2={y} stroke="var(--border)" strokeWidth={1} />
            <text
              x={lx}
              y={ly}
              fontSize={10.5}
              fill="var(--muted)"
              textAnchor={lx > CX + 4 ? "start" : lx < CX - 4 ? "end" : "middle"}
              dominantBaseline="middle"
            >
              {label}
            </text>
          </g>
        );
      })}
      {/* series polygons */}
      {series.map((s) => {
        const pts = s.values.map((v, i) => point(i, n, v).join(",")).join(" ");
        return (
          <g key={s.name}>
            <polygon points={pts} fill={s.color} fillOpacity={0.18} stroke={s.color} strokeWidth={2} />
            {s.values.map((v, i) => {
              const [x, y] = point(i, n, v);
              return <circle key={i} cx={x} cy={y} r={2.5} fill={s.color} />;
            })}
          </g>
        );
      })}
    </svg>
  );
}
