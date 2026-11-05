import type { MetricPoint } from '../api/client';

const WIDTH = 640;
const HEIGHT = 240;
const PAD = { top: 12, right: 16, bottom: 32, left: 52 };

type Series = { label: string; className: string; points: [number, number][] };

function path(points: [number, number][], x: (v: number) => number, y: (v: number) => number) {
  return points
    .map(([a, b], index) => `${index ? 'L' : 'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`)
    .join(' ');
}

// training and validation loss by step
export default function LossChart({ points }: { points: MetricPoint[] }) {
  const series: Series[] = [
    {
      label: 'Training loss',
      className: 'chart-line chart-line-train',
      points: points
        .filter((p) => p.loss !== null && Number.isFinite(p.loss))
        .map((p) => [p.step, p.loss as number]),
    },
    {
      label: 'Validation loss',
      className: 'chart-line chart-line-val',
      points: points
        .filter((p) => p.val_loss !== null && Number.isFinite(p.val_loss))
        .map((p) => [p.step, p.val_loss as number]),
    },
  ];
  const all = series.flatMap((s) => s.points);
  if (all.length < 2) {
    return <p className="muted">The chart appears after the first few steps.</p>;
  }

  const steps = all.map(([a]) => a);
  const values = all.map(([, b]) => b);
  const [x0, x1] = [Math.min(...steps), Math.max(...steps)];
  let [y0, y1] = [Math.min(...values), Math.max(...values)];
  if (y1 - y0 < 1e-6) [y0, y1] = [y0 - 0.5, y1 + 0.5];
  const x = (v: number) =>
    PAD.left + ((v - x0) / Math.max(1, x1 - x0)) * (WIDTH - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - (v - y0) / (y1 - y0)) * (HEIGHT - PAD.top - PAD.bottom);
  const ticks = [y0, (y0 + y1) / 2, y1];
  const last = (s: Series) => s.points[s.points.length - 1]?.[1];

  return (
    <figure className="loss-chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={series
          .filter((s) => s.points.length)
          .map((s) => `${s.label} ${last(s)?.toFixed(3)} at step ${s.points.at(-1)?.[0]}`)
          .join(', ')}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className="chart-grid"
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text className="chart-tick" x={PAD.left - 8} y={y(tick) + 4} textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        <text className="chart-tick" x={PAD.left} y={HEIGHT - 8}>
          Step {x0.toLocaleString()}
        </text>
        <text className="chart-tick" x={WIDTH - PAD.right} y={HEIGHT - 8} textAnchor="end">
          Step {x1.toLocaleString()}
        </text>
        {series.map((s) =>
          s.points.length > 1 ? (
            <path key={s.label} className={s.className} d={path(s.points, x, y)} />
          ) : s.points.length === 1 ? (
            <circle
              key={s.label}
              className={s.className}
              cx={x(s.points[0][0])}
              cy={y(s.points[0][1])}
              r={3}
            />
          ) : null,
        )}
      </svg>
      <figcaption className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i className={`${s.className}-key`} aria-hidden="true" />
            {s.label}
            {s.points.length ? `: ${last(s)?.toFixed(3)}` : ': not measured yet'}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
