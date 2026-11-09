import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { api, type EstimateRequest, type Preset, type Shape } from '../api/client';
import { compactNumber } from '../lib/format';
import { budgetWords, memoryWords, speedWords, timeWords } from '../lib/measure';
import { Notice } from './ui';

export type Size = { preset: string } | { shape: Shape };

export const CUSTOM_SHAPE: Shape = {
  n_layer: 4,
  d_model: 128,
  n_head: 4,
  ctx_len: 512,
  vocab_size: 16384,
};

function useSettled<T>(value: T, ms = 400): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return settled;
}

export function SizePicker({
  presets,
  value,
  onChange,
  label = 'Size',
}: {
  presets: Preset[];
  value: Size;
  onChange: (next: Size) => void;
  label?: string;
}) {
  const custom = 'shape' in value;
  const shape = custom ? value.shape : CUSTOM_SHAPE;
  const set = (key: keyof Shape) => (next: number) =>
    onChange({ shape: { ...shape, [key]: next } });
  return (
    <>
      <label className="field">
        <span className="label">{label}</span>
        <select
          className="input"
          aria-label={label}
          value={custom ? 'custom' : value.preset}
          onChange={(event) =>
            onChange(
              event.target.value === 'custom'
                ? { shape: CUSTOM_SHAPE }
                : { preset: event.target.value },
            )
          }
        >
          {presets.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} ({compactNumber(p.parameters)} weights)
            </option>
          ))}
          <option value="custom">A shape of my own</option>
        </select>
      </label>
      {custom ? (
        <>
          <ShapeNumber label="Blocks" value={shape.n_layer} onChange={set('n_layer')} />
          <ShapeNumber label="Width" value={shape.d_model} onChange={set('d_model')} />
          <ShapeNumber label="Attention heads" value={shape.n_head} onChange={set('n_head')} />
          <ShapeNumber label="Context" value={shape.ctx_len} onChange={set('ctx_len')} />
          <ShapeNumber
            label="Vocabulary size"
            value={shape.vocab_size}
            onChange={set('vocab_size')}
          />
        </>
      ) : null}
    </>
  );
}

function ShapeNumber({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (next: number) => void;
}) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      <input
        type="number"
        className="input"
        aria-label={label}
        min={1}
        value={Number.isFinite(value) ? value : ''}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

// hours and memory for one size, from a speed test on this computer
export default function EstimateBox({ ask }: { ask: EstimateRequest | null }) {
  const settled = useSettled(ask);
  const estimate = useQuery({
    queryKey: ['estimate', settled],
    queryFn: () => api.estimate(settled as EstimateRequest),
    enabled: Boolean(settled && settled.train_tokens > 0),
  });
  if (!ask) return null;
  if (estimate.error) return <Notice tone="bad">{estimate.error.message}</Notice>;
  if (!estimate.data) {
    return (
      <p className="muted" role="status">
        Working out the time and memory…
      </p>
    );
  }
  const data = estimate.data;
  const time = timeWords(data);
  return (
    <div className="stack estimate" aria-label="Time and memory">
      {time ? <p style={{ margin: 0 }}>{time}</p> : null}
      <p className="muted" style={{ margin: 0 }}>
        {speedWords(data)}
      </p>
      <p style={{ margin: 0 }}>{memoryWords(data)}</p>
      {budgetWords(data) ? <p style={{ margin: 0 }}>{budgetWords(data)}</p> : null}
      {data.fits === false && data.suggestion ? (
        <Notice tone="warn">{data.suggestion}</Notice>
      ) : null}
    </div>
  );
}
