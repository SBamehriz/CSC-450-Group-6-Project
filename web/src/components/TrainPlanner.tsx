import './train-planner.css';

import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useId, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';

import { api, type Bucket, type Model, type Plan, type PlanRequest } from '../api/client';
import { compactNumber, gigabytes, plural } from '../lib/format';
import { duration } from '../lib/training';
import { Card, CardHead, Chip, Empty, Field, Notice } from './ui';

// what you can set by hand, forge fills in the rest
type Key =
  | 'n_layer'
  | 'd_model'
  | 'n_head'
  | 'ctx_len'
  | 'vocab_size'
  | 'train_tokens'
  | 'batch_size'
  | 'learning_rate'
  | 'warmup_steps'
  | 'weight_decay'
  | 'dropout';
type Line = { key: Key; label: string; whole: boolean; compact?: boolean };

const SHAPE: Line[] = [
  { key: 'n_layer', label: 'Layers', whole: true },
  { key: 'd_model', label: 'Width', whole: true },
  { key: 'ctx_len', label: 'Context', whole: true },
  { key: 'vocab_size', label: 'Vocabulary', whole: true },
];
const TRAINING: Line[] = [
  { key: 'train_tokens', label: 'Word pieces to read', whole: true, compact: true },
  { key: 'batch_size', label: 'Sequences per step', whole: true },
  { key: 'learning_rate', label: 'Learning rate', whole: false },
];
const ADVANCED: Line[] = [
  { key: 'n_head', label: 'Attention heads', whole: true },
  { key: 'warmup_steps', label: 'Warm-up steps', whole: true },
  { key: 'weight_decay', label: 'Weight decay', whole: false },
  { key: 'dropout', label: 'Dropout', whole: false },
];
// a fine-tune keeps its base's shape
const SHAPED: Key[] = ['n_layer', 'd_model', 'n_head', 'ctx_len', 'vocab_size'];

function planned(plan: Plan, key: Key): number {
  if (key in plan.shape) return plan.shape[key as keyof Plan['shape']];
  if (key === 'train_tokens') return plan.train_tokens;
  if (key === 'dropout') return plan.dropout;
  return plan.settings[key as keyof Plan['settings']];
}

const UNITS: Record<string, number> = { k: 1e3, m: 1e6, b: 1e9, t: 1e12 };

// takes "135M", "2B", "1,000" or "6e-4", null if it's not a number
export function parseAmount(text: string): number | null {
  const match = text
    .trim()
    .replace(/[,\s_]/g, '')
    .match(/^(\d*\.?\d+(?:e[-+]?\d+)?)([kmbt])?$/i);
  if (!match) return null;
  return Number(match[1]) * (match[2] ? UNITS[match[2].toLowerCase()] : 1);
}

// so we don't ask for a plan on every key press
function useSettled<T>(value: T, ms = 300): [T, boolean] {
  const key = JSON.stringify(value);
  const [settled, setSettled] = useState(key);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(key), ms);
    return () => clearTimeout(timer);
  }, [key, ms]);
  return [JSON.parse(settled) as T, settled === key];
}

// names have to be unique, so add a number
function freeName(name: string, models: Model[]): string {
  const taken = new Set(models.map((model) => model.name));
  let out = name.slice(0, 64);
  for (let n = 2; taken.has(out); n += 1) out = `${name.slice(0, 60)} ${n}`;
  return out;
}

// keeps what you typed until you leave the box
function AmountField({
  label,
  name = label,
  value,
  onChange,
  onUndo,
  hint,
  compact,
  placeholder,
  end,
}: {
  label: string;
  name?: string;
  value: number | null;
  onChange: (next: number | null) => void;
  onUndo?: () => void;
  hint?: string;
  compact?: boolean;
  placeholder?: string;
  end?: ReactNode;
}) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const shown =
    draft ?? (value === null ? '' : compact ? compactNumber(value) : value.toLocaleString('en-US'));
  const bad = draft !== null && draft.trim() !== '' && parseAmount(draft) === null;
  const said = bad ? 'Type a number, like 5M or 1,000.' : hint;
  return (
    <div className={`field${onUndo ? ' planner-changed' : ''}`}>
      <div className="planner-label">
        <label className="label" htmlFor={id}>
          {label}
        </label>
        {onUndo ? (
          <>
            <Chip tone="idle">Changed</Chip>
            <button
              type="button"
              className="btn btn-link"
              aria-label={`Undo ${name}`}
              onClick={() => {
                setDraft(null);
                onUndo();
              }}
            >
              Undo
            </button>
          </>
        ) : null}
        {end ? <span className="planner-end">{end}</span> : null}
      </div>
      <input
        id={id}
        className="input"
        inputMode="decimal"
        autoComplete="off"
        aria-label={name}
        aria-invalid={bad || undefined}
        aria-describedby={said ? `${id}-hint` : undefined}
        value={shown}
        placeholder={placeholder}
        onChange={(event) => {
          const text = event.target.value;
          setDraft(text);
          if (!text.trim()) onChange(null);
          else {
            const parsed = parseAmount(text);
            if (parsed !== null) onChange(parsed);
          }
        }}
        onBlur={() => {
          if (!bad) setDraft(null);
        }}
      />
      {said ? (
        <span className={`hint${bad ? ' planner-bad' : ''}`} id={`${id}-hint`}>
          {said}
        </span>
      ) : null}
    </div>
  );
}

const TIME_SAID: Record<Plan['time']['source'], string> = {
  measured: 'From a speed test on this computer.',
  scaled: 'Only a guess, scaled from a speed test of another size.',
  none: 'Only a rough guess.',
};

function Summary({ plan, bucket }: { plan: Plan; bucket: Bucket | null }) {
  const { memory, time, data } = plan;
  return (
    <aside className="planner-summary stack" aria-label="Summary">
      <dl className="run-numbers">
        <div>
          <dt>Parameters</dt>
          <dd>{compactNumber(plan.parameters)}</dd>
          {plan.fine_tune ? <dd className="hint">The size of the model it starts from.</dd> : null}
        </div>
        <div>
          <dt>Memory</dt>
          <dd>
            {gigabytes(memory.needed_bytes)}
            {memory.free_bytes !== null ? ` of ${gigabytes(memory.free_bytes)} free` : ' needed'}
          </dd>
          <dd className="hint">
            {memory.cards
              ? `On ${memory.card_name ?? 'the graphics card'}, ${plural(memory.sequences_per_pass, 'sequence')} per pass.`
              : 'No graphics card was found, so it would train on the processor, slowly.'}
          </dd>
        </div>
        <div>
          <dt>Time</dt>
          <dd>{time.hours !== null ? `About ${duration(time.hours * 3600)}` : 'Not known yet'}</dd>
          <dd className="hint">
            {TIME_SAID[time.source]}
            {time.source === 'measured' ? null : (
              <>
                {' '}
                Run the speed test in <Link to="/settings">Settings</Link> for real hours.
              </>
            )}
          </dd>
        </div>
        {bucket ? (
          <div>
            <dt>Text in {bucket.name}</dt>
            <dd>
              {data.tokens !== null
                ? `${data.exact ? '' : 'About '}${compactNumber(data.tokens)} word pieces`
                : 'Not counted yet'}
            </dd>
            {data.passes !== null ? (
              <dd className="hint">
                Training reads it{' '}
                {data.passes < 10 ? data.passes.toFixed(1) : Math.round(data.passes)} times.
                {data.enough === false ? ' Not much text for this size.' : ''}
              </dd>
            ) : null}
          </div>
        ) : null}
      </dl>
      {plan.blockers.map((said) => (
        <Notice key={said} tone="bad">
          {said}
        </Notice>
      ))}
      {plan.warnings.map((said) => (
        <Notice key={said} tone="warn">
          {said}
        </Notice>
      ))}
    </aside>
  );
}

export default function TrainPlanner() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const models = useQuery({ queryKey: ['models', 'all'], queryFn: api.listAllModels });
  const buckets = useQuery({ queryKey: ['buckets'], queryFn: api.listBuckets });

  const [from, setFrom] = useState(params.get('from') ?? '');
  const [bucketId, setBucketId] = useState(params.get('data') ?? '');
  const [by, setBy] = useState<'size' | 'time'>('size');
  const [parameters, setParameters] = useState<number | null>(null);
  const [hours, setHours] = useState<number | null>(null);
  const [mine, setMine] = useState<Partial<Record<Key, number>>>({});
  const [name, setName] = useState<string | null>(null);

  const allModels = models.data?.items ?? [];
  const bases = allModels.filter((m) => m.source === 'forge' && m.ready);
  const base = bases.find((m) => m.id === from) ?? null;
  const collections = buckets.data?.items ?? [];
  const bucket = collections.find((b) => b.id === bucketId) ?? collections[0] ?? null;

  // only send what was changed by hand
  const request: PlanRequest = {
    bucket_id: bucket?.id ?? null,
    base_model_id: base?.id ?? null,
    device: 'auto',
  };
  if (!base && by === 'size' && parameters) request.parameters = parameters;
  if (!base && by === 'time' && hours) request.hours = hours;
  for (const [key, value] of Object.entries(mine))
    if (!(base && SHAPED.includes(key as Key))) request[key as Key] = value;

  // wait for the lists so the first plan knows the collection
  const [asked, settled] = useSettled(buckets.isPending || models.isPending ? null : request);
  const plan = useQuery({
    queryKey: ['plan', asked],
    queryFn: () => api.plan(asked as PlanRequest),
    placeholderData: keepPreviousData,
    enabled: asked !== null,
  });
  const current = settled && !plan.isPlaceholderData && !plan.isFetching && !plan.error;

  const suggested = base
    ? `${base.name} on ${bucket?.name ?? 'a collection'}`
    : `${bucket?.name ?? 'Model'} ${compactNumber(plan.data?.parameters ?? parameters ?? 0)}`;
  const runName = name ?? freeName(suggested, allModels);

  const start = useMutation({
    mutationFn: (ready: Plan) =>
      api.train({
        name: runName.trim(),
        bucket_id: (bucket as Bucket).id,
        base_model_id: base?.id ?? null,
        shape: ready.shape,
        dropout: ready.dropout,
        settings: ready.settings,
        seed: 0,
        device: 'auto',
      }),
    onSuccess: (run) => {
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
      void queryClient.invalidateQueries({ queryKey: ['models'] });
      navigate(`/runs/${run.id}`);
    },
  });

  const set = (key: Key, value: number | null) =>
    setMine((old) => {
      const next = { ...old };
      if (value === null) delete next[key];
      else next[key] = value;
      return next;
    });

  const fields = (lines: Line[], data: Plan) =>
    lines.map((line) => (
      <AmountField
        key={line.key}
        label={line.label}
        value={mine[line.key] ?? planned(data, line.key)}
        compact={line.compact}
        hint={data.why[line.key]}
        onChange={(value) =>
          set(line.key, value === null || !line.whole ? value : Math.round(value))
        }
        onUndo={mine[line.key] !== undefined ? () => set(line.key, null) : undefined}
      />
    ));

  const ready = Boolean(bucket && plan.data && current && !plan.data.blockers.length);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (ready && runName.trim() && plan.data) start.mutate(plan.data);
  };

  return (
    <Card>
      <CardHead
        title="Start a training run"
        end={
          plan.isFetching && plan.data ? (
            <span className="muted" role="status">
              Planning…
            </span>
          ) : null
        }
      />
      <form className="card-body stack" onSubmit={submit}>
        <div className="dataset-form">
          <label className="field">
            <span className="label">Start from</span>
            <select
              className="input"
              aria-label="Start from"
              value={base ? from : ''}
              onChange={(event) => setFrom(event.target.value)}
            >
              <option value="">A new model</option>
              {bases.map((m) => (
                <option key={m.id} value={m.id}>
                  Fine-tune {m.name} ({compactNumber(m.param_count)} weights)
                </option>
              ))}
            </select>
            <span className="hint">
              {base
                ? `Keeps the size of ${base.name} and learns more.`
                : 'Or fine-tune a model you trained, to teach it something new.'}
            </span>
          </label>

          {bucket ? (
            <label className="field">
              <span className="label">Learn from</span>
              <select
                className="input"
                aria-label="Learn from"
                value={bucket.id}
                onChange={(event) => setBucketId(event.target.value)}
              >
                {collections.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name} ({b.kind}, {compactNumber(b.stats.est_tokens)} word pieces)
                  </option>
                ))}
              </select>
              <span className="hint">
                {bucket.kind === 'chat'
                  ? 'Conversations: the model learns to answer.'
                  : 'Plain text: the model learns to carry it on.'}
              </span>
            </label>
          ) : null}

          {base ? null : (
            <AmountField
              key={by}
              label="How big?"
              name={by === 'size' ? 'Parameters' : 'Hours you have'}
              value={by === 'size' ? (parameters ?? plan.data?.parameters ?? null) : hours}
              compact={by === 'size'}
              onChange={by === 'size' ? setParameters : setHours}
              onUndo={by === 'size' && parameters !== null ? () => setParameters(null) : undefined}
              placeholder={by === 'size' ? 'Forge picks' : 'Hours, like 2'}
              hint={
                by === 'size'
                  ? (plan.data?.why.parameters ?? 'Any size, like 5M, 135M or 2B.')
                  : 'The hours you have. Forge picks the size that fits.'
              }
              end={
                <span className="seg" aria-label="Plan by">
                  {(['size', 'time'] as const).map((option) => (
                    <button
                      key={option}
                      type="button"
                      className={by === option ? 'on' : undefined}
                      aria-pressed={by === option}
                      onClick={() => setBy(option)}
                    >
                      {option === 'size' ? 'By size' : 'By time'}
                    </button>
                  ))}
                </span>
              }
            />
          )}
        </div>

        {buckets.error ? <Notice tone="bad">{buckets.error.message}</Notice> : null}
        {models.error ? <Notice tone="bad">{models.error.message}</Notice> : null}
        {!buckets.isPending && !buckets.error && !bucket ? (
          <Empty
            title="No collections yet"
            hint="A model learns from a collection of text. Add one on the Data page first."
            action={
              <Link className="btn" to="/data">
                Add text
              </Link>
            }
          />
        ) : null}
        {plan.error ? <Notice tone="bad">{plan.error.message}</Notice> : null}

        {plan.data ? (
          <div className="planner">
            <div className="stack">
              <h3 className="planner-group">Shape</h3>
              {base ? (
                <>
                  <dl className="run-numbers">
                    {[...SHAPE, ADVANCED[0]].map((line) => (
                      <div key={line.key}>
                        <dt>{line.label}</dt>
                        <dd>{planned(plan.data, line.key).toLocaleString('en-US')}</dd>
                      </div>
                    ))}
                  </dl>
                  <p className="hint">Set by {base.name}, the model it starts from.</p>
                </>
              ) : (
                <div className="dataset-form">{fields(SHAPE, plan.data)}</div>
              )}
              <h3 className="planner-group">Training</h3>
              <div className="dataset-form">{fields(TRAINING, plan.data)}</div>
              <details className="disclosure">
                <summary>Advanced settings</summary>
                <div className="dataset-form">
                  {fields(base ? ADVANCED.slice(1) : ADVANCED, plan.data)}
                </div>
              </details>
            </div>
            <Summary plan={plan.data} bucket={bucket} />
          </div>
        ) : plan.error || !bucket ? null : (
          <p className="muted" role="status">
            Planning…
          </p>
        )}

        {start.error ? <Notice tone="bad">{start.error.message}</Notice> : null}

        <div className="row-end planner-go">
          <Field label="Run name" value={runName} onChange={setName} maxLength={64} />
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!ready || !runName.trim() || start.isPending}
          >
            {start.isPending ? 'Starting…' : base ? 'Start fine-tuning' : 'Start training'}
          </button>
        </div>
      </form>
    </Card>
  );
}
