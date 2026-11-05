import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

import {
  api,
  type BenchmarkResult,
  type CompareModel,
  type Model,
  type PromptRun,
} from '../api/client';
import JobStatus, { useJob } from '../components/JobStatus';
import { Card, CardHead, Empty, ErrorBox, Notice } from '../components/ui';
import { compactNumber, relativeDate } from '../lib/format';
import { jobBusy, percent } from '../lib/measure';

const NOT_YET = 'Not measured yet';

function lossText(value: number | null): string {
  return value === null ? NOT_YET : value.toFixed(3);
}

export default function ComparePage() {
  const models = useQuery({ queryKey: ['models', 'all'], queryFn: api.listAllModels });
  const [first, setFirst] = useState('');
  const [second, setSecond] = useState('');
  const chosen = [first, second].filter(Boolean);
  const compared = useQuery({
    queryKey: ['compare', chosen],
    queryFn: () => api.compare(chosen),
    enabled: chosen.length > 0,
  });
  const list = models.data?.items ?? [];

  return (
    <>
      <div className="page-intro">
        <h1 className="h1">Compare models</h1>
        <p className="lede">
          Put two models side by side: their size and loss, benchmark scores with how sure each
          score is, and their answers to the same prompts.
        </p>
      </div>

      <Card>
        <CardHead title="Models" />
        <div className="card-body">
          {models.error ? (
            <ErrorBox message={models.error.message} onRetry={() => void models.refetch()} />
          ) : models.isPending ? (
            <p className="muted" role="status">
              Loading models…
            </p>
          ) : !list.length ? (
            <Empty
              title="No models yet"
              hint="Make one on the Models page, or import one from Hugging Face there."
            />
          ) : (
            <div className="dataset-form">
              <ModelSelect label="First model" models={list} value={first} onChange={setFirst} />
              <ModelSelect
                label="Second model"
                models={list.filter((m) => m.id !== first)}
                value={second}
                onChange={setSecond}
                optional
              />
            </div>
          )}
        </div>
      </Card>

      {compared.error ? (
        <ErrorBox message={compared.error.message} onRetry={() => void compared.refetch()} />
      ) : compared.data ? (
        <>
          <SideBySide models={compared.data.models} />
          {compared.data.models.map((side) => (
            <BenchmarkCard key={side.id} side={side} />
          ))}
          <PromptsCard models={compared.data.models} runs={compared.data.prompt_runs} />
        </>
      ) : chosen.length ? (
        <p className="muted" role="status">
          Loading the comparison…
        </p>
      ) : null}
    </>
  );
}

function ModelSelect({
  label,
  models,
  value,
  onChange,
  optional,
}: {
  label: string;
  models: Model[];
  value: string;
  onChange: (next: string) => void;
  optional?: boolean;
}) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      <select
        className="input"
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">{optional ? 'None' : 'Choose a model'}</option>
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}
            {m.source === 'hub' ? ' (imported)' : ''}
          </option>
        ))}
      </select>
    </label>
  );
}

function SideBySide({ models }: { models: CompareModel[] }) {
  const rows: [string, (m: CompareModel) => string][] = [
    ['Where it came from', (m) => (m.source === 'hub' ? 'Imported' : 'Trained here')],
    ['Weights', (m) => compactNumber(m.parameters)],
    ['Vocabulary', (m) => `${m.vocab_size.toLocaleString()} word pieces`],
    ['Context', (m) => `${m.ctx_len.toLocaleString()} word pieces`],
    ['Licence', (m) => m.license ?? (m.source === 'hub' ? 'Unknown' : 'Yours')],
    ['Word pieces trained on', (m) => (m.tokens_seen ? compactNumber(m.tokens_seen) : 'None here')],
    ['Best validation loss', (m) => lossText(m.best_val_loss)],
    ['Bits per byte', (m) => lossText(m.bits_per_byte)],
  ];
  return (
    <Card>
      <CardHead title="Side by side" />
      <div className="scroll-x">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">
                <span className="muted">What</span>
              </th>
              {models.map((m) => (
                <th key={m.id} scope="col">
                  {m.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, value]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                {models.map((m) => (
                  <td key={m.id}>{value(m)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="card-body hint">
        Loss is only comparable between models that read the same vocabulary. Bits per byte does not
        depend on the vocabulary.
      </p>
    </Card>
  );
}

function BenchmarkCard({ side }: { side: CompareModel }) {
  const queryClient = useQueryClient();
  const plan = useQuery({
    queryKey: ['benchmarkPlan', side.id],
    queryFn: () => api.benchmarkPlan(side.id),
  });
  const [limit, setLimit] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, () => {
    void queryClient.invalidateQueries({ queryKey: ['compare'] });
  });
  const start = useMutation({
    mutationFn: () =>
      api.startBenchmark({
        model_id: side.id,
        limit: limit ? Number(limit) : null,
        device: 'auto',
      }),
    onSuccess: (started) => setJobId(started.id),
  });
  const busy = start.isPending || jobBusy(job.data);
  const allowed = plan.data?.allowed ?? [];
  const refused = side.benchmark?.refused.length
    ? side.benchmark.refused
    : (plan.data?.refused ?? []);

  return (
    <Card>
      <CardHead title={`Benchmarks: ${side.name}`} />
      <div className="card-body stack">
        {plan.error ? (
          <ErrorBox message={plan.error.message} onRetry={() => void plan.refetch()} />
        ) : null}
        {allowed.length ? (
          <form
            className="dataset-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!busy) start.mutate();
            }}
          >
            <label className="field">
              <span className="label">Questions per benchmark (optional)</span>
              <input
                type="number"
                className="input"
                aria-label="Questions per benchmark"
                min={1}
                placeholder="All of them"
                value={limit}
                onChange={(event) => setLimit(event.target.value)}
              />
              <span className="hint">A few hundred gives a quick look with wide intervals.</span>
            </label>
            <div>
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? 'Answering…' : `Run ${allowed.length} benchmarks`}
              </button>
            </div>
          </form>
        ) : null}
        {start.error ? <Notice tone="bad">{start.error.message}</Notice> : null}
        <JobStatus job={job.data} />
        {side.benchmark ? (
          <Scores result={side.benchmark} />
        ) : allowed.length ? (
          <p className="muted">No benchmark results yet.</p>
        ) : null}
        {refused.length ? (
          <details className="disclosure" open={!allowed.length}>
            <summary>
              Not run on this size <span className="muted">({refused.length})</span>
            </summary>
            <ul>
              {refused.map((row) => (
                <li key={row.task}>
                  <strong>{row.name}</strong>: {row.reason}
                </li>
              ))}
            </ul>
          </details>
        ) : null}
      </div>
    </Card>
  );
}

export function Scores({ result }: { result: BenchmarkResult }) {
  return (
    <>
      <div className="scroll-x">
        <table className="table">
          <thead>
            <tr>
              <th>Benchmark</th>
              <th className="num">Score</th>
              <th>95% interval</th>
              <th className="num">Chance</th>
              <th className="num">Questions</th>
            </tr>
          </thead>
          <tbody>
            {result.scores.map((score) => (
              <tr key={score.task}>
                <td>{score.name}</td>
                <td className="num">
                  <strong>{percent(score.score)}</strong>
                </td>
                <td>
                  {score.low !== null && score.high !== null
                    ? `${percent(score.low)} to ${percent(score.high)}`
                    : 'Not known'}
                </td>
                <td className="num">{percent(score.chance, 0)}</td>
                <td className="num">{score.questions?.toLocaleString() ?? 'Not known'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint" style={{ margin: 0 }}>
        A score whose interval reaches down to chance is not better than guessing.
        {result.limit
          ? ` Quick look: at most ${result.limit.toLocaleString()} questions each.`
          : ''}
        {result.finished_at ? ` Measured ${relativeDate(result.finished_at)}.` : ''}
      </p>
    </>
  );
}

function PromptsCard({ models, runs }: { models: CompareModel[]; runs: PromptRun[] }) {
  const queryClient = useQueryClient();
  const [prompts, setPrompts] = useState('');
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [length, setLength] = useState(64);
  const [seed, setSeed] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, () => {
    void queryClient.invalidateQueries({ queryKey: ['compare'] });
  });
  const lines = prompts
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  const start = useMutation({
    mutationFn: () =>
      api.comparePrompts({
        model_ids: models.map((m) => m.id),
        prompts: lines,
        temperature,
        top_k: topK,
        max_new_tokens: length,
        seed,
        device: 'auto',
      }),
    onSuccess: (started) => setJobId(started.id),
  });
  const busy = start.isPending || jobBusy(job.data);
  const two = models.length === 2;

  return (
    <Card>
      <CardHead title="Same prompts" />
      <div className="card-body stack">
        {!two ? (
          <p className="muted">Choose a second model to ask both the same prompts.</p>
        ) : (
          <form
            className="stack"
            onSubmit={(event) => {
              event.preventDefault();
              if (lines.length && !busy) start.mutate();
            }}
          >
            <label className="field">
              <span className="label">Prompts, one per line</span>
              <textarea
                className="input"
                aria-label="Prompts, one per line"
                rows={4}
                value={prompts}
                onChange={(event) => setPrompts(event.target.value)}
              />
              <span className="hint">
                Both models get the same settings and the same seed, so the only difference is the
                model.
              </span>
            </label>
            <div className="dataset-form">
              <NumberInput
                label="Temperature"
                value={temperature}
                onChange={setTemperature}
                step={0.1}
              />
              <NumberInput label="Top-k" value={topK} onChange={setTopK} />
              <NumberInput
                label="Answer length (word pieces)"
                value={length}
                onChange={setLength}
              />
              <NumberInput label="Seed" value={seed} onChange={setSeed} />
            </div>
            {lines.length > 20 ? (
              <Notice tone="warn">Keep it to 20 prompts at a time.</Notice>
            ) : null}
            {start.error ? <Notice tone="bad">{start.error.message}</Notice> : null}
            <JobStatus job={job.data} />
            <div>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!lines.length || lines.length > 20 || busy}
              >
                {busy ? 'Asking…' : 'Compare answers'}
              </button>
            </div>
          </form>
        )}
        {runs.map((run) => (
          <PromptAnswers key={run.job_id} run={run} />
        ))}
      </div>
    </Card>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (next: number) => void;
  step?: number;
}) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      <input
        type="number"
        className="input"
        aria-label={label}
        step={step}
        min={0}
        value={Number.isFinite(value) ? value : ''}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export function PromptAnswers({ run }: { run: PromptRun }) {
  const settings = run.settings as {
    temperature?: number;
    top_k?: number;
    max_new_tokens?: number;
    seed?: number;
  };
  return (
    <div className="stack">
      <p className="muted" style={{ margin: 0 }}>
        Temperature {settings.temperature}, top-k {settings.top_k}, up to {settings.max_new_tokens}{' '}
        word pieces, seed {settings.seed}
        {run.finished_at ? `, ${relativeDate(run.finished_at)}` : ''}.
      </p>
      <div className="scroll-x">
        <table className="table">
          <thead>
            <tr>
              <th>Prompt</th>
              {run.model_names.map((name) => (
                <th key={name}>{name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {run.rows.map((row, index) => (
              <tr key={index}>
                <td>{row.prompt}</td>
                {row.answers.map((answer, side) => (
                  <td key={side} style={{ whiteSpace: 'pre-wrap' }}>
                    {answer || <span className="muted">(nothing)</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
