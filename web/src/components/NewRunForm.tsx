import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import type { FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, type RunCreate, type TrainSettings } from '../api/client';
import { compactNumber } from '../lib/format';
import { DEFAULT_SETTINGS, trainingTokens, usableBuilds } from '../lib/training';
import EstimateBox from './EstimateBox';
import { Card, CardHead, Field, Notice } from './ui';

type Length = 'tokens' | 'steps';

const ADVANCED: { key: keyof TrainSettings; label: string; hint: string; step?: number }[] = [
  {
    key: 'batch_size',
    label: 'Sequences per step',
    hint: 'How many text windows each step learns from.',
  },
  {
    key: 'micro_batch_size',
    label: 'Sequences per pass',
    hint: '0 picks the most that fit in memory. Lower it if the card runs out of memory.',
  },
  {
    key: 'learning_rate',
    label: 'Learning rate',
    hint: 'How big each correction is.',
    step: 0.0001,
  },
  { key: 'warmup_steps', label: 'Warm-up steps', hint: '-1 picks a twentieth of the run.' },
  { key: 'weight_decay', label: 'Weight decay', hint: 'Keeps weights small.', step: 0.01 },
  { key: 'grad_clip', label: 'Gradient clip', hint: 'Caps very large corrections.', step: 0.1 },
  {
    key: 'eval_every',
    label: 'Check every (steps)',
    hint: 'How often it measures held-back text.',
  },
  { key: 'save_every', label: 'Save every (steps)', hint: 'How often it keeps a save point.' },
];

function NumberField({
  label,
  value,
  onChange,
  hint,
  step = 1,
  min,
}: {
  label: string;
  value: number;
  onChange: (next: number) => void;
  hint?: string;
  step?: number;
  min?: number;
}) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      <input
        type="number"
        className="input"
        aria-label={label}
        value={Number.isFinite(value) ? value : ''}
        step={step}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint ? <span className="hint">{hint}</span> : null}
    </label>
  );
}

export default function NewRunForm() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const models = useQuery({ queryKey: ['models', 'all'], queryFn: api.listAllModels });
  const builds = useQuery({ queryKey: ['builds'], queryFn: api.listBuilds });
  const vocabularies = useQuery({ queryKey: ['tokenizers'], queryFn: api.listTokenizers });

  const [name, setName] = useState('');
  const [modelId, setModelId] = useState('');
  const [buildId, setBuildId] = useState('');
  const [length, setLength] = useState<Length>('tokens');
  const [tokens, setTokens] = useState(10_000_000);
  const [settings, setSettings] = useState<TrainSettings>(DEFAULT_SETTINGS);
  const [device, setDevice] = useState<RunCreate['device']>('auto');
  const [seed, setSeed] = useState(0);

  const trainable = (models.data?.items ?? []).filter((m) => m.source !== 'hub');
  const model = trainable.find((m) => m.id === modelId);
  const choices = usableBuilds(model, builds.data?.items ?? [], vocabularies.data?.items ?? []);
  const build = choices.find((b) => b.id === buildId);
  const setTokensInSet = trainingTokens(build);
  const contextLength = model?.ctx_len ?? 0;
  const steps =
    length === 'steps'
      ? settings.max_steps
      : contextLength
        ? Math.max(1, Math.ceil(tokens / (settings.batch_size * contextLength)))
        : 0;
  const totalTokens = steps * settings.batch_size * contextLength;

  const set = (key: keyof TrainSettings) => (value: number) =>
    setSettings((now) => ({ ...now, [key]: value }));

  const start = useMutation({
    mutationFn: async () => {
      const run = await api.createRun({
        name: name.trim(),
        model_id: modelId,
        build_id: buildId,
        device,
        seed,
        train_tokens: length === 'tokens' ? tokens : null,
        settings,
      });
      return api.changeRun(run.id, 'start');
    },
    onSuccess: (run) => {
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
      navigate(`/runs/${run.id}`);
    },
  });

  const ready = Boolean(name.trim() && model && build && steps > 0);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (ready) start.mutate();
  };

  return (
    <Card>
      <CardHead title="Start a training run" />
      <form className="card-body stack" onSubmit={submit}>
        <div className="dataset-form">
          <Field label="Run name" value={name} onChange={setName} maxLength={64} />
          <label className="field">
            <span className="label">Model</span>
            <select
              className="input"
              aria-label="Model"
              value={modelId}
              onChange={(event) => {
                setModelId(event.target.value);
                setBuildId('');
              }}
            >
              <option value="">Choose a model</option>
              {trainable.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({compactNumber(m.param_count)} weights)
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="label">Training set</span>
            <select
              className="input"
              aria-label="Training set"
              value={buildId}
              disabled={!model}
              onChange={(event) => setBuildId(event.target.value)}
            >
              <option value="">{model ? 'Choose a training set' : 'Choose a model first'}</option>
              {choices.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        {model && !choices.length && !builds.isPending ? (
          <Notice tone="warn">
            No ready training set uses this model&apos;s vocabulary. Make one on the Training sets
            page with the same vocabulary.
          </Notice>
        ) : null}

        <div className="dataset-form">
          <label className="field">
            <span className="label">Train for</span>
            <select
              className="input"
              aria-label="Train for"
              value={length}
              onChange={(event) => setLength(event.target.value as Length)}
            >
              <option value="tokens">A number of word pieces</option>
              <option value="steps">A number of steps</option>
            </select>
          </label>
          {length === 'tokens' ? (
            <NumberField label="Word pieces" value={tokens} onChange={setTokens} min={1} />
          ) : (
            <NumberField
              label="Steps"
              value={settings.max_steps}
              onChange={set('max_steps')}
              min={1}
            />
          )}
        </div>

        {model && steps > 0 ? (
          <p className="muted" role="status">
            {steps.toLocaleString()} steps, {compactNumber(totalTokens)} word pieces
            {setTokensInSet
              ? ` (about ${(totalTokens / setTokensInSet).toFixed(1)} passes over the training set)`
              : ''}
            .
          </p>
        ) : null}

        {model && totalTokens > 0 ? (
          <EstimateBox
            ask={{
              shape: {
                n_layer: model.n_layer,
                d_model: model.d_model,
                n_head: model.n_head,
                ctx_len: model.ctx_len,
                vocab_size: model.vocab_size,
              },
              train_tokens: totalTokens,
              sequences_per_step: settings.micro_batch_size || settings.batch_size,
            }}
          />
        ) : null}

        <details className="disclosure">
          <summary>Advanced settings</summary>
          <div className="dataset-form">
            {ADVANCED.map((item) => (
              <NumberField
                key={item.key}
                label={item.label}
                hint={item.hint}
                step={item.step}
                value={settings[item.key]}
                onChange={set(item.key)}
              />
            ))}
            <label className="field">
              <span className="label">Where it runs</span>
              <select
                className="input"
                aria-label="Where it runs"
                value={device}
                onChange={(event) => setDevice(event.target.value as RunCreate['device'])}
              >
                <option value="auto">The graphics card if there is one</option>
                <option value="cuda">The graphics card</option>
                <option value="cpu">The CPU</option>
              </select>
            </label>
            <NumberField
              label="Seed"
              value={seed}
              onChange={setSeed}
              min={0}
              hint="The same seed and settings give the same run."
            />
          </div>
        </details>

        {start.error ? <Notice tone="bad">{start.error.message}</Notice> : null}

        <div>
          <button type="submit" className="btn btn-primary" disabled={!ready || start.isPending}>
            {start.isPending ? 'Starting…' : 'Start training'}
          </button>
        </div>
      </form>
    </Card>
  );
}
