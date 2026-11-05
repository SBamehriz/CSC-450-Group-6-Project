// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import {
  api,
  type Checkpoint,
  type DatasetBuild,
  type Model,
  type Run,
  type Tokenizer,
} from '../api/client';
import NewRunForm from '../components/NewRunForm';
import { DEFAULT_SETTINGS, secondsLeft, usableBuilds } from '../lib/training';
import { open } from '../test-utils';
import RunDetailPage from './RunDetailPage';

const vocabulary = (id: string, size: number): Tokenizer => ({
  id,
  name: `${id} pieces`,
  dataset_id: null,
  dataset_name: 'Notes',
  requested_vocab_size: size,
  vocab_size: size,
  trained_on: {},
  status: 'ready',
  error: null,
  created_at: '2026-10-20T00:00:00Z',
});
const build = (id: string, tokenizer: string, status = 'ready'): DatasetBuild => ({
  id,
  name: `${id} set`,
  tokenizer_id: tokenizer,
  dataset_id: null,
  dataset_name: 'Notes',
  status,
  stats: {
    train: { documents: 3, tokens: 100_000, bytes: 400_000 },
    validation: { documents: 1, tokens: 10_000, bytes: 40_000 },
  },
  error: null,
  created_at: '2026-10-21T00:00:00Z',
});
const model = (id: string, tokenizer: string | null, vocab = 812): Model => ({
  id,
  name: `${id} model`,
  description: '',
  preset: 'nano',
  n_layer: 4,
  d_model: 128,
  n_head: 4,
  ctx_len: 256,
  vocab_size: vocab,
  dropout: 0,
  param_count: 900_000,
  config: {},
  tokenizer_id: tokenizer,
  source: 'forge',
  created_at: '2026-10-22T00:00:00Z',
});

const vocabularies = [vocabulary('notes', 812), vocabulary('wiki', 4096)];
const builds = [build('mine', 'notes'), build('other', 'wiki'), build('busy', 'notes', 'building')];

const run: Run = {
  id: 'run-1',
  name: 'First run',
  model_id: 'tied',
  model_name: 'tied model',
  model_preset: 'nano',
  dataset_id: null,
  dataset_name: 'Notes',
  build_id: 'mine',
  build_name: 'mine set',
  seed: 0,
  tokens_seen: 0,
  settings: DEFAULT_SETTINGS,
  job_id: null,
  job_status: null,
  status: 'pending',
  device: 'auto',
  batch_size: 16,
  learning_rate: 6e-4,
  max_steps: 1000,
  current_step: 0,
  loss: null,
  eval_loss: null,
  best_val_loss: null,
  tokens_per_sec: null,
  metrics: {},
  hyperparams: {},
  error: null,
  started_at: null,
  completed_at: null,
  created_at: '2026-10-23T10:00:00Z',
};
const save: Checkpoint = {
  id: 'save-1',
  run_id: 'run-1',
  step: 400,
  epoch: 0,
  loss: 3.2,
  val_loss: 3.4,
  uri: 'runs/run-1/step-00000400.pt',
  size_bytes: 12_000_000,
  is_best: true,
  metrics: {},
  sha256: 'abc',
  created_at: '2026-10-23T10:10:00Z',
};

beforeEach(() => {
  vi.spyOn(api, 'getRunMetrics').mockResolvedValue({ run_id: 'run-1', items: [], total: 0 });
  vi.spyOn(api, 'estimate').mockResolvedValue({
    parameters: 900_000,
    train_tokens: 10_000_000,
    speed_source: 'none',
    speed_test_id: null,
    device_name: null,
    tokens_per_sec: null,
    hours: null,
    memory_bytes: 2 ** 30,
    free_bytes: null,
    fits: null,
    suggestion: null,
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('offers only ready training sets made with the model vocabulary', () => {
  const tied = model('tied', 'notes');
  expect(usableBuilds(tied, builds, vocabularies).map((b) => b.id)).toEqual(['mine']);
  const bySize = model('old', null, 4096);
  expect(usableBuilds(bySize, builds, vocabularies).map((b) => b.id)).toEqual(['other']);
  expect(usableBuilds(undefined, builds, vocabularies)).toEqual([]);
});

it('estimates time left from the speed so far', () => {
  const going = {
    ...run,
    current_step: 100,
    max_steps: 300,
    tokens_seen: 409_600,
    tokens_per_sec: 4096,
  };
  expect(secondsLeft(going)).toBe(200);
  expect(secondsLeft(run)).toBeNull();
});

it('the form lists matching training sets, creates the run and starts it', async () => {
  vi.spyOn(api, 'listAllModels').mockResolvedValue({
    items: [model('tied', 'notes'), model('old', null, 4096)],
    total: 2,
  });
  vi.spyOn(api, 'listBuilds').mockResolvedValue({ items: builds, total: 3 });
  vi.spyOn(api, 'listTokenizers').mockResolvedValue({ items: vocabularies, total: 2 });
  const create = vi.spyOn(api, 'createRun').mockResolvedValue(run);
  const change = vi.spyOn(api, 'changeRun').mockResolvedValue({ ...run, job_status: 'pending' });

  open(
    <Routes>
      <Route path="/history" element={<NewRunForm />} />
      <Route path="/runs/:runId" element={<p>Run page</p>} />
    </Routes>,
    '/history',
  );

  fireEvent.change(screen.getByLabelText('Run name'), { target: { value: 'First run' } });
  const picker = screen.getByLabelText('Model');
  await within(picker).findByText(/tied model/);
  fireEvent.change(picker, { target: { value: 'tied' } });

  const sets = screen.getByLabelText('Training set');
  await waitFor(() => expect(within(sets).queryByText('mine set')).toBeTruthy());
  expect(within(sets).queryByText('other set')).toBeNull();
  expect(within(sets).queryByText('busy set')).toBeNull();
  fireEvent.change(sets, { target: { value: 'mine' } });

  fireEvent.change(screen.getByLabelText('Word pieces'), { target: { value: '409600' } });
  expect(screen.getByRole('status').textContent).toMatch(/100 steps/);

  fireEvent.click(screen.getByRole('button', { name: 'Start training' }));
  await screen.findByText('Run page');
  expect(create).toHaveBeenCalledWith(
    expect.objectContaining({ model_id: 'tied', build_id: 'mine', train_tokens: 409_600 }),
  );
  expect(change).toHaveBeenCalledWith('run-1', 'start');
});

function showRun(current: Run, saves: Checkpoint[] = []) {
  vi.spyOn(api, 'getRun').mockResolvedValue(current);
  vi.spyOn(api, 'listCheckpoints').mockResolvedValue(saves);
  return open(
    <Routes>
      <Route path="/runs/:runId" element={<RunDetailPage />} />
    </Routes>,
    '/runs/run-1',
  );
}

it('a new run shows Start and starts on click', async () => {
  const change = vi.spyOn(api, 'changeRun').mockResolvedValue({ ...run, job_status: 'pending' });
  showRun(run);
  fireEvent.click(await screen.findByRole('button', { name: 'Start training' }));
  await waitFor(() => expect(change).toHaveBeenCalledWith('run-1', 'start'));
  expect(await screen.findByRole('button', { name: 'Stop' })).toBeTruthy();
});

it('a training run shows Stop, its numbers and plain status', async () => {
  showRun({
    ...run,
    status: 'running',
    job_status: 'running',
    current_step: 100,
    max_steps: 300,
    tokens_seen: 409_600,
    tokens_per_sec: 4096,
  });
  expect(await screen.findByRole('button', { name: 'Stop' })).toBeTruthy();
  expect(screen.getByText('Training')).toBeTruthy();
  expect(screen.getByText('100 of 300')).toBeTruthy();
  expect(screen.getByText('About 3 min')).toBeTruthy();
});

it('a stopped run with a save point offers Resume and a download', async () => {
  showRun(
    { ...run, status: 'failed', current_step: 412, error: 'The disk filled up while saving.' },
    [save],
  );
  expect(await screen.findByRole('button', { name: 'Resume' })).toBeTruthy();
  expect(screen.getByText('The disk filled up while saving.')).toBeTruthy();
  const link = await screen.findByRole('link', { name: 'Download the save point at step 400' });
  expect(link.getAttribute('href')).toBe('/api/checkpoints/save-1/download');
  expect(screen.getByText('Best so far')).toBeTruthy();
});

it('a finished run has no action button', async () => {
  showRun({ ...run, status: 'completed', current_step: 1000 }, [save]);
  await screen.findByText('Finished');
  expect(screen.queryByRole('button', { name: /Start|Stop|Resume/ })).toBeNull();
});
