// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type Checkpoint, type Run } from '../api/client';
import { secondsLeft } from '../lib/training';
import { open } from '../test-utils';
import RunDetailPage from './RunDetailPage';

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
  settings: {
    batch_size: 16,
    micro_batch_size: 0,
    max_steps: 1000,
    learning_rate: 6e-4,
    warmup_steps: -1,
    weight_decay: 0.1,
    grad_clip: 1.0,
    eval_every: 100,
    eval_tokens: 100_000,
    save_every: 200,
  },
  job_id: null,
  job_status: null,
  fine_tune: false,
  stage: 'training',
  stage_message: '',
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
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
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

it('a finished run links to chat, fine-tune and compare its model', async () => {
  showRun({ ...run, status: 'completed', current_step: 1000 }, [save]);
  const href = async (name: string) =>
    (await screen.findByRole('link', { name })).getAttribute('href');
  expect(await href('Chat with it')).toBe('/chat?model=tied');
  expect(await href('Fine-tune it')).toBe('/train?from=tied');
  expect(await href('Compare it')).toBe('/compare?model=tied');
});

it('says Forge is getting the text ready before training starts', async () => {
  showRun({
    ...run,
    job_status: 'pending',
    stage: 'preparing',
    stage_message: 'Making the vocabulary (2 of 3).',
  });
  expect(await screen.findByText('Making the vocabulary (2 of 3).')).toBeTruthy();
  expect(screen.queryByRole('link', { name: 'Chat with it' })).toBeNull();
});

it('a fine-tune says which model it started from', async () => {
  showRun({ ...run, fine_tune: true, base_model_name: 'Letters model' });
  expect(await screen.findByText(/Fine-tuned from Letters model\./)).toBeTruthy();
});
