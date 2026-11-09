// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type Estimate, type GpuCheck, type Job, type Preset } from '../api/client';
import { open } from '../test-utils';
import ComputerPage from './ComputerPage';

const GB = 2 ** 30;

const nano: Preset = {
  name: 'nano',
  n_layer: 4,
  d_model: 128,
  n_head: 4,
  ctx_len: 512,
  vocab_size: 16384,
  parameters: 2_956_032,
  memory: [],
};

const card: GpuCheck = {
  cards: [
    {
      index: 0,
      name: 'NVIDIA GeForce RTX 3050 Laptop GPU',
      memory_bytes: 4 * GB,
      free_bytes: 3 * GB,
      driver: '595.79',
      compute_capability: '8.6',
    },
  ],
  tool_said: null,
  torch: 'cuda',
  torch_version: '2.14.1+cu130',
  torch_cuda: '13.0',
};

function estimate(change: Partial<Estimate>): Estimate {
  return {
    parameters: 2_956_032,
    train_tokens: 100_000_000,
    speed_source: 'measured',
    speed_test_id: 'test-1',
    device_name: 'NVIDIA GeForce RTX 3050 Laptop GPU',
    tokens_per_sec: 50_000,
    hours: 100_000_000 / 50_000 / 3600,
    memory_bytes: 1 * GB,
    free_bytes: 3 * GB,
    fits: true,
    suggestion: null,
    ...change,
  };
}

const job = (status: string, change: Partial<Job> = {}): Job => ({
  id: 'job-1',
  job_type: 'speed_test',
  status,
  payload: {},
  progress: { completed: 2, total: 7, message: 'Timing training steps' },
  error: null,
  attempts: 1,
  cancel_requested: false,
  result: {},
  started_at: null,
  completed_at: null,
  created_at: '2026-11-01T00:00:00Z',
  ...change,
});

beforeEach(() => {
  vi.spyOn(api, 'checkGpu').mockResolvedValue(card);
  vi.spyOn(api, 'listPresets').mockResolvedValue([nano]);
  vi.spyOn(api, 'listSpeedTests').mockResolvedValue([]);
  vi.spyOn(api, 'estimate').mockResolvedValue(estimate({}));
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('shows the card with its total and free memory', async () => {
  open(<ComputerPage />);
  expect(await screen.findByText('NVIDIA GeForce RTX 3050 Laptop GPU', { selector: 'dd' }));
  expect(screen.getByText('4.0 GB', { selector: 'dd' })).toBeTruthy();
  expect(screen.getByText('3.0 GB', { selector: 'dd' })).toBeTruthy();
  expect(screen.getByText('PyTorch 2.14.1+cu130 with CUDA 13.0')).toBeTruthy();
  expect(await screen.findByText('No speed tests yet')).toBeTruthy();
});

it('says when there is no card or no PyTorch', async () => {
  vi.mocked(api.checkGpu).mockResolvedValue({
    cards: [],
    tool_said: 'NVIDIA-SMI has failed.',
    torch: 'missing',
    torch_version: null,
    torch_cuda: null,
  });
  open(<ComputerPage />);
  expect(await screen.findByText(/No NVIDIA graphics card was found/)).toBeTruthy();
  expect(screen.getByText("The card's driver said: NVIDIA-SMI has failed.")).toBeTruthy();
  expect(screen.getByText(/PyTorch isn't installed/)).toBeTruthy();
});

it('recovers when the check fails', async () => {
  vi.mocked(api.checkGpu).mockRejectedValueOnce(new Error('Server unavailable.'));
  open(<ComputerPage />);
  expect(await screen.findByText('Server unavailable.')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Check again' }));
  expect(await screen.findByText('NVIDIA GeForce RTX 3050 Laptop GPU', { selector: 'dd' }));
});

it('says the hours come from a speed test on this computer', async () => {
  open(<ComputerPage />);
  expect(await screen.findByText(/About 33 min to train 100M word pieces/)).toBeTruthy();
  expect(screen.getByText(/Speed measured on this computer/)).toBeTruthy();
  expect(screen.getByText(/3.0 GB is free on the graphics card right now. It fits./)).toBeTruthy();
});

it('turns the hours you have into a word piece budget', async () => {
  vi.mocked(api.estimate).mockResolvedValue(estimate({ budget_tokens: 59_000_000 }));
  open(<ComputerPage />);
  const hours = await screen.findByLabelText('Hours you have');
  fireEvent.change(hours, { target: { value: '0.5' } });
  await waitFor(() =>
    expect(vi.mocked(api.estimate)).toHaveBeenLastCalledWith(
      expect.objectContaining({ budget_hours: 0.5 }),
    ),
  );
  expect(
    await screen.findByText(
      /In the hours you have it can train about 59M word pieces, 20.0 per parameter/,
    ),
  ).toBeTruthy();
});

it('says when the speed was scaled from another size', async () => {
  vi.mocked(api.estimate).mockResolvedValue(estimate({ speed_source: 'scaled' }));
  open(<ComputerPage />);
  expect(await screen.findByText(/Speed scaled from a speed test of another size/)).toBeTruthy();
});

it('asks for a speed test when there is none, and suggests a fix when it does not fit', async () => {
  vi.mocked(api.estimate).mockResolvedValue(
    estimate({
      speed_source: 'none',
      speed_test_id: null,
      device_name: null,
      tokens_per_sec: null,
      hours: null,
      memory_bytes: 5 * GB,
      fits: false,
      suggestion: 'Use 2 sequences per pass.',
    }),
  );
  open(<ComputerPage />);
  expect(await screen.findByText('Run a speed test to see hours.')).toBeTruthy();
  expect(screen.queryByText(/to train 100M word pieces/)).toBeNull();
  expect(screen.getByText(/only 3.0 GB is free/)).toBeTruthy();
  expect(screen.getByText('Use 2 sequences per pass.')).toBeTruthy();
});

it('runs a speed test and shows its progress', async () => {
  vi.spyOn(api, 'startSpeedTest').mockResolvedValue(job('pending'));
  vi.spyOn(api, 'getJob').mockResolvedValue(job('running'));
  open(<ComputerPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Run speed test' }));
  expect(await screen.findByText('Timing training steps')).toBeTruthy();
  await waitFor(() =>
    expect(api.startSpeedTest).toHaveBeenCalledWith({
      preset: 'nano',
      sequences_per_step: 4,
      device: 'auto',
    }),
  );
});
