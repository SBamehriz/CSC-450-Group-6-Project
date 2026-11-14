// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type GpuCheck, type Job } from '../api/client';
import { open } from '../test-utils';
import ComputerPage from './ComputerPage';

const GB = 2 ** 30;

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
  vi.spyOn(api, 'listSpeedTests').mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
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

it('chooses light, dark or this computer, light when nothing is kept', async () => {
  open(<ComputerPage />);
  const light = screen.getByRole('button', { name: 'Light' });
  expect(light.getAttribute('aria-pressed')).toBe('true');
  fireEvent.click(screen.getByRole('button', { name: 'Dark' }));
  expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  expect(localStorage.getItem('forge.appearance')).toBe('dark');
  fireEvent.click(screen.getByRole('button', { name: 'Match this computer' }));
  expect(document.documentElement.hasAttribute('data-theme')).toBe(false);
  expect(localStorage.getItem('forge.appearance')).toBe('system');
  fireEvent.click(light);
  expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  expect(localStorage.getItem('forge.appearance')).toBeNull();
});

it('names a speed test with no shape', async () => {
  vi.mocked(api.listSpeedTests).mockResolvedValue([
    {
      id: 'test-1',
      preset: null,
      shape: null,
      sequences_per_step: 4,
      device_name: 'CPU',
      tokens_per_sec: 1200,
      peak_memory_bytes: null,
      created_at: '2026-11-01T00:00:00Z',
    } as never,
  ]);
  open(<ComputerPage />);
  expect(await screen.findByText('A standard small model')).toBeTruthy();
  expect(screen.queryByLabelText('Size')).toBeNull();
  expect(screen.queryByText('How long would it take?')).toBeNull();
});

it('runs a speed test and shows its progress', async () => {
  vi.spyOn(api, 'startSpeedTest').mockResolvedValue(job('pending'));
  vi.spyOn(api, 'getJob').mockResolvedValue(job('running'));
  open(<ComputerPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Run speed test' }));
  expect(await screen.findByText('Timing training steps')).toBeTruthy();
  await waitFor(() =>
    expect(api.startSpeedTest).toHaveBeenCalledWith({ sequences_per_step: 4, device: 'auto' }),
  );
});
