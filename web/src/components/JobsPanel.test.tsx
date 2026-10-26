// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { api, type Job } from '../api/client';
import { open } from '../test-utils';
import JobsPanel from './JobsPanel';

const job: Job = {
  id: 'job-1',
  job_type: 'tokenizer_train',
  status: 'running',
  payload: { target_id: 'pieces', target_name: 'Notes pieces' },
  progress: { completed: 2, total: 3, message: 'Reading training text' },
  error: null,
  attempts: 1,
  cancel_requested: false,
  started_at: '2026-10-23T00:00:00Z',
  completed_at: null,
  created_at: '2026-10-23T00:00:00Z',
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('shows a running job with its progress and cancels it', async () => {
  vi.spyOn(api, 'listJobs').mockResolvedValue({ items: [job], total: 1 });
  const change = vi.spyOn(api, 'changeJob').mockResolvedValue({ ...job, cancel_requested: true });
  open(<JobsPanel />);
  expect(await screen.findByText('Reading training text')).toBeTruthy();
  expect(screen.getByText('2 / 3 documents')).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Notes pieces' }).getAttribute('href')).toBe(
    '/vocabularies',
  );
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  await waitFor(() => expect(change).toHaveBeenCalledWith('job-1', 'cancel'));
});

it('offers retry on a failed job and shows its reason', async () => {
  vi.spyOn(api, 'listJobs').mockResolvedValue({
    items: [
      {
        ...job,
        job_type: 'dataset_build',
        status: 'failed',
        error: 'Could not finish this job: disk full',
      },
    ],
    total: 1,
  });
  const change = vi.spyOn(api, 'changeJob').mockResolvedValue({ ...job, status: 'pending' });
  open(<JobsPanel />);
  expect(await screen.findByText('Could not finish this job: disk full')).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Notes pieces' }).getAttribute('href')).toBe(
    '/training-sets',
  );
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  await waitFor(() => expect(change).toHaveBeenCalledWith('job-1', 'retry'));
});

it('says canceling while a cancel is on its way, and shows the empty state', async () => {
  vi.spyOn(api, 'listJobs').mockResolvedValueOnce({
    items: [{ ...job, cancel_requested: true }],
    total: 1,
  });
  open(<JobsPanel />);
  expect(await screen.findByText('canceling')).toBeTruthy();
  expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true);
  cleanup();
  vi.spyOn(api, 'listJobs').mockResolvedValue({ items: [], total: 0 });
  open(<JobsPanel />);
  expect(await screen.findByText(/No jobs yet/)).toBeTruthy();
});
