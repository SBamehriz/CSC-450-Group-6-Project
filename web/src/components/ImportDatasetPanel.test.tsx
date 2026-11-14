// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type Job } from '../api/client';
import { open } from '../test-utils';
import ImportDatasetPanel from './ImportDatasetPanel';

beforeEach(() => {
  vi.spyOn(api, 'listJobs').mockResolvedValue({ items: [], total: 0 } as never);
  vi.spyOn(api, 'inspectHfDataset').mockResolvedValue({
    canonical_id: 'roneneldan/TinyStories',
    sha: 'c0ffee',
    configs: [
      {
        name: 'default',
        splits: [{ name: 'train', num_examples: 2_119_719 }],
        columns: ['id', 'text', 'messages'],
        text_columns: ['text'],
        chat_columns: ['messages'],
      },
    ],
  } as never);
  vi.spyOn(api, 'importDataset').mockResolvedValue({ id: 'job-1', status: 'pending' } as never);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

async function inspect() {
  open(<ImportDatasetPanel bucketId="bucket-1" onImportQueued={() => {}} />);
  fireEvent.change(screen.getByPlaceholderText(/e\.g\. imdb/), {
    target: { value: 'roneneldan/TinyStories' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect dataset' }));
  await screen.findByText('roneneldan/TinyStories');
}

it('imports the whole split by default, grouped into files', async () => {
  await inspect();
  expect(screen.getByLabelText('The whole split (2,119,719 rows)')).toHaveProperty('checked', true);
  expect(screen.getByText(/About 1,060 rows go in each file/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Start import' }));
  await waitFor(() => expect(vi.mocked(api.importDataset)).toHaveBeenCalled());
  const [bucketId, payload] = vi.mocked(api.importDataset).mock.calls[0];
  expect(bucketId).toBe('bucket-1');
  expect(payload).toEqual({
    dataset: 'roneneldan/TinyStories',
    config: 'default',
    split: 'train',
    text_column: 'text',
  });
});

it('offers text and conversation columns, and only the first rows', async () => {
  await inspect();
  const options = screen.getAllByRole('option').map((option) => option.textContent);
  expect(options).toContain('messages (conversations)');
  expect(options).not.toContain('id');
  fireEvent.change(screen.getByDisplayValue('text'), { target: { value: 'messages' } });
  fireEvent.click(screen.getByLabelText('Only the first N rows'));
  fireEvent.change(screen.getByLabelText('How many rows'), { target: { value: '300' } });
  fireEvent.click(screen.getByRole('button', { name: 'Start import' }));
  await waitFor(() =>
    expect(vi.mocked(api.importDataset)).toHaveBeenCalledWith(
      'bucket-1',
      expect.objectContaining({ text_column: 'messages', limit: 300 }),
    ),
  );
});

it('offers to try a stopped import again', async () => {
  const stopped = {
    id: 'job-9',
    job_type: 'dataset_import',
    status: 'cancelled',
    payload: { target_id: 'bucket-1' },
    progress: { message: 'Stopped' },
    error: null,
    attempts: 1,
    cancel_requested: false,
    started_at: null,
    completed_at: null,
    created_at: '2026-11-01T00:00:00Z',
  } satisfies Job;
  vi.mocked(api.listJobs).mockResolvedValue({ items: [stopped], total: 1 });
  const change = vi.spyOn(api, 'changeJob').mockResolvedValue({ ...stopped, status: 'pending' });
  open(<ImportDatasetPanel bucketId="bucket-1" />);
  fireEvent.click(await screen.findByRole('button', { name: 'Try again' }));
  expect(screen.getByText('Try again carries on where it stopped.')).toBeTruthy();
  await waitFor(() => expect(change).toHaveBeenCalledWith('job-9', 'retry'));
});
