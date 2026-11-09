import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api } from '../api/client';
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
        columns: ['text'],
        text_columns: ['text'],
      },
    ],
  } as never);
  vi.spyOn(api, 'importDataset').mockResolvedValue({ id: 'job-1', status: 'pending' } as never);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('imports a whole big split, grouped into files', async () => {
  open(<ImportDatasetPanel bucketId="bucket-1" onImportQueued={() => {}} />);
  fireEvent.change(screen.getByPlaceholderText(/e\.g\. imdb/), {
    target: { value: 'roneneldan/TinyStories' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect dataset' }));
  fireEvent.click(await screen.findByRole('button', { name: 'All 2,119,719 rows' }));
  expect(screen.getByText(/About 1,060 rows go in each file/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Start import' }));
  await waitFor(() =>
    expect(vi.mocked(api.importDataset)).toHaveBeenCalledWith(
      'bucket-1',
      expect.objectContaining({ limit: 2_119_719, text_column: 'text' }),
    ),
  );
});
