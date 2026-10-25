// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type Dataset, type DatasetBuild, type Tokenizer } from '../api/client';
import { open } from '../test-utils';
import TrainingSetsPage from './TrainingSetsPage';

const snapshot: Dataset = {
  id: 'saved',
  name: 'Weekly notes',
  description: '',
  bucket_id: null,
  status: 'ready',
  doc_count: 4,
  char_count: 1200,
  token_count: 0,
  config: {},
  artifact_uri: 'datasets/saved.zip',
  created_at: '2026-10-12T00:00:00Z',
  download_ready: true,
  readiness_note: '',
};
const vocabulary: Tokenizer = {
  id: 'pieces',
  name: 'Notes pieces',
  dataset_id: 'saved',
  dataset_name: 'Weekly notes',
  requested_vocab_size: 4096,
  vocab_size: 812,
  trained_on: {},
  status: 'ready',
  error: null,
  created_at: '2026-10-20T00:00:00Z',
};
const build: DatasetBuild = {
  id: 'tokens',
  name: 'Notes tokens',
  tokenizer_id: 'pieces',
  dataset_id: 'saved',
  dataset_name: 'Weekly notes',
  status: 'ready',
  stats: {
    train: { documents: 3, tokens: 12_400, bytes: 50_000 },
    validation: { documents: 1, tokens: 4_100, bytes: 16_000 },
  },
  error: null,
  created_at: '2026-10-21T00:00:00Z',
};

beforeEach(() => {
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [snapshot], total: 1 });
  vi.spyOn(api, 'listTokenizers').mockResolvedValue({ items: [vocabulary], total: 1 });
  vi.spyOn(api, 'listBuilds').mockResolvedValue({ items: [], total: 0 });
  vi.spyOn(api, 'listJobs').mockResolvedValue({ items: [], total: 0 });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('shows loading, then the empty state', async () => {
  open(<TrainingSetsPage />);
  expect(screen.getByText('Loading training sets…')).toBeTruthy();
  expect(await screen.findByText('No training sets yet')).toBeTruthy();
});

it('makes a training set from a snapshot and a ready vocabulary', async () => {
  const create = vi
    .spyOn(api, 'createBuild')
    .mockResolvedValue({ ...build, status: 'pending', stats: {} });
  open(<TrainingSetsPage />);
  fireEvent.change(await screen.findByLabelText('Training set name'), {
    target: { value: 'Notes tokens' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Make training set' }));
  await waitFor(() =>
    expect(create).toHaveBeenCalledWith({
      name: 'Notes tokens',
      dataset_id: 'saved',
      tokenizer_id: 'pieces',
    }),
  );
  expect(await screen.findByText('Training set queued. Follow its progress below.')).toBeTruthy();
});

it('asks for a ready vocabulary first', async () => {
  vi.spyOn(api, 'listTokenizers').mockResolvedValue({
    items: [{ ...vocabulary, status: 'building' }],
    total: 1,
  });
  open(<TrainingSetsPage />);
  expect(await screen.findByRole('link', { name: 'Make a vocabulary' })).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Make training set' })).toBeNull();
});

it('lists training sets with token counts and the manifest', async () => {
  vi.spyOn(api, 'listBuilds').mockResolvedValue({ items: [build], total: 1 });
  open(<TrainingSetsPage />);
  expect(await screen.findByText('12.4K')).toBeTruthy();
  expect(screen.getByText('4.1K')).toBeTruthy();
  expect(screen.getByText('Notes pieces', { selector: 'div' })).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Manifest' }).getAttribute('href')).toBe(
    '/api/dataset-builds/tokens/manifest',
  );
});

it('shows a failed training set’s reason', async () => {
  vi.spyOn(api, 'listBuilds').mockResolvedValue({
    items: [{ ...build, status: 'failed', stats: {}, error: 'The snapshot was removed.' }],
    total: 1,
  });
  open(<TrainingSetsPage />);
  expect(await screen.findByText('The snapshot was removed.')).toBeTruthy();
  expect(screen.queryByRole('link', { name: 'Manifest' })).toBeNull();
});
