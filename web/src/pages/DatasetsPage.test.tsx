// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { api, type Dataset } from '../api/client';
import { bucket, open } from '../test-utils';
import DatasetsPage from './DatasetsPage';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('keeps the bucket from the review link and refreshes dashboard after a snapshot', async () => {
  const second = { ...bucket, id: 'second', name: 'Second source' };
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket, second], total: 2 });
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [], total: 0 });
  const create = vi.spyOn(api, 'createDataset').mockResolvedValue({ id: 'snapshot' } as Dataset);
  const { queryClient } = open(<DatasetsPage />, '/datasets?bucket=second');
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  const selector = await screen.findByLabelText('Source bucket');
  expect((selector as HTMLSelectElement).value).toBe('second');
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Weekly snapshot' } });
  fireEvent.click(screen.getByRole('button', { name: 'Build snapshot' }));
  await waitFor(() =>
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ bucket_id: 'second', name: 'Weekly snapshot' }),
    ),
  );
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['overview'] });
});

it('does not silently build from another bucket when the requested source is ineligible', async () => {
  const short = { ...bucket, id: 'short', stats: { ...bucket.stats, parsed_documents: 1 } };
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket, short], total: 2 });
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [], total: 0 });
  open(<DatasetsPage />, '/datasets?bucket=short');
  const selector = await screen.findByLabelText('Source bucket');
  expect((selector as HTMLSelectElement).value).toBe('');
  expect(screen.getByText(/The requested bucket needs/)).toBeTruthy();
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Weekly snapshot' } });
  expect(
    (screen.getByRole('button', { name: 'Build snapshot' }) as HTMLButtonElement).disabled,
  ).toBe(true);
  fireEvent.change(selector, { target: { value: bucket.id } });
  expect(
    (screen.getByRole('button', { name: 'Build snapshot' }) as HTMLButtonElement).disabled,
  ).toBe(false);
});

const snapshot: Dataset = {
  id: 'saved',
  name: 'Week five',
  description: 'Course notes',
  bucket_id: bucket.id,
  status: 'ready',
  doc_count: 50,
  char_count: 10000,
  token_count: 0,
  artifact_uri: 'datasets/saved.zip',
  config: { train_documents: 45, validation_documents: 5, seed: 42, sha256: 'a'.repeat(64) },
  created_at: bucket.created_at,
};

it('previews the actual source, updates the split target, and explains the archive', async () => {
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket], total: 1 });
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [], total: 0 });
  open(<DatasetsPage />);
  await screen.findByLabelText('Source bucket');
  expect(screen.getByRole('complementary', { name: 'Snapshot preview' }).textContent).toContain(
    '50',
  );
  fireEvent.change(screen.getByLabelText('Validation percent'), { target: { value: '25' } });
  expect(screen.getByText(/75% train · 25% validation target/)).toBeTruthy();
  fireEvent.click(screen.getByRole('checkbox', { name: 'Explain what I’m seeing' }));
  expect(screen.getByText(/is held aside to check it later/)).toBeTruthy();
  expect(screen.getByText('manifest.json')).toBeTruthy();
});

it('blocks invalid split settings and oversized sources before a request', async () => {
  const large = { ...bucket, id: 'large', stats: { ...bucket.stats, parsed_documents: 1001 } };
  const create = vi.spyOn(api, 'createDataset');
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [large, bucket], total: 2 });
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [], total: 0 });
  open(<DatasetsPage />, '/datasets?bucket=large');
  await screen.findByLabelText('Source bucket');
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'New snapshot' } });
  expect(
    (screen.getByRole('button', { name: 'Build snapshot' }) as HTMLButtonElement).disabled,
  ).toBe(true);
  fireEvent.change(screen.getByLabelText('Source bucket'), { target: { value: bucket.id } });
  fireEvent.change(screen.getByLabelText('Validation percent'), { target: { value: '' } });
  expect(
    (screen.getByRole('button', { name: 'Build snapshot' }) as HTMLButtonElement).disabled,
  ).toBe(true);
  expect(create).not.toHaveBeenCalled();
});

it('searches saved snapshots and cancels then confirms deletion', async () => {
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket], total: 1 });
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [snapshot], total: 1 });
  const remove = vi.spyOn(api, 'deleteDataset').mockResolvedValue(undefined);
  open(<DatasetsPage />);
  const download = await screen.findByRole('link', { name: 'Download ZIP' });
  expect(download.getAttribute('href')).toBe('/api/datasets/saved/download');
  fireEvent.change(screen.getByLabelText('Find snapshots'), { target: { value: 'missing' } });
  expect(screen.getByText('No matching snapshots')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Clear search' }));
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
  expect(remove).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Keep it' }));
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
  fireEvent.click(screen.getByRole('button', { name: 'Delete Week five' }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith(snapshot.id));
});

it('shows the saved snapshot download after a real ready response', async () => {
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket], total: 1 });
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [], total: 0 });
  const create = vi.spyOn(api, 'createDataset').mockResolvedValue(snapshot);
  open(<DatasetsPage />);
  await screen.findByLabelText('Source bucket');
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: snapshot.name } });
  fireEvent.click(screen.getByRole('button', { name: 'Build snapshot' }));
  const download = await screen.findByRole('link', { name: 'Download your snapshot →' });
  expect(download.getAttribute('href')).toBe('/api/datasets/saved/download');
  expect(create).toHaveBeenCalledWith(
    expect.objectContaining({ validation_fraction: 0.1, seed: 42 }),
  );
});
