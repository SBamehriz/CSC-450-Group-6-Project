// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type Dataset } from '../api/client';
import type { components } from '../api/types';
import { open } from '../test-utils';
import PreparedDataPage from './PreparedDataPage';

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
const prepared: components['schemas']['PreparedOut'] = {
  ready: true,
  message: '',
  vocab_size: 400,
  requested_vocab_size: 512,
  train_tokens: 1000,
  validation_tokens: 300,
  trained_on_documents: 3,
  eos_id: 0,
  sha256: 'a'.repeat(64),
  snapshot_sha256: 'b'.repeat(64),
};
const empty = { ...prepared, ready: false, sha256: '', train_tokens: 0, validation_tokens: 0 };

beforeEach(() => {
  vi.spyOn(api, 'getDataset').mockResolvedValue(snapshot);
  vi.spyOn(api, 'getPrepared').mockResolvedValue(empty);
  vi.spyOn(api, 'preparationTools').mockResolvedValue({ available: true, max_text_mb: 10 });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function showPage() {
  return open(
    <Routes>
      <Route path="/datasets/:datasetId/prepare" element={<PreparedDataPage />} />
    </Routes>,
    '/datasets/saved/prepare',
  );
}

it('prepares with the chosen vocabulary and shows real counts and a download', async () => {
  const build = vi.spyOn(api, 'prepareDataset').mockResolvedValue(prepared);
  showPage();
  await screen.findByLabelText('Vocabulary size');
  fireEvent.change(screen.getByLabelText('Vocabulary size'), { target: { value: '512' } });
  await waitFor(() =>
    expect(
      (screen.getByRole('button', { name: 'Prepare data' }) as HTMLButtonElement).disabled,
    ).toBe(false),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Prepare data' }));
  const link = await screen.findByRole('link', { name: 'Download prepared ZIP' });
  expect(link.getAttribute('href')).toBe('/api/datasets/saved/prepared/download');
  expect(build).toHaveBeenCalledWith('saved', 512);
  expect(screen.getByText('3 training files')).toBeTruthy();
});

it('blocks invalid settings and explains missing tokenizer tools', async () => {
  vi.spyOn(api, 'preparationTools').mockResolvedValue({ available: false, max_text_mb: 10 });
  const build = vi.spyOn(api, 'prepareDataset');
  showPage();
  await screen.findByText(/Tokenizer tools are not installed/);
  fireEvent.change(screen.getByLabelText('Vocabulary size'), { target: { value: '256' } });
  expect((screen.getByRole('button', { name: 'Prepare data' }) as HTMLButtonElement).disabled).toBe(
    true,
  );
  expect(build).not.toHaveBeenCalled();
});

it('shows tokens from the saved vocabulary', async () => {
  vi.spyOn(api, 'getPrepared').mockResolvedValue(prepared);
  const preview = vi
    .spyOn(api, 'previewTokens')
    .mockResolvedValue({ token_count: 2, ids: [5, 9], pieces: ['Hello', 'world'] });
  showPage();
  const input = await screen.findByLabelText('Sample text');
  expect((screen.getByLabelText('Vocabulary size') as HTMLInputElement).value).toBe('512');
  fireEvent.change(input, { target: { value: 'Hello world' } });
  fireEvent.click(screen.getByRole('button', { name: 'Show tokens' }));
  expect(await screen.findByText('2 tokens in this sample.')).toBeTruthy();
  expect(preview).toHaveBeenCalledWith('saved', 'Hello world');
});

it('shows preparation errors without claiming success', async () => {
  vi.spyOn(api, 'prepareDataset').mockRejectedValue(new Error('Snapshot checksum changed.'));
  showPage();
  await waitFor(() =>
    expect(
      (screen.getByRole('button', { name: 'Prepare data' }) as HTMLButtonElement).disabled,
    ).toBe(false),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Prepare data' }));
  expect(await screen.findByText('Snapshot checksum changed.')).toBeTruthy();
  expect(screen.queryByRole('link', { name: 'Download prepared ZIP' })).toBeNull();
});

it('requires confirmation to remove prepared files', async () => {
  vi.spyOn(api, 'getPrepared').mockResolvedValue(prepared);
  const remove = vi.spyOn(api, 'deletePrepared').mockResolvedValue(undefined);
  showPage();
  fireEvent.click(await screen.findByRole('button', { name: 'Remove prepared files' }));
  expect(remove).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Remove token files' }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith('saved'));
});
