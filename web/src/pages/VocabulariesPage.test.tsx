// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type Dataset, type Tokenizer } from '../api/client';
import { open } from '../test-utils';
import VocabulariesPage from './VocabulariesPage';

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

beforeEach(() => {
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [snapshot], total: 1 });
  vi.spyOn(api, 'listTokenizers').mockResolvedValue({ items: [], total: 0 });
  vi.spyOn(api, 'listJobs').mockResolvedValue({ items: [], total: 0 });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('shows loading, then the empty state', async () => {
  open(<VocabulariesPage />);
  expect(screen.getByText('Loading vocabularies…')).toBeTruthy();
  expect(await screen.findByText('No vocabularies yet')).toBeTruthy();
  expect(screen.queryByText('Try a vocabulary')).toBeNull();
});

it('makes a vocabulary from a snapshot with the chosen size', async () => {
  const create = vi.spyOn(api, 'createTokenizer').mockResolvedValue({
    ...vocabulary,
    status: 'pending',
    vocab_size: null,
  });
  open(<VocabulariesPage />);
  fireEvent.change(await screen.findByLabelText('Vocabulary name'), {
    target: { value: 'Notes pieces' },
  });
  fireEvent.change(screen.getByLabelText('Vocabulary size'), { target: { value: '512' } });
  fireEvent.click(screen.getByRole('button', { name: 'Make vocabulary' }));
  await waitFor(() =>
    expect(create).toHaveBeenCalledWith({
      name: 'Notes pieces',
      dataset_id: 'saved',
      vocab_size: 512,
    }),
  );
  expect(await screen.findByText('Vocabulary queued. Follow its progress below.')).toBeTruthy();
});

it('blocks sizes outside 257 to 65,535', async () => {
  const create = vi.spyOn(api, 'createTokenizer');
  open(<VocabulariesPage />);
  fireEvent.change(await screen.findByLabelText('Vocabulary name'), { target: { value: 'big' } });
  fireEvent.change(screen.getByLabelText('Vocabulary size'), { target: { value: '70000' } });
  expect(
    (screen.getByRole('button', { name: 'Make vocabulary' }) as HTMLButtonElement).disabled,
  ).toBe(true);
  expect(create).not.toHaveBeenCalled();
});

it('asks for a snapshot first when there is none', async () => {
  vi.spyOn(api, 'listDatasets').mockResolvedValue({ items: [], total: 0 });
  open(<VocabulariesPage />);
  expect(await screen.findByRole('link', { name: 'Datasets' })).toBeTruthy();
});

it('lists vocabularies and lets you try one on your own text', async () => {
  vi.spyOn(api, 'listTokenizers').mockResolvedValue({ items: [vocabulary], total: 1 });
  const preview = vi
    .spyOn(api, 'previewTokenizer')
    .mockResolvedValue({ token_count: 2, ids: [5, 9], pieces: ['Hello', 'Ġworld'] });
  open(<VocabulariesPage />);
  expect(await screen.findByText('Notes pieces', { selector: 'td' })).toBeTruthy();
  expect(screen.getByText('812')).toBeTruthy();
  fireEvent.change(screen.getByLabelText('Sample text'), { target: { value: 'Hello world' } });
  fireEvent.click(screen.getByRole('button', { name: 'Show word pieces' }));
  expect(await screen.findByText('2 word pieces in this sample.')).toBeTruthy();
  expect(preview).toHaveBeenCalledWith('pieces', 'Hello world');
});

it('asks before deleting and shows why a delete was refused', async () => {
  vi.spyOn(api, 'listTokenizers').mockResolvedValue({ items: [vocabulary], total: 1 });
  const remove = vi
    .spyOn(api, 'deleteTokenizer')
    .mockRejectedValue(new Error('A training set uses this vocabulary.'));
  open(<VocabulariesPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'Delete' }));
  expect(remove).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Delete Notes pieces' }));
  expect(await screen.findByText('A training set uses this vocabulary.')).toBeTruthy();
  expect(remove.mock.calls[0][0]).toBe('pieces');
});

it('shows load errors with a retry', async () => {
  vi.spyOn(api, 'listTokenizers').mockRejectedValue(new Error('Server unavailable.'));
  open(<VocabulariesPage />);
  expect(await screen.findByText('Server unavailable.')).toBeTruthy();
  expect(screen.getAllByRole('button', { name: 'Try again' }).length).toBeGreaterThan(0);
});
