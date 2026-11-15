// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { bucket, open } from '../test-utils';
import BucketsPage from './BucketsPage';

const empty = {
  ...bucket,
  id: 'empty',
  name: 'Empty notes',
  description: 'A handbook',
  stats: { documents: 0, parsed_documents: 0, chars: 0, est_tokens: 0 },
};
const notes = {
  ...bucket,
  id: 'notes',
  name: 'Class notes',
  description: 'A website',
  kind: 'chat',
  stats: { ...bucket.stats, documents: 3, parsed_documents: 1 },
};
beforeEach(() => {
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket, empty, notes], total: 3 });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('searches source notes, filters eligibility, switches view, and clears no matches', async () => {
  open(<BucketsPage />);
  await screen.findByRole('article', { name: bucket.name });
  fireEvent.change(screen.getByLabelText('Find collections'), { target: { value: 'handbook' } });
  expect(screen.getAllByRole('article')).toHaveLength(1);
  expect(screen.getByRole('article', { name: empty.name })).toBeTruthy();
  fireEvent.change(screen.getByLabelText('Filter collections'), { target: { value: 'ready' } });
  expect(screen.getByText('No matches')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }));
  expect(screen.getAllByRole('article')).toHaveLength(3);
  fireEvent.change(screen.getByLabelText('Filter collections'), { target: { value: 'ready' } });
  expect(screen.getAllByRole('article')).toHaveLength(1);
  expect(screen.getByRole('link', { name: 'Train on this →' }).getAttribute('href')).toBe(
    '/train?data=bucket-1',
  );
  fireEvent.click(screen.getByRole('button', { name: 'List' }));
  expect(screen.getAllByRole('row')).toHaveLength(2);
  expect(screen.getByRole('button', { name: 'List' }).getAttribute('aria-pressed')).toBe('true');
});

it('tags each collection as text or chat', async () => {
  open(<BucketsPage />);
  const card = await screen.findByRole('article', { name: notes.name });
  expect(within(card).getByText('chat')).toBeTruthy();
  expect(within(screen.getByRole('article', { name: bucket.name })).getByText('text')).toBeTruthy();
});

it('sorts by name and document count', async () => {
  open(<BucketsPage />);
  await screen.findByRole('article', { name: bucket.name });
  fireEvent.change(screen.getByLabelText('Sort collections'), { target: { value: 'name' } });
  expect(screen.getAllByRole('article').map((item) => item.getAttribute('aria-label'))).toEqual([
    notes.name,
    empty.name,
    bucket.name,
  ]);
  fireEvent.change(screen.getByLabelText('Sort collections'), { target: { value: 'documents' } });
  expect(screen.getAllByRole('article').map((item) => item.getAttribute('aria-label'))).toEqual([
    bucket.name,
    notes.name,
    empty.name,
  ]);
});

it('creates a collection with trimmed values and refreshes the overview', async () => {
  const create = vi.spyOn(api, 'createBucket').mockResolvedValue({ ...empty, name: 'New notes' });
  const { queryClient } = open(<BucketsPage />);
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: ' New notes ' } });
  fireEvent.change(screen.getByLabelText('Where it came from'), { target: { value: ' class ' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create collection' }));
  await waitFor(() => expect(create).toHaveBeenCalledWith('New notes', 'class'));
  await screen.findByText('Created New notes.');
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['overview'] });
  expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('');
});

it('shows deletion summary for empty and nonempty collections across card and list views without deleting on cancel', async () => {
  const emptySummary = {
    bucket_id: empty.id,
    bucket_name: empty.name,
    document_count: 0,
    raw_file_count: 0,
    text_file_count: 0,
    physical_file_count: 0,
    raw_bytes: 0,
    text_bytes: 0,
    total_bytes: 0,
    shared_file_count: 0,
    missing_file_count: 0,
    active_jobs: [],
    scope_token: 'tok-empty',
  };
  const corpusSummary = {
    bucket_id: bucket.id,
    bucket_name: bucket.name,
    document_count: 1500,
    raw_file_count: 1500,
    text_file_count: 1500,
    physical_file_count: 3000,
    raw_bytes: 1_048_576,
    text_bytes: 524_288,
    total_bytes: 1_572_864,
    shared_file_count: 1,
    missing_file_count: 0,
    active_jobs: [],
    scope_token: 'tok-corpus',
  };
  const summarySpy = vi
    .spyOn(api, 'bucketDeletionSummary')
    .mockImplementation(async (id) => (id === empty.id ? emptySummary : corpusSummary));

  let resolveDelete!: () => void;
  const pendingDelete = new Promise<void>((resolve) => {
    resolveDelete = resolve;
  });
  const removeSpy = vi.spyOn(api, 'deleteBucket').mockReturnValueOnce(pendingDelete);

  const { queryClient } = open(<BucketsPage />);
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');

  // Nonempty card offers Delete alongside Save snapshot ->
  const corpusCard = await screen.findByRole('article', { name: bucket.name });
  fireEvent.click(within(corpusCard).getByRole('button', { name: 'Delete' }));

  const dialog = await screen.findByRole('dialog');
  expect(summarySpy).toHaveBeenCalledWith(bucket.id);
  expect(within(dialog).getByText('1,500 documents')).toBeTruthy();
  expect(within(dialog).getByText('3,000 files')).toBeTruthy();
  expect(within(dialog).getByText('1.5 MB')).toBeTruthy();
  expect(
    within(dialog).getByText(/1 shared raw file still referenced by other collections/),
  ).toBeTruthy();
  expect(removeSpy).not.toHaveBeenCalled();

  fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  expect(removeSpy).not.toHaveBeenCalled();

  // Switch to List view and verify nonempty collection deletion works and prevents duplicate submits
  fireEvent.click(screen.getByRole('button', { name: 'List' }));
  const corpusRow = screen.getByRole('link', { name: bucket.name }).closest('tr')!;
  fireEvent.click(within(corpusRow).getByRole('button', { name: 'Delete' }));

  const listDialog = await screen.findByRole('dialog');
  const confirmBtn = within(listDialog).getByRole('button', {
    name: 'Delete collection',
  }) as HTMLButtonElement;
  await waitFor(() => expect(confirmBtn.disabled).toBe(false));
  fireEvent.click(confirmBtn);
  fireEvent.click(confirmBtn);

  await waitFor(() => expect(removeSpy).toHaveBeenCalledTimes(1));
  expect(removeSpy).toHaveBeenCalledWith(bucket.id, {
    confirm: true,
    scopeToken: 'tok-corpus',
  });
  expect(
    (within(listDialog).getByRole('button', { name: 'Deleting collection…' }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);

  resolveDelete();
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['buckets'] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['overview'] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['datasets'] });

  // Empty card also still supports deletion cleanly
  fireEvent.click(screen.getByRole('button', { name: 'Cards' }));
  removeSpy.mockResolvedValueOnce(undefined);
  const emptyCard = await screen.findByRole('article', { name: empty.name });
  fireEvent.click(within(emptyCard).getByRole('button', { name: 'Delete' }));
  const emptyDialog = await screen.findByRole('dialog');
  const emptyConfirm = (await within(emptyDialog).findByRole('button', {
    name: 'Delete collection',
  })) as HTMLButtonElement;
  await waitFor(() => expect(emptyConfirm.disabled).toBe(false));
  fireEvent.click(emptyConfirm);
  await waitFor(() =>
    expect(removeSpy).toHaveBeenCalledWith(empty.id, {
      confirm: true,
      scopeToken: 'tok-empty',
    }),
  );
});

it('recovers a failed list without losing the create form', async () => {
  vi.mocked(api.listBuckets).mockRejectedValueOnce(new Error('Connection lost.'));
  open(<BucketsPage />);
  await screen.findByText('Connection lost.');
  expect(screen.getByLabelText('Name')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  await screen.findByRole('article', { name: bucket.name });
});
