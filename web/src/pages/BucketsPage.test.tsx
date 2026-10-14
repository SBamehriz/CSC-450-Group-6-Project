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
  expect(screen.getByRole('link', { name: 'Save snapshot →' }).getAttribute('href')).toBe(
    '/datasets?bucket=bucket-1',
  );
  fireEvent.click(screen.getByRole('button', { name: 'List' }));
  expect(screen.getAllByRole('row')).toHaveLength(2);
  expect(screen.getByRole('button', { name: 'List' }).getAttribute('aria-pressed')).toBe('true');
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

it('deletes only an empty collection after confirmation and lets you cancel', async () => {
  const remove = vi.spyOn(api, 'deleteBucket').mockResolvedValue(undefined);
  open(<BucketsPage />);
  const card = await screen.findByRole('article', { name: empty.name });
  expect(
    within(screen.getByRole('article', { name: bucket.name })).queryByRole('button', {
      name: 'Delete',
    }),
  ).toBeNull();
  fireEvent.click(within(card).getByRole('button', { name: 'Delete' }));
  expect(remove).not.toHaveBeenCalled();
  fireEvent.click(within(card).getByRole('button', { name: 'Keep it' }));
  expect(within(card).queryByRole('button', { name: 'Delete Empty notes' })).toBeNull();
  fireEvent.click(within(card).getByRole('button', { name: 'Delete' }));
  fireEvent.click(within(card).getByRole('button', { name: 'Delete Empty notes' }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith(empty.id));
});

it('recovers a failed list without losing the create form', async () => {
  vi.mocked(api.listBuckets).mockRejectedValueOnce(new Error('Connection lost.'));
  open(<BucketsPage />);
  await screen.findByText('Connection lost.');
  expect(screen.getByLabelText('Name')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  await screen.findByRole('article', { name: bucket.name });
});
