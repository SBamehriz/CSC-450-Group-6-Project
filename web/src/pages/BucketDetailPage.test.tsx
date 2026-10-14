// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { bucket, document, open } from '../test-utils';
import BucketDetailPage from './BucketDetailPage';

function visit(path = '/data/bucket-1') {
  return open(
    <Routes>
      <Route path="/data/:bucketId" element={<BucketDetailPage />} />
    </Routes>,
    path,
  );
}
beforeEach(() => {
  vi.spyOn(api, 'getBucket').mockResolvedValue(bucket);
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket], total: 1 });
  vi.spyOn(api, 'listDocuments').mockResolvedValue({ items: [document], total: 51 });
  vi.spyOn(api, 'getDocument').mockResolvedValue(document);
  vi.spyOn(api, 'documentText').mockResolvedValue({
    text: 'Cleaned words.',
    offset: 0,
    length: 14,
    total: 14,
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('opens an activity link independently of the first document page', async () => {
  vi.mocked(api.listDocuments).mockResolvedValue({ items: [], total: 0 });
  visit('/data/bucket-1?open=doc-1');
  expect(await screen.findByText('Extraction engine')).toBeTruthy();
  expect(screen.getByText('openxml')).toBeTruthy();
  expect(screen.getByText('OCR used').nextElementSibling?.textContent).toBe('No');
  expect(await screen.findByText('Cleaned words.')).toBeTruthy();
  expect(api.getDocument).toHaveBeenCalledWith('doc-1');
  expect(
    screen.getByRole('link', { name: 'Create a snapshot from this bucket →' }).getAttribute('href'),
  ).toBe('/datasets?bucket=bucket-1');
});

it('shows failed document details without requesting text', async () => {
  vi.mocked(api.getDocument).mockResolvedValue({
    ...document,
    status: 'failed',
    error: 'Damaged DOCX.',
    quality: {},
  });
  visit('/data/bucket-1?open=doc-1');
  expect(await screen.findByText('Damaged DOCX.')).toBeTruthy();
  expect(screen.getByText('No extraction details stored for this file.')).toBeTruthy();
  expect(api.documentText).not.toHaveBeenCalled();
});

it('loads later documents and resets pagination when a filter changes', async () => {
  visit();
  const next = await screen.findByRole('button', { name: 'Next documents' });
  fireEvent.click(next);
  await waitFor(() =>
    expect(api.listDocuments).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 50 })),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Flagged' }));
  await waitFor(() =>
    expect(api.listDocuments).toHaveBeenLastCalledWith(
      expect.objectContaining({ offset: 0, flagged: true }),
    ),
  );
  fireEvent.change(screen.getByLabelText('Filter by filename'), { target: { value: 'notes' } });
  await waitFor(() =>
    expect(api.listDocuments).toHaveBeenLastCalledWith(
      expect.objectContaining({ offset: 0, q: 'notes' }),
    ),
  );
});

it('refreshes dashboard and activity after rejecting a document', async () => {
  vi.spyOn(api, 'rejectDocument').mockResolvedValue({ ...document, status: 'rejected' });
  const { queryClient } = visit('/data/bucket-1?open=doc-1');
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  fireEvent.click(await screen.findByText('Manage this file'));
  fireEvent.click(screen.getByRole('button', { name: 'Reject' }));
  await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ['overview'] }));
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['recentUploads'] });
});

it('sends a stale activity link to the document current bucket', async () => {
  vi.mocked(api.getDocument).mockResolvedValue({ ...document, bucket_id: 'moved-bucket' });
  visit('/data/bucket-1?open=doc-1');
  expect(
    (await screen.findByRole('link', { name: 'Open its current bucket' })).getAttribute('href'),
  ).toBe('/data/moved-bucket?open=doc-1');
  expect(screen.queryByText('Extraction engine')).toBeNull();
});

it('returns to the first document page after an upload', async () => {
  vi.spyOn(api, 'uploadDocuments').mockResolvedValue({ parsed: 1, failed: 0, outcomes: [] });
  visit();
  fireEvent.click(await screen.findByRole('button', { name: 'Next documents' }));
  await waitFor(() =>
    expect(api.listDocuments).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 50 })),
  );
  fireEvent.change(screen.getByLabelText('Files to upload'), {
    target: { files: [new File(['new'], 'new.txt')] },
  });
  expect(await screen.findByText('1 parsed, 0 failed')).toBeTruthy();
  await waitFor(() =>
    expect(api.listDocuments).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 0 })),
  );
});

it('records a typed rejection reason', async () => {
  const reject = vi
    .spyOn(api, 'rejectDocument')
    .mockResolvedValue({ ...document, status: 'rejected' });
  visit('/data/bucket-1?open=doc-1');
  fireEvent.click(await screen.findByText('Manage this file'));
  fireEvent.change(screen.getByLabelText('Reason to reject (optional)'), {
    target: { value: ' Wrong source ' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Reject' }));
  await waitFor(() => expect(reject).toHaveBeenCalledWith(document.id, 'Wrong source'));
});

it('waits for an explicit move click and refreshes after moving', async () => {
  const other = { ...bucket, id: 'other', name: 'Other source' };
  vi.mocked(api.listBuckets).mockResolvedValue({ items: [bucket, other], total: 2 });
  const move = vi.spyOn(api, 'moveDocument').mockResolvedValue({ ...document, bucket_id: 'other' });
  const { queryClient } = visit('/data/bucket-1?open=doc-1');
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  fireEvent.click(await screen.findByText('Manage this file'));
  await screen.findByRole('option', { name: 'Other source' });
  fireEvent.change(screen.getByLabelText('Move to another collection'), {
    target: { value: 'other' },
  });
  expect(move).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Move file' }));
  await waitFor(() => expect(move).toHaveBeenCalledWith(document.id, 'other'));
  await waitFor(() =>
    expect(screen.queryByRole('region', { name: 'Review source.docx' })).toBeNull(),
  );
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['recentUploads'] });
});

it('confirms document deletion and does not send a request when cancelled', async () => {
  const remove = vi.spyOn(api, 'deleteDocument').mockResolvedValue(undefined);
  visit('/data/bucket-1?open=doc-1');
  fireEvent.click(await screen.findByText('Manage this file'));
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
  expect(remove).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Keep it' }));
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
  fireEvent.click(screen.getByRole('button', { name: 'Delete this file' }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith(document.id));
});

it('opens linked filename and quality filters without losing the deep link', async () => {
  visit('/data/bucket-1?filter=flagged&q=source&open=doc-1');
  await screen.findByText('Extraction engine');
  expect(api.listDocuments).toHaveBeenCalledWith(
    expect.objectContaining({ flagged: true, q: 'source', offset: 0 }),
  );
  expect(screen.getByRole('region', { name: 'Review source.docx' })).toBeTruthy();
});

it('keeps counted zeroes visible and explains a quality flag', async () => {
  const short = { ...document, char_count: 0, word_count: 0, quality: { flags: ['too_short'] } };
  vi.mocked(api.listDocuments).mockResolvedValue({ items: [short], total: 1 });
  vi.mocked(api.getDocument).mockResolvedValue(short);
  visit('/data/bucket-1?open=doc-1');
  expect(await screen.findByText('There may be too little text here to be useful.')).toBeTruthy();
  expect(screen.getAllByRole('cell', { name: '0' })).toHaveLength(2);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Explain what I’m seeing' }));
  expect(screen.getByText(/A snapshot includes every parsed file/)).toBeTruthy();
});
