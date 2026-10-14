// @vitest-environment jsdom
import { act, cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type UploadResult } from '../api/client';
import { bucket, document, open, stats } from '../test-utils';
import OverviewPage from './OverviewPage';

beforeEach(() => {
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [bucket], total: 1 });
  vi.spyOn(api, 'overviewStats').mockResolvedValue(stats);
  vi.spyOn(api, 'recentUploads').mockResolvedValue({
    items: [{ ...document, bucket_name: bucket.name }],
    total: 1,
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('keeps totals unknown while loading and shows the latest document link', async () => {
  vi.mocked(api.overviewStats).mockReturnValue(new Promise(() => {}));
  open(<OverviewPage />);
  expect(screen.getByText('Loading library statistics…')).toBeTruthy();
  expect(screen.queryByText('Ready snapshots')).toBeNull();
  expect((await screen.findByRole('link', { name: 'source.docx' })).getAttribute('href')).toBe(
    '/data/bucket-1?open=doc-1',
  );
});

it('recovers statistics without hiding working upload activity', async () => {
  vi.mocked(api.overviewStats).mockRejectedValueOnce(new Error('Stats unavailable.'));
  open(<OverviewPage />);
  expect(await screen.findByText('Stats unavailable.')).toBeTruthy();
  expect(await screen.findByRole('link', { name: 'source.docx' })).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(await screen.findByText('Flagged for review')).toBeTruthy();
  expect(screen.queryByText('Stats unavailable.')).toBeNull();
});

it('offers a bucket action when the library is empty', async () => {
  vi.mocked(api.listBuckets).mockResolvedValue({ items: [], total: 0 });
  vi.mocked(api.recentUploads).mockResolvedValue({ items: [], total: 0 });
  vi.mocked(api.overviewStats).mockResolvedValue(
    Object.fromEntries(Object.keys(stats).map((key) => [key, 0])) as typeof stats,
  );
  open(<OverviewPage />);
  expect(await screen.findByRole('link', { name: 'Create a collection' })).toBeTruthy();
  expect(await screen.findByText('No uploads yet')).toBeTruthy();
  expect(screen.queryByLabelText('Files to upload')).toBeNull();
});

it('blocks more uploads while parsing, then refreshes data and reports each outcome', async () => {
  let finish!: (result: UploadResult) => void;
  const upload = vi.spyOn(api, 'uploadDocuments').mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const { queryClient } = open(<OverviewPage />);
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  const picker = await screen.findByLabelText('Files to upload');
  const file = new File(['text'], 'file.txt');
  fireEvent.change(picker, { target: { files: [file] } });
  await waitFor(() => expect(upload).toHaveBeenCalledWith(bucket.id, [file], ''));
  expect((screen.getByLabelText('Upload destination') as HTMLSelectElement).disabled).toBe(true);
  expect((screen.getByRole('button', { name: 'choose files' }) as HTMLButtonElement).disabled).toBe(
    true,
  );
  fireEvent.drop(picker.parentElement!, { dataTransfer: { files: [file] } });
  expect(upload).toHaveBeenCalledTimes(1);
  expect(screen.getByText(/Uploading and parsing 1 file/)).toBeTruthy();
  await act(async () =>
    finish({
      parsed: 1,
      failed: 1,
      outcomes: [
        { filename: 'file.txt', status: 'parsed' },
        { filename: 'bad.exe', status: 'failed', error: 'Unsupported format.' },
      ],
    }),
  );
  expect(await screen.findByText('1 parsed, 1 failed')).toBeTruthy();
  expect(screen.getByText(/Unsupported format/)).toBeTruthy();
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['overview'] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ['recentUploads'] });
});

it('clears the previous upload report when another request fails', async () => {
  vi.spyOn(api, 'uploadDocuments')
    .mockResolvedValueOnce({ parsed: 1, failed: 0, outcomes: [] })
    .mockRejectedValueOnce(new Error('Network lost.'));
  open(<OverviewPage />);
  const picker = await screen.findByLabelText('Files to upload');
  const file = new File(['text'], 'file.txt');
  fireEvent.change(picker, { target: { files: [file] } });
  expect(await screen.findByText('1 parsed, 0 failed')).toBeTruthy();
  fireEvent.change(picker, { target: { files: [file] } });
  expect(await screen.findByText(/Network lost/)).toBeTruthy();
  expect(screen.queryByText('1 parsed, 0 failed')).toBeNull();
});

it('rejects an oversized selection before sending it', async () => {
  const upload = vi.spyOn(api, 'uploadDocuments');
  open(<OverviewPage />);
  const picker = await screen.findByLabelText('Files to upload');
  const file = new File(['x'], 'big.txt');
  Object.defineProperty(file, 'size', { value: 50 * 1024 * 1024 + 1 });
  fireEvent.change(picker, { target: { files: [file] } });
  expect(screen.getByText('big.txt is over 50 MB.')).toBeTruthy();
  expect(upload).not.toHaveBeenCalled();
});
