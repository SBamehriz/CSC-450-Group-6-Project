import { afterEach, expect, it, vi } from 'vitest';
import { api } from './client';

afterEach(() => vi.unstubAllGlobals());

it('loads the next page before reporting library totals', async () => {
  const firstPage = Array.from({ length: 200 }, (_, index) => ({
    id: `bucket-${index}`,
  }));
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ items: firstPage, total: 201 })))
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ items: [{ id: 'last-bucket' }], total: 201 })),
    );
  vi.stubGlobal('fetch', fetch);
  const result = await api.listBuckets();
  expect(result.items).toHaveLength(201);
  expect(result.items[200].id).toBe('last-bucket');
  expect(result.total).toBe(201);
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    '/api/buckets?limit=200&offset=0',
    '/api/buckets?limit=200&offset=200',
  ]);
});

it('rejects a partial library when a later page fails', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ items: [{ id: 'one' }], total: 2 })))
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          error: { code: 'unavailable', message: 'Database unavailable.' },
        }),
        { status: 503 },
      ),
    );
  vi.stubGlobal('fetch', fetch);
  await expect(api.listBuckets()).rejects.toThrow('Database unavailable.');
});

it('uses the dashboard endpoints and loads document details', async () => {
  const fetch = vi
    .fn()
    .mockImplementation(async () => new Response(JSON.stringify({ items: [], total: 0 })));
  vi.stubGlobal('fetch', fetch);
  await api.overviewStats();
  await api.recentUploads();
  await api.getDocument('doc-1');
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    '/api/overview/stats',
    '/api/overview/recent-uploads?limit=8&offset=0',
    '/api/documents/doc-1',
  ]);
});

it('sends the page offset and encoded document filters', async () => {
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ items: [], total: 0 })));
  vi.stubGlobal('fetch', fetch);
  await api.listDocuments({ bucketId: 'bucket-1', offset: 50, q: 'notes & tests', flagged: true });
  const url = new URL(fetch.mock.calls[0][0], 'http://localhost');
  expect(url.searchParams.get('limit')).toBe('50');
  expect(url.searchParams.get('offset')).toBe('50');
  expect(url.searchParams.get('q')).toBe('notes & tests');
  expect(url.searchParams.get('flagged')).toBe('true');
});

it('fetches bucket deletion summary and sends explicit confirmation params on delete', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          bucket_id: 'bucket-1',
          bucket_name: 'Corpus',
          document_count: 1200,
          raw_file_count: 1200,
          text_file_count: 1200,
          physical_file_count: 2400,
          raw_bytes: 1024,
          text_bytes: 512,
          total_bytes: 1536,
          shared_file_count: 0,
          missing_file_count: 0,
          active_jobs: [],
          scope_token: 'tok&1',
        }),
      ),
    )
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal('fetch', fetch);

  const summary = await api.bucketDeletionSummary('bucket-1');
  expect(summary.document_count).toBe(1200);
  await api.deleteBucket('bucket-1', { confirm: true, scopeToken: summary.scope_token });

  expect(fetch.mock.calls[0][0]).toBe('/api/buckets/bucket-1/deletion-summary');
  const deleteUrl = new URL(fetch.mock.calls[1][0], 'http://localhost');
  expect(fetch.mock.calls[1][1]).toEqual(expect.objectContaining({ method: 'DELETE' }));
  expect(deleteUrl.pathname).toBe('/api/buckets/bucket-1');
  expect(deleteUrl.searchParams.get('confirm')).toBe('true');
  expect(deleteUrl.searchParams.get('scope_token')).toBe('tok&1');
});
