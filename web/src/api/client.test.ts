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

it('creates a dataset snapshot with its split settings', async () => {
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: 'dataset-1' })));
  vi.stubGlobal('fetch', fetch);
  const payload = {
    name: 'weekly',
    description: 'Class demo',
    bucket_id: 'bucket-1',
    validation_fraction: 0.1,
    seed: 42,
  };
  await api.createDataset(payload);
  expect(fetch).toHaveBeenCalledWith('/api/datasets', {
    method: 'POST',
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json' },
  });
  expect(api.datasetDownloadUrl('dataset-1')).toBe('/api/datasets/dataset-1/download');
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
