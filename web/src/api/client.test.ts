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
