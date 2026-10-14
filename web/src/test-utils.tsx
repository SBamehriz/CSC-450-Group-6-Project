import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter } from 'react-router-dom';

import type { Bucket, Document, OverviewStats } from './api/client';

export const bucket: Bucket = {
  id: 'bucket-1',
  name: 'Sources',
  description: '',
  created_at: '2026-10-02T00:00:00Z',
  stats: { documents: 51, parsed_documents: 50, chars: 10000, est_tokens: 2500 },
};
export const document: Document = {
  id: 'doc-1',
  bucket_id: bucket.id,
  filename: 'source.docx',
  source_format: 'docx',
  source_note: 'Class demo',
  status: 'parsed',
  error: null,
  content_hash: null,
  char_count: 4500,
  word_count: 900,
  quality: { flags: [], extraction: { engine: 'openxml', ocr_applied: false, table_count: 1 } },
  created_at: '2026-10-02T00:00:00Z',
};
export const stats: OverviewStats = {
  buckets: 1,
  documents: 51,
  parsed_documents: 50,
  failed_documents: 1,
  rejected_documents: 0,
  pending_documents: 0,
  flagged_documents: 3,
  chars: 10000,
  words: 2000,
  est_tokens: 2500,
  snapshots: 1,
  ready_snapshots: 1,
};
export function open(children: ReactNode, path = '/') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return {
    queryClient,
    ...render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[path]}>{children}</MemoryRouter>
      </QueryClientProvider>,
    ),
  };
}
