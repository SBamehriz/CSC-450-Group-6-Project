import type { components } from './types';

export type Bucket = components['schemas']['BucketOut'];
export type BucketPage = components['schemas']['Page_BucketOut_'];
export type Health = components['schemas']['Health'];
export type Document = components['schemas']['DocumentOut'];
export type DocumentPage = components['schemas']['Page_DocumentOut_'];
export type DocumentText = components['schemas']['DocumentText'];
export type UploadResult = components['schemas']['UploadResult'];

const API_PREFIX = '/api';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  // FormData sets its own content-type
  const isJson = init.body !== undefined && !(init.body instanceof FormData);
  const response = await fetch(`${API_PREFIX}${path}`, {
    ...init,
    headers: {
      ...(isJson ? { 'Content-Type': 'application/json' } : {}),
      ...init.headers,
    },
  });

  if (response.status === 204) {
    return undefined as T;
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const envelope = body?.error;
    throw new ApiError(
      response.status,
      envelope?.code ?? 'unknown',
      envelope?.message ?? `Request failed with status ${response.status}.`,
    );
  }

  return body as T;
}

export const api = {
  health: () => request<Health>('/health'),
  listBuckets: () => request<BucketPage>('/buckets'),
  getBucket: (id: string) => request<Bucket>(`/buckets/${id}`),
  createBucket: (name: string, description: string) =>
    request<Bucket>('/buckets', {
      method: 'POST',
      body: JSON.stringify({ name, description }),
    }),
  deleteBucket: (id: string) => request<void>(`/buckets/${id}`, { method: 'DELETE' }),

  uploadDocuments: (bucketId: string, files: File[], sourceNote: string) => {
    const form = new FormData();
    for (const file of files) form.append('files', file);
    form.append('source_note', sourceNote);
    return request<UploadResult>(`/buckets/${bucketId}/documents`, {
      method: 'POST',
      body: form,
    });
  },
  listDocuments: (params: { bucketId: string; status?: string; q?: string; flagged?: boolean }) => {
    const query = new URLSearchParams({ bucket_id: params.bucketId, limit: '200' });
    if (params.status) query.set('status', params.status);
    if (params.q) query.set('q', params.q);
    if (params.flagged) query.set('flagged', 'true');
    return request<DocumentPage>(`/documents?${query}`);
  },
  documentText: (id: string, offset = 0, length = 4000) =>
    request<DocumentText>(`/documents/${id}/text?offset=${offset}&length=${length}`),
  moveDocument: (id: string, bucketId: string) =>
    request<Document>(`/documents/${id}/move`, {
      method: 'POST',
      body: JSON.stringify({ bucket_id: bucketId }),
    }),
  rejectDocument: (id: string, reason: string) =>
    request<Document>(`/documents/${id}/reject`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: 'DELETE' }),
};
