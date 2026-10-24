import type { components } from './types';

export type Bucket = components['schemas']['BucketOut'];
export type BucketPage = components['schemas']['Page_BucketOut_'];
export type Health = components['schemas']['Health'];
export type Document = components['schemas']['DocumentOut'];
export type DocumentPage = components['schemas']['Page_DocumentOut_'];
export type DocumentText = components['schemas']['DocumentText'];
export type UploadResult = components['schemas']['UploadResult'];
export type Dataset = components['schemas']['DatasetOut'];
export type DatasetCreate = components['schemas']['DatasetCreate'];
export type OverviewStats = components['schemas']['OverviewStats'];
export type RecentUpload = components['schemas']['RecentUpload'];
export type Preset = components['schemas']['PresetOut'];
export type GpuCheck = components['schemas']['GpuCheck'];
export type Model = components['schemas']['ModelOut'];
export type ModelCreate = components['schemas']['ModelCreate'];
export type ModelPreviewRequest = components['schemas']['ModelPreviewRequest'];
export type ModelPreviewOut = components['schemas']['ModelPreviewOut'];
export type ModelPage = components['schemas']['Page_ModelOut_'];
export type Vocabulary = components['schemas']['VocabularyOut'];
export type Run = components['schemas']['RunOut'];
export type RunCreate = components['schemas']['RunCreate'];
export type RunPage = components['schemas']['Page_RunOut_'];
export type MetricPoint = components['schemas']['MetricPointOut'];
export type RunMetrics = components['schemas']['RunMetricsOut'];

export const DOCUMENT_PAGE = 50;
export const RUN_PAGE = 20;

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

async function allPages<T>(path: string): Promise<Omit<BucketPage, 'items'> & { items: T[] }> {
  const items: T[] = [];
  let page: Omit<BucketPage, 'items'> & { items: T[] };

  do {
    page = await request<typeof page>(
      `${path}${path.includes('?') ? '&' : '?'}limit=200&offset=${items.length}`,
    );
    items.push(...page.items);
  } while (items.length < page.total && page.items.length > 0);

  return { items, total: items.length };
}

export const api = {
  health: () => request<Health>('/health'),
  overviewStats: () => request<OverviewStats>('/overview/stats'),
  recentUploads: () =>
    request<{ items: RecentUpload[]; total: number }>('/overview/recent-uploads?limit=8&offset=0'),
  listBuckets: () => allPages<Bucket>('/buckets'),
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
  listDocuments: (params: {
    bucketId: string;
    status?: string;
    q?: string;
    flagged?: boolean;
    offset?: number;
  }) => {
    const query = new URLSearchParams({
      bucket_id: params.bucketId,
      limit: String(DOCUMENT_PAGE),
      offset: String(params.offset ?? 0),
    });
    if (params.status) query.set('status', params.status);
    if (params.q) query.set('q', params.q);
    if (params.flagged) query.set('flagged', 'true');
    return request<DocumentPage>(`/documents?${query}`);
  },
  getDocument: (id: string) => request<Document>(`/documents/${id}`),
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
  listDatasets: () => allPages<Dataset>('/datasets'),
  getDataset: (id: string) => request<Dataset>(`/datasets/${id}`),
  verifyDataset: (id: string) =>
    request<components['schemas']['SnapshotCheck']>(`/datasets/${id}/verify`),
  createDataset: (payload: DatasetCreate) =>
    request<Dataset>('/datasets', { method: 'POST', body: JSON.stringify(payload) }),
  deleteDataset: (id: string) => request<void>(`/datasets/${id}`, { method: 'DELETE' }),
  datasetDownloadUrl: (id: string) => `${API_PREFIX}/datasets/${id}/download`,
  preparationTools: () => request<components['schemas']['PrepareTools']>('/preparation/tools'),
  getPrepared: (id: string) =>
    request<components['schemas']['PreparedOut']>(`/datasets/${id}/prepared`),
  prepareDataset: (id: string, vocabSize: number) =>
    request<components['schemas']['PreparedOut']>(`/datasets/${id}/prepare`, {
      method: 'POST',
      body: JSON.stringify({ vocab_size: vocabSize }),
    }),
  deletePrepared: (id: string) => request<void>(`/datasets/${id}/prepared`, { method: 'DELETE' }),
  previewTokens: (id: string, text: string) =>
    request<components['schemas']['TokenPreview']>(`/datasets/${id}/prepared/preview`, {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),
  preparedDownloadUrl: (id: string) => `${API_PREFIX}/datasets/${id}/prepared/download`,
  listPresets: () => request<Preset[]>('/presets'),
  checkGpu: () => request<GpuCheck>('/gpu'),

  // Models (SP8.9)
  listModels: (params?: { offset?: number; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.offset !== undefined) query.set('offset', String(params.offset));
    if (params?.limit !== undefined) query.set('limit', String(params.limit));
    const qs = query.toString();
    return request<ModelPage>(`/models${qs ? `?${qs}` : ''}`);
  },
  listAllModels: () => allPages<Model>('/models'),
  getModel: (id: string) => request<Model>(`/models/${id}`),
  createModel: (payload: ModelCreate) =>
    request<Model>('/models', { method: 'POST', body: JSON.stringify(payload) }),
  deleteModel: (id: string) => request<void>(`/models/${id}`, { method: 'DELETE' }),
  previewModel: (payload: ModelPreviewRequest) =>
    request<ModelPreviewOut>('/models/preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  listVocabularies: () => request<Vocabulary[]>('/models/vocabularies'),
  getModelPresets: () =>
    request<{
      presets: Record<
        string,
        {
          n_layer: number;
          d_model: number;
          n_head: number;
          ctx_len: number;
          vocab_size: number;
          dropout: number;
          param_count: number;
          estimated_memory_bytes: number;
        }
      >;
      limits: {
        max_n_layer: number;
        max_d_model: number;
        allowed_ctx: number[];
        max_dropout: number;
      };
    }>('/models/presets'),

  // Runs and Training History (SP8.18 & SP8.12)
  listRuns: (params?: { status?: string; offset?: number; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.status) query.set('status', params.status);
    if (params?.offset !== undefined) query.set('offset', String(params.offset));
    if (params?.limit !== undefined) query.set('limit', String(params.limit));
    const qs = query.toString();
    return request<RunPage>(`/runs${qs ? `?${qs}` : ''}`);
  },
  getRun: (id: string) => request<Run>(`/runs/${id}`),
  createRun: (payload: RunCreate) =>
    request<Run>('/runs', { method: 'POST', body: JSON.stringify(payload) }),
  deleteRun: (id: string) => request<void>(`/runs/${id}`, { method: 'DELETE' }),
  getRunMetrics: (id: string, limit?: number) => {
    const qs = limit ? `?limit=${limit}` : '';
    return request<RunMetrics>(`/runs/${id}/metrics${qs}`);
  },
};
