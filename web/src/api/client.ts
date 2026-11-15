import type { components } from './types';

export type Bucket = components['schemas']['BucketOut'];
export type BucketActiveJob = components['schemas']['BucketActiveJob'];
export type BucketDeletionSummary = components['schemas']['BucketDeletionSummary'];
export type BucketPage = components['schemas']['Page_BucketOut_'];
export type Health = components['schemas']['Health'];
export type Document = components['schemas']['DocumentOut'];
export type DocumentPage = components['schemas']['Page_DocumentOut_'];
export type DocumentText = components['schemas']['DocumentText'];
export type UploadResult = components['schemas']['UploadResult'];
export type Dataset = components['schemas']['DatasetOut'];
export type OverviewStats = components['schemas']['OverviewStats'];
export type RecentUpload = components['schemas']['RecentUpload'];
export type GpuCheck = components['schemas']['GpuCheck'];
export type Model = components['schemas']['ModelOut'];
export type Run = components['schemas']['RunOut'];
export type RunPage = components['schemas']['Page_RunOut_'];
export type MetricPoint = components['schemas']['MetricPointOut'];
export type RunMetrics = components['schemas']['RunMetricsOut'];
export type Checkpoint = components['schemas']['CheckpointOut'];
export type Job = components['schemas']['JobOut'];
export type HfInspectResponse = components['schemas']['HfInspectResponse'];
export type HfConfigInfo = components['schemas']['HfConfigInfo'];
export type DatasetImportCreate = components['schemas']['DatasetImportCreate'];
export type PromptSet = components['schemas']['PromptSetOut'];
export type PromptSetCreate = components['schemas']['PromptSetCreate'];
export type PromptSetUpdate = components['schemas']['PromptSetUpdate'];
export type Evaluation = components['schemas']['EvaluationOut'];
export type EvaluationCreate = components['schemas']['EvaluationCreate'];
export type BpbEvaluation = components['schemas']['BpbEvaluationOut'];
export type BpbEvaluationCreate = components['schemas']['BpbEvaluationCreate'];
export type EvalResult = components['schemas']['EvalResultOut'];
export type SpeedTestCreate = components['schemas']['SpeedTestCreate'];
export type SpeedTest = components['schemas']['SpeedTestOut'];
export type HubImportRequest = components['schemas']['HubImportRequest'];
export type BenchmarkPlan = components['schemas']['BenchmarkPlanOut'];
export type BenchmarkCreate = components['schemas']['BenchmarkCreate'];
export type BenchmarkResult = components['schemas']['BenchmarkResult'];
export type PromptCompareCreate = components['schemas']['PromptCompareCreate'];
export type PromptRun = components['schemas']['PromptCompareResult'];
export type CompareModel = components['schemas']['CompareModel'];
export type Comparison = components['schemas']['CompareOut'];

export type Plan = components['schemas']['PlanOut'];
export type PlanRequest = components['schemas']['PlanRequest'];
export type TrainCreate = components['schemas']['TrainCreate'];
export type ChatMessage = components['schemas']['ChatMessage'];
export type ChatRequest = Pick<components['schemas']['ChatRequest'], 'model_id' | 'messages'> &
  Partial<components['schemas']['ChatRequest']>;
export type ChatEvent =
  | { text: string }
  | { done: true; tokens: number; seconds: number; stopped: 'end' | 'length' }
  | { error: string };

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
  bucketDeletionSummary: (id: string) =>
    request<BucketDeletionSummary>(`/buckets/${id}/deletion-summary`),
  deleteBucket: (id: string, options: { confirm: true; scopeToken: string }) => {
    const query = new URLSearchParams({
      confirm: String(options.confirm),
      scope_token: options.scopeToken,
    });
    return request<void>(`/buckets/${id}?${query.toString()}`, { method: 'DELETE' });
  },

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
  checkGpu: () => request<GpuCheck>('/gpu'),

  // Models
  listAllModels: () => allPages<Model>('/models'),
  deleteModel: (id: string) => request<void>(`/models/${id}`, { method: 'DELETE' }),

  // Runs and Training History
  listRuns: (params?: { status?: string; offset?: number; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.status) query.set('status', params.status);
    if (params?.offset !== undefined) query.set('offset', String(params.offset));
    if (params?.limit !== undefined) query.set('limit', String(params.limit));
    const qs = query.toString();
    return request<RunPage>(`/runs${qs ? `?${qs}` : ''}`);
  },
  getRun: (id: string) => request<Run>(`/runs/${id}`),
  deleteRun: (id: string) => request<void>(`/runs/${id}`, { method: 'DELETE' }),
  changeRun: (id: string, action: 'start' | 'stop' | 'resume') =>
    request<Run>(`/runs/${id}/${action}`, { method: 'POST' }),
  setStopAfter: (id: string, hours: number | null) =>
    request<Run>(`/runs/${id}/stop-after`, { method: 'PUT', body: JSON.stringify({ hours }) }),
  listCheckpoints: (id: string) => request<Checkpoint[]>(`/runs/${id}/checkpoints`),
  checkpointDownloadUrl: (id: string) => `${API_PREFIX}/checkpoints/${id}/download`,
  getRunMetrics: (id: string, limit?: number) => {
    const qs = limit ? `?limit=${limit}` : '';
    return request<RunMetrics>(`/runs/${id}/metrics${qs}`);
  },
  getJob: (id: string) => request<Job>(`/jobs/${id}`),

  // measuring this computer
  startSpeedTest: (payload: SpeedTestCreate) =>
    request<Job>('/speed-tests', { method: 'POST', body: JSON.stringify(payload) }),
  listSpeedTests: () => request<SpeedTest[]>('/speed-tests'),

  // models from the hub, and save points out to other tools
  importModel: (payload: HubImportRequest) =>
    request<Job>('/models/import', { method: 'POST', body: JSON.stringify(payload) }),
  exportCheckpoint: (id: string) => request<Job>(`/checkpoints/${id}/export`, { method: 'POST' }),
  checkpointExportUrl: (id: string) => `${API_PREFIX}/checkpoints/${id}/export`,

  // comparing models
  benchmarkPlan: (modelId: string) =>
    request<BenchmarkPlan>(`/benchmarks/plan?model_id=${encodeURIComponent(modelId)}`),
  startBenchmark: (payload: BenchmarkCreate) =>
    request<Job>('/benchmarks', { method: 'POST', body: JSON.stringify(payload) }),
  comparePrompts: (payload: PromptCompareCreate) =>
    request<Job>('/prompt-comparisons', { method: 'POST', body: JSON.stringify(payload) }),
  compare: (modelIds: string[]) =>
    request<Comparison>(
      `/compare?${modelIds.map((id) => `model_ids=${encodeURIComponent(id)}`).join('&')}`,
    ),
  listJobs: () => request<{ items: Job[]; total: number }>('/jobs?limit=20&offset=0'),
  changeJob: (id: string, action: 'cancel' | 'retry') =>
    request<Job>(`/jobs/${id}/${action}`, { method: 'POST' }),
  inspectHfDataset: (dataset: string) =>
    request<HfInspectResponse>('/buckets/huggingface/inspect', {
      method: 'POST',
      body: JSON.stringify({ dataset }),
    }),
  importDataset: (bucketId: string, payload: DatasetImportCreate) =>
    request<Job>(`/buckets/${bucketId}/import`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Prompt Sets & Evaluation
  listPromptSets: () => request<PromptSet[]>('/prompt-sets'),
  createPromptSet: (payload: PromptSetCreate) =>
    request<PromptSet>('/prompt-sets', { method: 'POST', body: JSON.stringify(payload) }),
  updatePromptSet: (id: string, payload: PromptSetUpdate) =>
    request<PromptSet>(`/prompt-sets/${id}`, { method: 'PUT', body: JSON.stringify(payload) }),
  deletePromptSet: (id: string) => request<void>(`/prompt-sets/${id}`, { method: 'DELETE' }),

  listEvaluations: () => request<Evaluation[]>('/evaluations'),
  createEvaluation: (payload: EvaluationCreate) =>
    request<Evaluation>('/evaluations', { method: 'POST', body: JSON.stringify(payload) }),
  deleteEvaluation: (id: string) => request<void>(`/evaluations/${id}`, { method: 'DELETE' }),

  // Bits Per Byte (BPB) Evaluations
  listBpbEvaluations: () => request<BpbEvaluation[]>('/bpb-evaluations'),
  createBpbEvaluation: (payload: BpbEvaluationCreate) =>
    request<BpbEvaluation>('/bpb-evaluations', { method: 'POST', body: JSON.stringify(payload) }),
  deleteBpbEvaluation: (id: string) =>
    request<void>(`/bpb-evaluations/${id}`, { method: 'DELETE' }),
  computeRunBpb: (runId: string) =>
    request<BpbEvaluation>(`/runs/${runId}/bpb`, { method: 'POST' }),

  // Saved Evaluation Results
  listEvalResults: (params?: { result_type?: string; model_id?: string }) => {
    const query = new URLSearchParams();
    if (params?.result_type) query.set('result_type', params.result_type);
    if (params?.model_id) query.set('model_id', params.model_id);
    const qs = query.toString();
    return request<EvalResult[]>(`/eval-results${qs ? `?${qs}` : ''}`);
  },
  deleteEvalResult: (id: string) => request<void>(`/eval-results/${id}`, { method: 'DELETE' }),
  plan: (payload: PlanRequest) =>
    request<Plan>('/plan', { method: 'POST', body: JSON.stringify(payload) }),
  // one call makes the model, gets the text ready and starts training
  train: (payload: TrainCreate) =>
    request<Run>('/train', { method: 'POST', body: JSON.stringify(payload) }),
};

// streams an answer, one event at a time; stopping it just returns
export async function chatStream(
  payload: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) return;
    throw error;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(
      response.status,
      body?.error?.code ?? 'unknown',
      body?.error?.message ?? `Request failed with status ${response.status}.`,
    );
  }
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const flush = (block: string) => {
    for (const line of block.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data) onEvent(JSON.parse(data) as ChatEvent);
    }
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let end = buffer.indexOf('\n\n');
      while (end !== -1) {
        flush(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        end = buffer.indexOf('\n\n');
      }
    }
    flush(buffer + decoder.decode());
  } catch (error) {
    if (signal?.aborted) return;
    throw error;
  }
}
