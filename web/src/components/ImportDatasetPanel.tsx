import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { api, type HfInspectResponse, type Job } from '../api/client';
import { tones } from './JobsPanel';
import { Card, CardHead, Chip, Notice } from './ui';

interface ImportDatasetPanelProps {
  bucketId: string;
  onImportQueued?: (jobId: string) => void;
}

function progress(job: Job): {
  completed: number;
  total: number;
  message: string;
  processed?: number;
  imported?: number;
  skipped?: number;
  failed?: number;
} {
  const value = job.progress as {
    completed?: number;
    total?: number;
    message?: string;
    processed?: number;
    imported?: number;
    skipped?: number;
    failed?: number;
  };
  return {
    completed: value.completed ?? 0,
    total: value.total ?? 0,
    message: value.message ?? '',
    processed: value.processed,
    imported: value.imported,
    skipped: value.skipped,
    failed: value.failed,
  };
}

export default function ImportDatasetPanel({ bucketId, onImportQueued }: ImportDatasetPanelProps) {
  const queryClient = useQueryClient();
  const [datasetInput, setDatasetInput] = useState('');
  const [inspected, setInspected] = useState<HfInspectResponse | null>(null);
  const [selectedConfig, setSelectedConfig] = useState('');
  const [selectedSplit, setSelectedSplit] = useState('');
  const [selectedTextColumn, setSelectedTextColumn] = useState('');
  const [importLimit, setImportLimit] = useState(500);
  const [trackedJobId, setTrackedJobId] = useState<string | null>(null);

  // Poll background jobs to track active or recent import jobs for this collection
  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: api.listJobs,
    refetchInterval: 3000,
  });

  const allJobs = jobsQuery.data?.items ?? [];
  // Find latest import job for this collection
  const latestImportJob = allJobs.find(
    (j) =>
      j.job_type === 'dataset_import' &&
      (j.id === trackedJobId ||
        (typeof j.payload.target_id === 'string' && j.payload.target_id === bucketId)),
  );

  // Auto-track the latest active job if not already tracking
  useEffect(() => {
    if (
      !trackedJobId &&
      latestImportJob &&
      ['pending', 'running'].includes(latestImportJob.status)
    ) {
      setTrackedJobId(latestImportJob.id);
    }
  }, [trackedJobId, latestImportJob]);

  // When job status changes or finishes, invalidate collection documents and stats
  useEffect(() => {
    if (latestImportJob?.status === 'completed' || latestImportJob?.status === 'running') {
      void queryClient.invalidateQueries({ queryKey: ['bucket', bucketId] });
      void queryClient.invalidateQueries({ queryKey: ['buckets'] });
      void queryClient.invalidateQueries({ queryKey: ['documents', bucketId] });
      void queryClient.invalidateQueries({ queryKey: ['overview'] });
    }
  }, [latestImportJob?.status, latestImportJob?.progress, bucketId, queryClient]);

  // Inspection mutation
  const inspectMutation = useMutation({
    mutationFn: (name: string) => api.inspectHfDataset(name),
    onSuccess: (data) => {
      setInspected(data);
      if (data.configs.length > 0) {
        const firstConfig = data.configs[0];
        setSelectedConfig(firstConfig.name);
        if (firstConfig.splits.length > 0) {
          setSelectedSplit(firstConfig.splits[0].name);
        }
        if (firstConfig.text_columns.length > 0) {
          setSelectedTextColumn(firstConfig.text_columns[0]);
        } else if (firstConfig.columns.length > 0) {
          setSelectedTextColumn(firstConfig.columns[0]);
        }
      }
    },
  });

  // Start import mutation
  const importMutation = useMutation({
    mutationFn: (params: {
      dataset: string;
      config: string;
      split: string;
      text_column: string;
      limit: number;
    }) => api.importDataset(bucketId, params),
    onSuccess: (job) => {
      setTrackedJobId(job.id);
      onImportQueued?.(job.id);
      void queryClient.invalidateQueries({ queryKey: ['jobs'] });
      void queryClient.invalidateQueries({ queryKey: ['bucket', bucketId] });
    },
  });

  // Job cancel/retry mutation
  const changeJobMutation = useMutation({
    mutationFn: ({ id, action }: { id: string; action: 'cancel' | 'retry' }) =>
      api.changeJob(id, action),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['jobs'] });
      void queryClient.invalidateQueries({ queryKey: ['bucket', bucketId] });
      void queryClient.invalidateQueries({ queryKey: ['documents', bucketId] });
    },
  });

  const handleInspect = (event: React.FormEvent) => {
    event.preventDefault();
    if (!datasetInput.trim()) return;
    setInspected(null);
    inspectMutation.mutate(datasetInput.trim());
  };

  const currentConfigObj = inspected?.configs.find((c) => c.name === selectedConfig);

  const handleConfigChange = (cfgName: string) => {
    setSelectedConfig(cfgName);
    const cfg = inspected?.configs.find((c) => c.name === cfgName);
    if (cfg) {
      if (cfg.splits.length > 0) setSelectedSplit(cfg.splits[0].name);
      if (cfg.text_columns.length > 0) {
        setSelectedTextColumn(cfg.text_columns[0]);
      } else if (cfg.columns.length > 0) {
        setSelectedTextColumn(cfg.columns[0]);
      }
    }
  };

  const handleStartImport = (event: React.FormEvent) => {
    event.preventDefault();
    if (!inspected || !selectedConfig || !selectedSplit || !selectedTextColumn) return;
    importMutation.mutate({
      dataset: inspected.canonical_id,
      config: selectedConfig,
      split: selectedSplit,
      text_column: selectedTextColumn,
      limit: importLimit,
    });
  };

  const isImportRunning =
    latestImportJob && ['pending', 'running'].includes(latestImportJob.status);

  return (
    <Card>
      <CardHead title="Import from Hugging Face" />
      <div className="card-body stack">
        <p className="muted" style={{ margin: 0 }}>
          Stream public dataset records directly into this collection in bounded batches. The
          dataset will be pinned to an immutable commit revision.
        </p>

        {/* Dataset identifier lookup */}
        <form onSubmit={handleInspect} className="stack">
          <label className="field">
            <span className="label">Hugging Face dataset name or URL</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                className="input"
                style={{ flex: 1 }}
                placeholder="e.g. imdb, Salesforce/wikitext, or https://huggingface.co/datasets/imdb"
                value={datasetInput}
                disabled={inspectMutation.isPending || isImportRunning}
                onChange={(event) => setDatasetInput(event.target.value)}
              />
              <button
                type="submit"
                className="btn btn-primary"
                disabled={inspectMutation.isPending || !datasetInput.trim() || isImportRunning}
              >
                {inspectMutation.isPending ? 'Inspecting…' : 'Inspect dataset'}
              </button>
            </div>
          </label>
        </form>

        {inspectMutation.error ? <Notice tone="bad">{inspectMutation.error.message}</Notice> : null}

        {/* Inspected schema & import configuration */}
        {inspected ? (
          <form onSubmit={handleStartImport} className="stack">
            <div
              style={{
                background: 'var(--color-surface-soft, rgba(128, 0, 32, 0.04))',
                padding: '12px 16px',
                borderRadius: 'var(--radius, 6px)',
                border: '1px solid var(--color-border, #e5e5e5)',
              }}
            >
              <div
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
              >
                <div>
                  <strong>{inspected.canonical_id}</strong>{' '}
                  <span className="muted" title={inspected.sha}>
                    @ {inspected.sha.slice(0, 7)}
                  </span>
                </div>
                <Chip tone="ok">Pinned revision</Chip>
              </div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                gap: 12,
              }}
            >
              <label className="field">
                <span className="label">Configuration</span>
                <select
                  className="input"
                  value={selectedConfig}
                  disabled={importMutation.isPending || isImportRunning}
                  onChange={(event) => handleConfigChange(event.target.value)}
                >
                  {inspected.configs.map((cfg) => (
                    <option key={cfg.name} value={cfg.name}>
                      {cfg.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span className="label">Split</span>
                <select
                  className="input"
                  value={selectedSplit}
                  disabled={importMutation.isPending || isImportRunning}
                  onChange={(event) => setSelectedSplit(event.target.value)}
                >
                  {currentConfigObj?.splits.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name}
                      {s.num_examples != null ? ` (${s.num_examples} rows)` : ''}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span className="label">Text column</span>
                <select
                  className="input"
                  value={selectedTextColumn}
                  disabled={importMutation.isPending || isImportRunning}
                  onChange={(event) => setSelectedTextColumn(event.target.value)}
                >
                  {currentConfigObj?.columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                      {currentConfigObj.text_columns.includes(col) ? ' (text)' : ''}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span className="label">Import size limit</span>
                <input
                  type="number"
                  className="input"
                  min={1}
                  max={50000}
                  value={importLimit}
                  disabled={importMutation.isPending || isImportRunning}
                  onChange={(event) => {
                    const val = Number.parseInt(event.target.value, 10);
                    setImportLimit(Number.isNaN(val) ? 1 : Math.max(1, Math.min(50000, val)));
                  }}
                />
              </label>
            </div>

            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={
                  importMutation.isPending ||
                  isImportRunning ||
                  !selectedConfig ||
                  !selectedSplit ||
                  !selectedTextColumn ||
                  importLimit < 1
                }
              >
                {importMutation.isPending ? 'Starting import…' : 'Start import'}
              </button>
              {isImportRunning ? (
                <span className="muted">
                  An import job is currently running for this collection.
                </span>
              ) : null}
            </div>
          </form>
        ) : null}

        {importMutation.error ? <Notice tone="bad">{importMutation.error.message}</Notice> : null}

        {/* Live import job status & progress */}
        {latestImportJob ? (
          <div
            style={{
              marginTop: 8,
              padding: '12px 16px',
              borderRadius: 'var(--radius, 6px)',
              border: '1px solid var(--color-border, #e5e5e5)',
              background: 'var(--color-surface, #fff)',
            }}
          >
            {(() => {
              const step = progress(latestImportJob);
              return (
                <div className="stack" style={{ gap: 8 }}>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <strong>Import job</strong>{' '}
                      <span className="muted" style={{ fontSize: '0.85em' }}>
                        ID: {latestImportJob.id.slice(0, 8)}
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <Chip tone={tones[latestImportJob.status] ?? 'idle'}>
                        {latestImportJob.cancel_requested && latestImportJob.status === 'running'
                          ? 'canceling'
                          : latestImportJob.status}
                      </Chip>
                      {['pending', 'running', 'failed', 'cancelled'].includes(
                        latestImportJob.status,
                      ) ? (
                        <button
                          type="button"
                          className="btn btn-sm"
                          disabled={
                            (latestImportJob.cancel_requested &&
                              latestImportJob.status === 'running') ||
                            changeJobMutation.isPending
                          }
                          onClick={() =>
                            changeJobMutation.mutate({
                              id: latestImportJob.id,
                              action: ['failed', 'cancelled'].includes(latestImportJob.status)
                                ? 'retry'
                                : 'cancel',
                            })
                          }
                        >
                          {['failed', 'cancelled'].includes(latestImportJob.status)
                            ? 'Retry'
                            : 'Cancel'}
                        </button>
                      ) : null}
                    </div>
                  </div>

                  <div>
                    {latestImportJob.error ? (
                      <span className="job-error">{latestImportJob.error}</span>
                    ) : (
                      <span>{step.message || 'Waiting to start'}</span>
                    )}
                  </div>

                  {/* Show counts or bounded progress */}
                  {step.total > 0 ? (
                    <div className="muted" style={{ fontSize: '0.875rem' }}>
                      {step.completed} / {step.total} records ({step.imported ?? 0} imported,{' '}
                      {step.skipped ?? 0} skipped, {step.failed ?? 0} failed)
                    </div>
                  ) : step.processed !== undefined && step.processed > 0 ? (
                    <div className="muted" style={{ fontSize: '0.875rem' }}>
                      {step.processed} records processed ({step.imported ?? 0} imported,{' '}
                      {step.skipped ?? 0} skipped, {step.failed ?? 0} failed)
                    </div>
                  ) : null}
                </div>
              );
            })()}
          </div>
        ) : null}
      </div>
    </Card>
  );
}
