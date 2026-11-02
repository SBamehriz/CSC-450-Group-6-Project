import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';

import { api, type Job } from '../api/client';
import { relativeDate } from '../lib/format';
import { Card, CardHead, Chip, ErrorBox, Notice, type Tone } from './ui';

export const tones: Record<string, Tone> = {
  ready: 'ok',
  completed: 'ok',
  failed: 'bad',
  cancelled: 'warn',
  pending: 'idle',
  building: 'idle',
  running: 'idle',
};

function targetName(job: Job): string {
  if (typeof job.payload.target_name === 'string') return job.payload.target_name;
  if (job.job_type === 'eval') return 'Model Evaluation';
  return 'Job';
}

function targetLink(job: Job): string {
  if (job.job_type === 'dataset_import') {
    const targetId = typeof job.payload.target_id === 'string' ? job.payload.target_id : '';
    return targetId ? `/data/${targetId}` : '/data';
  }
  if (job.job_type === 'snapshot_create') {
    return '/datasets';
  }
  if (job.job_type === 'eval') {
    return '/evaluations';
  }
  return job.job_type === 'dataset_build' ? '/training-sets' : '/vocabularies';
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

export default function JobsPanel() {
  const queryClient = useQueryClient();
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.listJobs, refetchInterval: 3000 });
  const change = useMutation({
    mutationFn: ({ id, action }: { id: string; action: 'cancel' | 'retry' }) =>
      api.changeJob(id, action),
    onSuccess: () => {
      for (const key of [
        'jobs',
        'builds',
        'tokenizers',
        'buckets',
        'bucket',
        'documents',
        'overview',
      ])
        void queryClient.invalidateQueries({ queryKey: [key] });
    },
  });
  const rows = jobs.data?.items ?? [];
  return (
    <Card>
      <CardHead title="Background jobs" />
      <div className="card-body stack">
        {jobs.error ? (
          <ErrorBox message={jobs.error.message} onRetry={() => void jobs.refetch()} />
        ) : jobs.isPending ? (
          <p className="muted" role="status">
            Loading jobs…
          </p>
        ) : !rows.length ? (
          <p className="muted">No jobs yet. Making a vocabulary or a training set starts one.</p>
        ) : (
          <div className="scroll-x">
            <table className="table">
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((job) => {
                  const step = progress(job);
                  return (
                    <tr key={job.id}>
                      <td>
                        <Link to={targetLink(job)}>{targetName(job)}</Link>
                        <div className="muted">{relativeDate(job.created_at)}</div>
                      </td>
                      <td>
                        <Chip tone={tones[job.status] ?? 'idle'}>
                          {job.cancel_requested && job.status === 'running'
                            ? 'canceling'
                            : job.status}
                        </Chip>
                      </td>
                      <td>
                        {job.error ? (
                          <span className="job-error">{job.error}</span>
                        ) : (
                          step.message || 'Waiting to start'
                        )}
                        {job.job_type === 'dataset_import' ? (
                          step.total > 0 ? (
                            <div className="muted">
                              {step.completed} / {step.total} records
                            </div>
                          ) : step.processed !== undefined && step.processed > 0 ? (
                            <div className="muted">{step.processed} records processed</div>
                          ) : null
                        ) : step.total ? (
                          <div className="muted">
                            {step.completed} / {step.total} documents
                          </div>
                        ) : null}
                      </td>
                      <td>
                        {['pending', 'running', 'failed', 'cancelled'].includes(job.status) ? (
                          <button
                            type="button"
                            className="btn btn-sm"
                            disabled={
                              (job.cancel_requested && job.status === 'running') || change.isPending
                            }
                            onClick={() =>
                              change.mutate({
                                id: job.id,
                                action: ['failed', 'cancelled'].includes(job.status)
                                  ? 'retry'
                                  : 'cancel',
                              })
                            }
                          >
                            {['failed', 'cancelled'].includes(job.status) ? 'Retry' : 'Cancel'}
                          </button>
                        ) : (
                          '—'
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {change.error ? <Notice tone="bad">{change.error.message}</Notice> : null}
      </div>
    </Card>
  );
}
