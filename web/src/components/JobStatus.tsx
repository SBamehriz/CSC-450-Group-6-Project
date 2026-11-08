import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';

import { api, type Job } from '../api/client';
import { jobBusy, jobProgress } from '../lib/measure';
import { Notice } from './ui';

const POLL_MS = 1500;

export function useJob(jobId: string | null, onDone?: (job: Job) => void) {
  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.getJob(jobId as string),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (!query.state.data || jobBusy(query.state.data) ? POLL_MS : false),
  });
  const told = useRef<string | null>(null);
  const done = onDone;
  useEffect(() => {
    const data = job.data;
    if (data && !jobBusy(data) && told.current !== data.id) {
      told.current = data.id;
      done?.(data);
    }
  }, [job.data, done]);
  return job;
}

export default function JobStatus({ job, doneText }: { job: Job | undefined; doneText?: string }) {
  if (!job) return null;
  if (job.status === 'failed' || job.status === 'cancelled') {
    return (
      <Notice tone={job.status === 'failed' ? 'bad' : 'warn'}>
        {job.error ?? (job.status === 'failed' ? 'It stopped with a problem.' : 'Canceled.')}
      </Notice>
    );
  }
  if (job.status === 'completed') {
    return doneText ? (
      <p className="muted" role="status">
        {doneText}
      </p>
    ) : null;
  }
  const step = jobProgress(job);
  return (
    <div className="stack" role="status">
      <p className="muted" style={{ margin: 0 }}>
        {job.status === 'pending' ? 'Waiting its turn…' : step.message || 'Working…'}
      </p>
      <progress
        aria-label="Progress"
        max={step.total || 1}
        value={step.total ? step.completed : undefined}
        style={{ width: '100%' }}
      />
    </div>
  );
}
