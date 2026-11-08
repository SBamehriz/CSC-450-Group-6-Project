import { useMutation } from '@tanstack/react-query';
import { useState } from 'react';

import { api } from '../api/client';
import { jobBusy } from '../lib/measure';
import JobStatus, { useJob } from './JobStatus';
import { Notice } from './ui';

export default function ExportButton({
  checkpointId,
  step,
}: {
  checkpointId: string;
  step: number;
}) {
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);
  const start = useMutation({
    mutationFn: () => api.exportCheckpoint(checkpointId),
    onSuccess: (started) => setJobId(started.id),
  });
  if (job.data?.status === 'completed') {
    return (
      <a
        className="btn btn-sm"
        href={api.checkpointExportUrl(checkpointId)}
        download
        aria-label={`Download the Hugging Face folder for step ${step}`}
      >
        Download Hugging Face folder
      </a>
    );
  }
  const busy = start.isPending || jobBusy(job.data);
  return (
    <div className="stack">
      <button
        type="button"
        className="btn btn-sm"
        disabled={busy}
        aria-label={`Export the save point at step ${step} for Hugging Face`}
        onClick={() => start.mutate()}
      >
        {busy ? 'Exporting…' : 'Export for Hugging Face'}
      </button>
      {start.error ? <Notice tone="bad">{start.error.message}</Notice> : null}
      <JobStatus job={job.data} />
    </div>
  );
}
