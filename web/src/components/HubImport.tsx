import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

import { api } from '../api/client';
import JobStatus, { useJob } from './JobStatus';
import { Card, CardHead, Field, Notice } from './ui';
import { jobBusy } from '../lib/measure';

export default function HubImport() {
  const queryClient = useQueryClient();
  const [repo, setRepo] = useState('');
  const [revision, setRevision] = useState('main');
  const [name, setName] = useState('');
  const [onCpu, setOnCpu] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, () => {
    void queryClient.invalidateQueries({ queryKey: ['models'] });
  });
  const start = useMutation({
    mutationFn: () =>
      api.importModel({
        repo_id: repo.trim(),
        revision: revision.trim() || 'main',
        name: name.trim() || null,
        on_cpu: onCpu,
      }),
    onSuccess: (started) => setJobId(started.id),
  });
  const busy = start.isPending || jobBusy(job.data);
  const result = job.data?.status === 'completed' ? job.data.result : null;
  const added = typeof result?.name === 'string' ? result.name : null;

  return (
    <Card>
      <CardHead title="Import from Hugging Face" />
      <form
        className="card-body stack"
        onSubmit={(event) => {
          event.preventDefault();
          if (repo.trim() && !busy) start.mutate();
        }}
      >
        <p className="muted" style={{ margin: 0 }}>
          Bring in an open model to measure and compare with yours. Only safe weight files are
          downloaded, and the exact version and licence are kept with it.
        </p>
        <div className="dataset-form">
          <Field
            label="Model on the hub"
            value={repo}
            onChange={setRepo}
            placeholder="openai-community/gpt2"
            maxLength={200}
            disabled={busy}
          />
          <Field
            label="Version"
            value={revision}
            onChange={setRevision}
            maxLength={200}
            disabled={busy}
          />
          <Field
            label="Call it (optional)"
            value={name}
            onChange={setName}
            maxLength={64}
            disabled={busy}
          />
        </div>
        <label className="help-toggle">
          <input
            type="checkbox"
            checked={onCpu}
            disabled={busy}
            onChange={(event) => setOnCpu(event.target.checked)}
          />
          Run it on the CPU (slower)
        </label>
        {start.error ? <Notice tone="bad">{start.error.message}</Notice> : null}
        <JobStatus job={job.data} doneText={added ? `Added ${added} to your models.` : undefined} />
        <div>
          <button type="submit" className="btn btn-primary" disabled={!repo.trim() || busy}>
            {busy ? 'Importing…' : 'Import model'}
          </button>
        </div>
      </form>
    </Card>
  );
}
