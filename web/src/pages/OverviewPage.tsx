import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type Bucket } from '../api/client';
import { Card, CardHead, Dropzone, ErrorBox, Notice } from '../components/ui';
import { plural } from '../lib/format';

export default function OverviewPage() {
  const buckets = useQuery({
    queryKey: ['buckets'],
    queryFn: api.listBuckets,
    refetchInterval: 5000,
  });
  const documents = buckets.data?.items.reduce((n, bucket) => n + bucket.stats.documents, 0);

  return (
    <>
      <div className="overview-heading">
        <div>
          <h1 className="h1">Overview</h1>
          <p className="lede">Add source files and keep your data ready for training.</p>
        </div>
        <Link to="/data">Manage data →</Link>
      </div>
      {buckets.isPending ? (
        <p className="muted" role="status">
          Loading your data…
        </p>
      ) : buckets.error ? (
        <ErrorBox message={buckets.error.message} onRetry={() => void buckets.refetch()} />
      ) : buckets.data.items.length === 0 ? (
        <Card>
          <div className="card-body stack">
            <h2 className="h2">Start with your data</h2>
            <p className="muted">Create a bucket for a source of text, then add your files.</p>
            <div>
              <Link className="btn btn-primary" to="/data">
                Create a bucket
              </Link>
            </div>
          </div>
        </Card>
      ) : (
        <>
          <p className="overview-summary">
            {plural(buckets.data.total, 'bucket')} · {plural(documents ?? 0, 'document')}
          </p>
          <AddData buckets={buckets.data.items} />
        </>
      )}
    </>
  );
}

function AddData({ buckets }: { buckets: Bucket[] }) {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState('');
  const bucket = buckets.find((item) => item.id === selectedId) ?? buckets[0];
  const upload = useMutation({
    mutationFn: ({ id, files }: { id: string; files: File[] }) =>
      api.uploadDocuments(id, files, ''),
    onSuccess: (_data, { id }) => {
      void queryClient.invalidateQueries({ queryKey: ['buckets'] });
      void queryClient.invalidateQueries({ queryKey: ['bucket', id] });
      void queryClient.invalidateQueries({ queryKey: ['documents', id] });
    },
  });
  const result = upload.variables?.id === bucket.id ? upload.data : undefined;
  const error = upload.variables?.id === bucket.id ? upload.error : null;
  const notParsed = bucket.stats.documents - bucket.stats.parsed_documents;

  return (
    <Card>
      <CardHead title="Add data" />
      <div className="card-body stack">
        <label className="field overview-bucket">
          <span className="label">Bucket</span>
          <select
            className="input"
            value={bucket.id}
            disabled={upload.isPending}
            onChange={(event) => {
              setSelectedId(event.target.value);
              upload.reset();
            }}
          >
            {buckets.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <Dropzone
          onFiles={(files) => upload.mutate({ id: bucket.id, files })}
          busy={upload.isPending}
          compact
        />
        {error ? <Notice tone="bad">{error.message}</Notice> : null}
        {result ? (
          <div className="stack" role="status">
            <p>
              {result.parsed} parsed · {result.failed} failed
            </p>
            {result.failed > 0 ? (
              <details className="upload-errors">
                <summary>See upload errors</summary>
                <ul>
                  {result.outcomes
                    .filter((outcome) => outcome.status === 'failed')
                    .map((outcome, index) => (
                      <li key={index}>
                        <strong>{outcome.filename}</strong>: {outcome.error}
                      </li>
                    ))}
                </ul>
              </details>
            ) : null}
          </div>
        ) : null}
        <div className="dash-foot">
          <span className="muted">
            {bucket.stats.documents === 0
              ? 'No files in this bucket yet'
              : `${bucket.stats.parsed_documents.toLocaleString()} parsed${notParsed > 0 ? ` · ${notParsed.toLocaleString()} not parsed` : ''}`}
          </span>
          <Link to={`/data/${bucket.id}`}>Review documents →</Link>
        </div>
      </div>
    </Card>
  );
}
