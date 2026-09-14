import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { ApiError, api, type Bucket } from '../api/client';
import { Card, CardHead, Chip, Empty, ErrorBox, Field, Notice } from '../components/ui';
import { compactNumber, plural, relativeDate } from '../lib/format';

export default function BucketsPage() {
  const queryClient = useQueryClient();
  const buckets = useQuery({ queryKey: ['buckets'], queryFn: api.listBuckets });

  return (
    <>
      <div>
        <h1 className="h1">Buckets</h1>
        <p className="lede">
          Use buckets to group documents by source. This keeps the data organized and makes later
          testing easier.
        </p>
      </div>

      <NewBucketForm onCreated={() => queryClient.invalidateQueries({ queryKey: ['buckets'] })} />

      <BucketTable
        isPending={buckets.isPending}
        error={buckets.error}
        buckets={buckets.data?.items ?? []}
        total={buckets.data?.total ?? 0}
        onRetry={() => void buckets.refetch()}
      />
    </>
  );
}

function NewBucketForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const create = useMutation({
    mutationFn: () => api.createBucket(name.trim(), description.trim()),
    onSuccess: () => {
      setName('');
      setDescription('');
      onCreated();
    },
  });

  return (
    <Card>
      <form
        className="card-body row-end"
        onSubmit={(event) => {
          event.preventDefault();
          if (name.trim()) create.mutate();
        }}
      >
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="gutenberg fiction"
          maxLength={64}
          width={230}
        />
        <div className="grow">
          <Field
            label="Where it came from"
            value={description}
            onChange={setDescription}
            placeholder="Project Gutenberg, public domain, downloaded August 2026"
            maxLength={2000}
          />
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!name.trim() || create.isPending}
        >
          {create.isPending ? 'Creating.' : 'Create bucket'}
        </button>
      </form>
      {create.error ? (
        <div style={{ padding: '0 16px 16px' }}>
          <Notice tone="bad">{create.error.message}</Notice>
        </div>
      ) : null}
    </Card>
  );
}

function BucketTable({
  isPending,
  error,
  buckets,
  total,
  onRetry,
}: {
  isPending: boolean;
  error: Error | null;
  buckets: Bucket[];
  total: number;
  onRetry: () => void;
}) {
  const queryClient = useQueryClient();
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteBucket(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['buckets'] }),
  });

  if (isPending) return <p className="muted">Loading buckets.</p>;

  if (error) {
    const hint = error instanceof ApiError && error.status === 401 ? ' Check the API token.' : '';
    return <ErrorBox message={error.message + hint} onRetry={onRetry} />;
  }

  if (buckets.length === 0) {
    return <Empty title="No buckets yet" hint="Create one above, then open it to upload files." />;
  }

  const estimatedTokens = buckets.reduce((sum, bucket) => sum + bucket.stats.est_tokens, 0);

  return (
    <Card>
      <CardHead
        title={plural(total, 'bucket')}
        end={<span className="muted">{compactNumber(estimatedTokens)} estimated tokens</span>}
      />
      {remove.error ? (
        <div style={{ padding: '12px 16px 0' }}>
          <Notice tone="bad">{remove.error.message}</Notice>
        </div>
      ) : null}
      <div className="scroll-x">
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Source</th>
              <th className="num">Documents</th>
              <th className="num">Characters</th>
              <th className="num">Est. tokens</th>
              <th>Created</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {buckets.map((bucket) => (
              <tr key={bucket.id}>
                <td className="name">
                  <Link to={`/data/${bucket.id}`}>{bucket.name}</Link>
                </td>
                <td className="muted">{bucket.description || 'None'}</td>
                <td className="num">
                  {bucket.stats.documents === 0 ? (
                    <span className="muted">empty</span>
                  ) : (
                    bucket.stats.documents.toLocaleString()
                  )}
                </td>
                <td className="num">{compactNumber(bucket.stats.chars)}</td>
                <td className="num">{compactNumber(bucket.stats.est_tokens)}</td>
                <td className="muted">{relativeDate(bucket.created_at)}</td>
                <td className="num">
                  {bucket.stats.documents > 0 ? (
                    <Chip tone="idle">{plural(bucket.stats.documents, 'document')}</Chip>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-link"
                      disabled={remove.isPending && remove.variables === bucket.id}
                      onClick={() => remove.mutate(bucket.id)}
                    >
                      {remove.isPending && remove.variables === bucket.id ? 'Deleting.' : 'Delete'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}