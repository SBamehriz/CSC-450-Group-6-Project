import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type Dataset } from '../api/client';
import { Card, CardHead, Chip, Empty, ErrorBox, Field, Notice } from '../components/ui';
import { relativeDate } from '../lib/format';

function count(dataset: Dataset, key: string): number {
  const value = dataset.config[key];
  return typeof value === 'number' ? value : 0;
}

export default function DatasetsPage() {
  const queryClient = useQueryClient();
  const buckets = useQuery({ queryKey: ['buckets'], queryFn: api.listBuckets });
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: api.listDatasets });
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [bucketId, setBucketId] = useState('');
  const [validationPercent, setValidationPercent] = useState(10);
  const [seed, setSeed] = useState(42);

  const create = useMutation({
    mutationFn: () =>
      api.createDataset({
        name: name.trim(),
        description: description.trim(),
        bucket_id: selectedBucket,
        validation_fraction: validationPercent / 100,
        seed,
      }),
    onSuccess: () => {
      setName('');
      void queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
  const remove = useMutation({
    mutationFn: api.deleteDataset,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  });
  const eligible = buckets.data?.items.filter((bucket) => bucket.stats.parsed_documents >= 2) ?? [];
  const selectedBucket = eligible.some((bucket) => bucket.id === bucketId)
    ? bucketId
    : (eligible[0]?.id ?? '');

  return (
    <>
      <div>
        <h1 className="h1">Datasets</h1>
        <p className="lede">
          Freeze cleaned documents from one bucket into train and validation files. Each download
          includes a manifest with source hashes and split settings. Tokenization and training come
          later.
        </p>
      </div>

      {buckets.error ? (
        <ErrorBox message={buckets.error.message} onRetry={() => void buckets.refetch()} />
      ) : buckets.isPending ? (
        <p className="muted" role="status">
          Loading buckets…
        </p>
      ) : (
        <Card>
          <CardHead title="Create a text snapshot" />
          <form
            className="card-body dataset-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (name.trim() && selectedBucket && !create.isPending) {
                create.mutate();
              }
            }}
          >
            <Field label="Name" value={name} onChange={setName} maxLength={64} />
            <Field
              label="Description"
              value={description}
              onChange={setDescription}
              maxLength={2000}
            />
            <label className="field">
              <span className="label">Source bucket</span>
              <select
                className="input"
                value={selectedBucket}
                onChange={(event) => setBucketId(event.target.value)}
                disabled={!eligible.length}
              >
                {!eligible.length ? (
                  <option value="">Upload two parsed documents first</option>
                ) : null}
                {eligible.map((bucket) => (
                  <option key={bucket.id} value={bucket.id}>
                    {bucket.name} ({bucket.stats.parsed_documents} parsed)
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span className="label">Validation percent</span>
              <input
                className="input"
                type="number"
                min="1"
                max="50"
                value={validationPercent}
                onChange={(event) => setValidationPercent(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span className="label">Split seed</span>
              <input
                className="input"
                type="number"
                min="0"
                max="4294967295"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
            </label>
            <div>
              <button
                className="btn btn-primary"
                type="submit"
                disabled={
                  !name.trim() ||
                  !selectedBucket ||
                  create.isPending ||
                  !Number.isInteger(validationPercent) ||
                  validationPercent < 1 ||
                  validationPercent > 50 ||
                  !Number.isInteger(seed) ||
                  seed < 0 ||
                  seed > 4294967295
                }
              >
                {create.isPending ? 'Building…' : 'Build snapshot'}
              </button>
              <p className="muted dataset-hint">Up to 1,000 documents and 20 MB of cleaned text.</p>
            </div>
            {create.error ? <Notice tone="bad">{create.error.message}</Notice> : null}
          </form>
        </Card>
      )}

      {datasets.error ? (
        <ErrorBox message={datasets.error.message} onRetry={() => void datasets.refetch()} />
      ) : datasets.isPending ? (
        <p className="muted" role="status">
          Loading datasets…
        </p>
      ) : !datasets.data?.items.length ? (
        <Empty title="No datasets yet" hint="Upload documents, then build a snapshot above." />
      ) : (
        <Card>
          <CardHead title="Saved snapshots" />
          {remove.error ? (
            <div className="card-body">
              <Notice tone="bad">{remove.error.message}</Notice>
            </div>
          ) : null}
          <div className="scroll-x">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th className="num">Train</th>
                  <th className="num">Validation</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.data.items.map((dataset) => (
                  <tr key={dataset.id}>
                    <td className="name">
                      {dataset.name}
                      {dataset.description ? (
                        <div className="muted">{dataset.description}</div>
                      ) : null}
                    </td>
                    <td>
                      <Chip tone={dataset.status === 'ready' ? 'ok' : 'idle'}>
                        {dataset.status}
                      </Chip>
                    </td>
                    <td className="num">{count(dataset, 'train_documents')}</td>
                    <td className="num">{count(dataset, 'validation_documents')}</td>
                    <td className="muted">{relativeDate(dataset.created_at)}</td>
                    <td className="dataset-actions">
                      {dataset.status === 'ready' ? (
                        <a href={api.datasetDownloadUrl(dataset.id)}>Download ZIP</a>
                      ) : null}
                      <button
                        type="button"
                        className="btn btn-link"
                        disabled={remove.isPending}
                        onClick={() => {
                          if (window.confirm(`Delete ${dataset.name}?`)) remove.mutate(dataset.id);
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      <p className="muted">
        Need a source? <Link to="/data">Open your buckets</Link>.
      </p>
    </>
  );
}
