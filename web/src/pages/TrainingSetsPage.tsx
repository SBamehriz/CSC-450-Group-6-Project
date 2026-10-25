import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type Dataset, type Tokenizer } from '../api/client';
import JobsPanel, { tones } from '../components/JobsPanel';
import { Card, CardHead, Chip, ConfirmButton, Empty, ErrorBox, Notice } from '../components/ui';
import { compactNumber, relativeDate } from '../lib/format';

export default function TrainingSetsPage() {
  const queryClient = useQueryClient();
  const snapshots = useQuery({ queryKey: ['datasets'], queryFn: api.listDatasets });
  const tokenizers = useQuery({
    queryKey: ['tokenizers'],
    queryFn: api.listTokenizers,
    refetchInterval: 3000,
  });
  const builds = useQuery({ queryKey: ['builds'], queryFn: api.listBuilds, refetchInterval: 3000 });
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['builds'] });
    void queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };
  const remove = useMutation({ mutationFn: api.deleteBuild, onSuccess: refresh });
  const error = snapshots.error ?? tokenizers.error ?? builds.error;
  const names = new Map(tokenizers.data?.items.map((item) => [item.id, item.name]));

  return (
    <>
      <div className="page-intro">
        <h1 className="h1">Training sets</h1>
        <p className="lede">
          Turn a snapshot into token files with one of your vocabularies. A ready training set never
          changes, so every model trained on it sees the same data.
        </p>
      </div>
      {error ? (
        <ErrorBox
          message={error.message}
          onRetry={() => {
            void snapshots.refetch();
            void tokenizers.refetch();
            void builds.refetch();
          }}
        />
      ) : snapshots.isPending || tokenizers.isPending || builds.isPending ? (
        <p className="muted" role="status">
          Loading training sets…
        </p>
      ) : (
        <>
          <BuildForm
            snapshots={snapshots.data?.items ?? []}
            tokenizers={tokenizers.data?.items ?? []}
            onCreated={refresh}
          />
          <Card>
            <CardHead title="Training sets" />
            {builds.data?.items.length ? (
              <div className="scroll-x">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Status</th>
                      <th>Made from</th>
                      <th className="num">Training tokens</th>
                      <th className="num">Validation tokens</th>
                      <th>Created</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {builds.data.items.map((build) => (
                      <tr key={build.id}>
                        <td className="name">
                          {build.name}
                          {build.error ? <Notice tone="bad">{build.error}</Notice> : null}
                        </td>
                        <td>
                          <Chip tone={tones[build.status] ?? 'idle'}>{build.status}</Chip>
                        </td>
                        <td>
                          {build.dataset_name}
                          <div className="muted">{names.get(build.tokenizer_id) ?? '—'}</div>
                        </td>
                        <td className="num">
                          {build.stats.train ? compactNumber(build.stats.train.tokens) : '—'}
                        </td>
                        <td className="num">
                          {build.stats.validation
                            ? compactNumber(build.stats.validation.tokens)
                            : '—'}
                        </td>
                        <td className="muted">{relativeDate(build.created_at)}</td>
                        <td>
                          <span className="actions">
                            {build.status === 'ready' ? (
                              <a className="btn btn-sm" href={api.buildManifestUrl(build.id)}>
                                Manifest
                              </a>
                            ) : null}
                            <ConfirmButton
                              label="Delete"
                              confirmLabel={`Delete ${build.name}`}
                              disabled={remove.isPending}
                              onConfirm={() => remove.mutate(build.id)}
                            />
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="card-body">
                <Empty
                  title="No training sets yet"
                  hint="Pick a snapshot and a ready vocabulary above to make one."
                />
              </div>
            )}
            {remove.error ? (
              <div className="card-body">
                <Notice tone="bad">{remove.error.message}</Notice>
              </div>
            ) : null}
          </Card>
        </>
      )}
      <JobsPanel />
    </>
  );
}

function BuildForm({
  snapshots,
  tokenizers,
  onCreated,
}: {
  snapshots: Dataset[];
  tokenizers: Tokenizer[];
  onCreated: () => void;
}) {
  const [name, setName] = useState('');
  const [chosenSnapshot, setChosenSnapshot] = useState('');
  const [chosenTokenizer, setChosenTokenizer] = useState('');
  const usable = snapshots.filter((snapshot) => snapshot.download_ready);
  const ready = tokenizers.filter((item) => item.status === 'ready');
  const snapshotId = usable.some((snapshot) => snapshot.id === chosenSnapshot)
    ? chosenSnapshot
    : (usable[0]?.id ?? '');
  const tokenizerId = ready.some((item) => item.id === chosenTokenizer)
    ? chosenTokenizer
    : (ready[0]?.id ?? '');
  const create = useMutation({
    mutationFn: () =>
      api.createBuild({ name: name.trim(), dataset_id: snapshotId, tokenizer_id: tokenizerId }),
    onSuccess: () => {
      setName('');
      onCreated();
    },
  });
  return (
    <Card>
      <CardHead title="Make a training set" />
      <div className="card-body">
        {usable.length === 0 ? (
          <p>
            Save a snapshot on <Link to="/datasets">Datasets</Link> first.
          </p>
        ) : ready.length === 0 ? (
          <p>
            <Link to="/vocabularies">Make a vocabulary</Link> before making a training set.
          </p>
        ) : (
          <form
            className="stack"
            onSubmit={(event) => {
              event.preventDefault();
              if (!create.isPending && name.trim() && snapshotId && tokenizerId) create.mutate();
            }}
          >
            <div className="form-grid">
              <label className="field">
                <span className="label">Training set name</span>
                <input
                  className="input"
                  aria-label="Training set name"
                  required
                  maxLength={64}
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  disabled={create.isPending}
                />
              </label>
              <label className="field">
                <span className="label">Snapshot</span>
                <select
                  className="input"
                  aria-label="Snapshot"
                  value={snapshotId}
                  onChange={(event) => setChosenSnapshot(event.target.value)}
                  disabled={create.isPending}
                >
                  {usable.map((snapshot) => (
                    <option key={snapshot.id} value={snapshot.id}>
                      {snapshot.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="label">Vocabulary</span>
                <select
                  className="input"
                  aria-label="Vocabulary"
                  value={tokenizerId}
                  onChange={(event) => setChosenTokenizer(event.target.value)}
                  disabled={create.isPending}
                >
                  {ready.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p className="muted">
              Training and validation files keep the snapshot’s split. Each document ends with token
              0.
            </p>
            {create.error ? <Notice tone="bad">{create.error.message}</Notice> : null}
            {create.isSuccess ? (
              <p role="status">Training set queued. Follow its progress below.</p>
            ) : null}
            <div>
              <button
                className="btn btn-primary"
                type="submit"
                disabled={!name.trim() || !snapshotId || !tokenizerId || create.isPending}
              >
                {create.isPending ? 'Queuing…' : 'Make training set'}
              </button>
            </div>
          </form>
        )}
      </div>
    </Card>
  );
}
