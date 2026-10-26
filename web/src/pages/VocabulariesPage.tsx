import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type Dataset, type Tokenizer } from '../api/client';
import JobsPanel, { tones } from '../components/JobsPanel';
import { Card, CardHead, Chip, ConfirmButton, Empty, ErrorBox, Notice } from '../components/ui';
import { relativeDate } from '../lib/format';

export default function VocabulariesPage() {
  const queryClient = useQueryClient();
  const snapshots = useQuery({ queryKey: ['datasets'], queryFn: api.listDatasets });
  const tokenizers = useQuery({
    queryKey: ['tokenizers'],
    queryFn: api.listTokenizers,
    refetchInterval: 3000,
  });
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['tokenizers'] });
    void queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };
  const remove = useMutation({ mutationFn: api.deleteTokenizer, onSuccess: refresh });
  const error = snapshots.error ?? tokenizers.error;
  const ready = tokenizers.data?.items.filter((item) => item.status === 'ready') ?? [];

  return (
    <>
      <div className="page-intro">
        <h1 className="h1">Vocabularies</h1>
        <p className="lede">
          Learn word pieces from a snapshot’s training files once, then reuse them for every
          training set.
        </p>
      </div>
      {error ? (
        <ErrorBox
          message={error.message}
          onRetry={() => {
            void snapshots.refetch();
            void tokenizers.refetch();
          }}
        />
      ) : snapshots.isPending || tokenizers.isPending ? (
        <p className="muted" role="status">
          Loading vocabularies…
        </p>
      ) : (
        <>
          <TokenizerForm snapshots={snapshots.data?.items ?? []} onCreated={refresh} />
          <Card>
            <CardHead title="Vocabularies" />
            {tokenizers.data?.items.length ? (
              <div className="scroll-x">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Status</th>
                      <th>Snapshot</th>
                      <th className="num">Word pieces</th>
                      <th>Created</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tokenizers.data.items.map((item) => (
                      <tr key={item.id}>
                        <td className="name">
                          {item.name}
                          {item.error ? <Notice tone="bad">{item.error}</Notice> : null}
                        </td>
                        <td>
                          <Chip tone={tones[item.status] ?? 'idle'}>{item.status}</Chip>
                        </td>
                        <td>{item.dataset_name}</td>
                        <td className="num">
                          {item.vocab_size?.toLocaleString() ?? '—'}
                          <div className="muted">
                            {item.requested_vocab_size.toLocaleString()} requested
                          </div>
                        </td>
                        <td className="muted">{relativeDate(item.created_at)}</td>
                        <td>
                          <span className="actions">
                            {item.status === 'ready' ? (
                              <a className="btn btn-sm" href={api.tokenizerDownloadUrl(item.id)}>
                                Download
                              </a>
                            ) : null}
                            <ConfirmButton
                              label="Delete"
                              confirmLabel={`Delete ${item.name}`}
                              disabled={remove.isPending}
                              onConfirm={() => remove.mutate(item.id)}
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
                  title="No vocabularies yet"
                  hint="Choose a snapshot above and make the first one."
                />
              </div>
            )}
            {remove.error ? (
              <div className="card-body">
                <Notice tone="bad">{remove.error.message}</Notice>
              </div>
            ) : null}
          </Card>
          {ready.length ? <TryVocabulary tokenizers={ready} /> : null}
        </>
      )}
      <JobsPanel />
    </>
  );
}

function TokenizerForm({ snapshots, onCreated }: { snapshots: Dataset[]; onCreated: () => void }) {
  const [name, setName] = useState('');
  const [vocabSize, setVocabSize] = useState(4096);
  const [chosen, setChosen] = useState('');
  const usable = snapshots.filter((snapshot) => snapshot.download_ready);
  const snapshotId = usable.some((snapshot) => snapshot.id === chosen)
    ? chosen
    : (usable[0]?.id ?? '');
  const validSize = Number.isInteger(vocabSize) && vocabSize >= 257 && vocabSize <= 65535;
  const create = useMutation({
    mutationFn: () =>
      api.createTokenizer({ name: name.trim(), dataset_id: snapshotId, vocab_size: vocabSize }),
    onSuccess: () => {
      setName('');
      onCreated();
    },
  });
  return (
    <Card>
      <CardHead title="Make a vocabulary" />
      <div className="card-body">
        {usable.length === 0 ? (
          <p>
            Save a snapshot on <Link to="/datasets">Datasets</Link> first.
          </p>
        ) : (
          <form
            className="stack"
            onSubmit={(event) => {
              event.preventDefault();
              if (!create.isPending && name.trim() && snapshotId && validSize) create.mutate();
            }}
          >
            <div className="form-grid">
              <label className="field">
                <span className="label">Vocabulary name</span>
                <input
                  className="input"
                  aria-label="Vocabulary name"
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
                  onChange={(event) => setChosen(event.target.value)}
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
                <span className="label">Vocabulary size</span>
                <input
                  className="input"
                  aria-label="Vocabulary size"
                  type="number"
                  min="257"
                  max="65535"
                  value={Number.isFinite(vocabSize) ? vocabSize : ''}
                  onChange={(event) =>
                    setVocabSize(event.target.value === '' ? NaN : Number(event.target.value))
                  }
                  disabled={create.isPending}
                />
                <span className="hint">
                  257 to 65,535 word pieces. Small snapshots can produce fewer pieces than
                  requested.
                </span>
              </label>
            </div>
            <p className="muted">
              Only the snapshot’s training files teach the vocabulary. It runs in the background, so
              you can leave this page.
            </p>
            {create.error ? <Notice tone="bad">{create.error.message}</Notice> : null}
            {create.isSuccess ? (
              <p role="status">Vocabulary queued. Follow its progress below.</p>
            ) : null}
            <div>
              <button
                className="btn btn-primary"
                type="submit"
                disabled={!name.trim() || !snapshotId || !validSize || create.isPending}
              >
                {create.isPending ? 'Queuing…' : 'Make vocabulary'}
              </button>
            </div>
          </form>
        )}
      </div>
    </Card>
  );
}

function TryVocabulary({ tokenizers }: { tokenizers: Tokenizer[] }) {
  const [chosen, setChosen] = useState('');
  const [text, setText] = useState('A quick example from my notes.');
  const tokenizerId = tokenizers.some((item) => item.id === chosen)
    ? chosen
    : (tokenizers[0]?.id ?? '');
  const preview = useMutation({ mutationFn: () => api.previewTokenizer(tokenizerId, text) });
  return (
    <Card>
      <CardHead title="Try a vocabulary" />
      <form
        className="card-body stack"
        onSubmit={(event) => {
          event.preventDefault();
          if (text.trim() && tokenizerId) preview.mutate();
        }}
      >
        <label className="field">
          <span className="label">Vocabulary</span>
          <select
            className="input"
            aria-label="Vocabulary to try"
            value={tokenizerId}
            onChange={(event) => {
              setChosen(event.target.value);
              preview.reset();
            }}
          >
            {tokenizers.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="label">Sample text</span>
          <textarea
            className="input token-sample"
            aria-label="Sample text"
            maxLength={2000}
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              preview.reset();
            }}
          />
        </label>
        <button className="btn" type="submit" disabled={!text.trim() || preview.isPending}>
          {preview.isPending ? 'Reading text…' : 'Show word pieces'}
        </button>
        {preview.error ? <Notice tone="bad">{preview.error.message}</Notice> : null}
        {preview.data ? (
          <div role="status">
            <p>{preview.data.token_count} word pieces in this sample.</p>
            <pre className="preview">
              {preview.data.pieces
                .map((piece, index) => `${preview.data?.ids[index]}: ${piece}`)
                .join('\n')}
            </pre>
          </div>
        ) : null}
      </form>
    </Card>
  );
}
