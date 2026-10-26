import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import { api } from '../api/client';
import { Card, CardHead, ConfirmButton, ErrorBox, Notice } from '../components/ui';
import { compactNumber } from '../lib/format';

export default function PreparedDataPage() {
  const { datasetId = '' } = useParams();
  return <PreparedData key={datasetId} datasetId={datasetId} />;
}

function PreparedData({ datasetId }: { datasetId: string }) {
  const queryClient = useQueryClient();
  const snapshot = useQuery({
    queryKey: ['dataset', datasetId],
    queryFn: () => api.getDataset(datasetId),
  });
  const prepared = useQuery({
    queryKey: ['prepared', datasetId],
    queryFn: () => api.getPrepared(datasetId),
  });
  const tools = useQuery({ queryKey: ['prepareTools'], queryFn: api.preparationTools });
  const [vocabulary, setVocabulary] = useState(4096);
  const [text, setText] = useState('A quick example from my notes.');
  const prepare = useMutation({
    mutationFn: () => api.prepareDataset(datasetId, vocabulary),
    onSuccess: (result) => {
      queryClient.setQueryData(['prepared', datasetId], result);
      void queryClient.invalidateQueries({ queryKey: ['datasets'] });
      void queryClient.invalidateQueries({ queryKey: ['dataset', datasetId] });
      preview.reset();
    },
  });
  const remove = useMutation({
    mutationFn: () => api.deletePrepared(datasetId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['prepared', datasetId] });
      void queryClient.invalidateQueries({ queryKey: ['datasets'] });
      preview.reset();
      prepare.reset();
    },
  });
  const preview = useMutation({ mutationFn: () => api.previewTokens(datasetId, text) });
  const validVocabulary = Number.isInteger(vocabulary) && vocabulary >= 257 && vocabulary <= 16384;
  const busy = prepare.isPending || remove.isPending;

  if (snapshot.error)
    return <ErrorBox message={snapshot.error.message} onRetry={() => void snapshot.refetch()} />;
  if (snapshot.isPending) return <p role="status">Loading snapshot…</p>;
  return (
    <>
      <p>
        <Link to="/datasets">Back to snapshots</Link>
      </p>
      <div className="page-intro">
        <h1 className="h1">Prepare {snapshot.data.name}</h1>
        <p className="lede">
          Turn the saved text into word pieces and token files. This runs on your CPU.
        </p>
      </div>
      <div className="preparation-grid">
        <Card>
          <CardHead title="Prepare this snapshot" />
          <div className="card-body stack">
            <p>
              The vocabulary learns from training documents. Validation stays separate, using the
              same split you saved.
            </p>
            <label className="field">
              <span className="label">Vocabulary size</span>
              <input
                className="input"
                aria-label="Vocabulary size"
                type="number"
                min="257"
                max="16384"
                value={
                  prepared.data?.ready
                    ? prepared.data.requested_vocab_size
                    : Number.isFinite(vocabulary)
                      ? vocabulary
                      : ''
                }
                disabled={busy || prepared.data?.ready}
                onChange={(event) =>
                  setVocabulary(event.target.value === '' ? NaN : Number(event.target.value))
                }
              />
              <span className="hint">
                257 to 16,384 word pieces. Small samples can produce fewer pieces than requested.
              </span>
            </label>
            {!snapshot.data.download_ready ? (
              <Notice tone="warn">This record needs a new text snapshot first.</Notice>
            ) : null}
            {tools.data && !tools.data.available ? (
              <Notice tone="warn">
                Tokenizer tools are not installed. Add the prepare extra from the README.
              </Notice>
            ) : null}
            {tools.error ? (
              <ErrorBox message={tools.error.message} onRetry={() => void tools.refetch()} />
            ) : null}
            {prepared.error ? (
              <ErrorBox message={prepared.error.message} onRetry={() => void prepared.refetch()} />
            ) : null}
            <button
              className="btn btn-primary"
              type="button"
              disabled={
                !snapshot.data.download_ready ||
                !tools.data?.available ||
                prepared.isPending ||
                !!prepared.error ||
                !!prepared.data?.ready ||
                !!prepared.data?.sha256 ||
                !validVocabulary ||
                busy
              }
              onClick={() => prepare.mutate()}
            >
              {prepare.isPending
                ? 'Preparing word pieces…'
                : prepared.data?.ready
                  ? 'Prepared'
                  : 'Prepare data'}
            </button>
            <p className="hint">
              Up to 10 MB of saved text. Keep this page open until the files are ready.
            </p>
            {prepare.error ? <Notice tone="bad">{prepare.error.message}</Notice> : null}
            {remove.error ? <Notice tone="bad">{remove.error.message}</Notice> : null}
          </div>
        </Card>
        <Card>
          <CardHead title="Prepared files" />
          <div className="card-body stack">
            {prepared.isPending ? (
              <p role="status">Checking saved files…</p>
            ) : prepared.data?.ready ? (
              <>
                <dl className="preview-counts">
                  <div>
                    <dt>Training tokens</dt>
                    <dd>{compactNumber(prepared.data.train_tokens)}</dd>
                  </div>
                  <div>
                    <dt>Validation tokens</dt>
                    <dd>{compactNumber(prepared.data.validation_tokens)}</dd>
                  </div>
                  <div>
                    <dt>Actual vocabulary</dt>
                    <dd>{compactNumber(prepared.data.vocab_size)}</dd>
                  </div>
                  <div>
                    <dt>Vocabulary learned from</dt>
                    <dd>{prepared.data.trained_on_documents} training files</dd>
                  </div>
                </dl>
                <a className="btn" href={api.preparedDownloadUrl(datasetId)}>
                  Download prepared ZIP
                </a>
                <p className="hint">
                  Includes train.bin, validation.bin, tokenizer.json, and manifest.json. Each
                  document ends with token 0.
                </p>
                <p className="hint">
                  The files passed their checksum checks. Model training controls are the next step.
                </p>
                <details>
                  <summary>ZIP checksum</summary>
                  <p className="code hash-value">{prepared.data.sha256}</p>
                </details>
              </>
            ) : (
              <p className="muted">
                {prepared.data?.message ||
                  'Prepare the snapshot to get real token counts and downloadable files.'}
              </p>
            )}
            {prepared.data?.sha256 ? (
              <ConfirmButton
                label="Remove prepared files"
                confirmLabel="Remove token files"
                disabled={busy}
                onConfirm={() => remove.mutate()}
              />
            ) : null}
          </div>
        </Card>
      </div>
      {prepared.data?.ready && tools.data?.available ? (
        <Card>
          <CardHead title="Try the tokenizer" />
          <form
            className="card-body stack"
            onSubmit={(event) => {
              event.preventDefault();
              if (text.trim()) preview.mutate();
            }}
          >
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
            <button
              className="btn"
              type="submit"
              disabled={!text.trim() || preview.isPending || busy}
            >
              {preview.isPending ? 'Reading text…' : 'Show tokens'}
            </button>
            {preview.error ? <Notice tone="bad">{preview.error.message}</Notice> : null}
            {preview.data ? (
              <div role="status">
                <p>{preview.data.token_count} tokens in this sample.</p>
                <pre className="preview">
                  {preview.data.pieces
                    .map((piece, index) => `${preview.data?.ids[index]}: ${piece}`)
                    .join('\n')}
                </pre>
              </div>
            ) : null}
          </form>
        </Card>
      ) : null}
    </>
  );
}
