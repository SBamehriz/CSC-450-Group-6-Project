import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { api, type Dataset } from '../api/client';
import {
  Card,
  CardHead,
  Chip,
  ConfirmButton,
  Empty,
  ErrorBox,
  ExplainToggle,
  Field,
  Notice,
} from '../components/ui';
import { compactNumber, relativeDate } from '../lib/format';

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
  const [params] = useSearchParams();
  const [bucketId, setBucketId] = useState(params.get('bucket') ?? '');
  const [validationPercent, setValidationPercent] = useState(10);
  const [seed, setSeed] = useState(42);
  const [explain, setExplain] = useState(false);
  const [search, setSearch] = useState('');
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['datasets'] });
    void queryClient.invalidateQueries({ queryKey: ['overview'] });
  };
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
      setDescription('');
      refresh();
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteDataset(id),
    onSuccess: refresh,
  });
  const eligible =
    buckets.data?.items.filter(
      (bucket) => bucket.stats.parsed_documents >= 2 && bucket.stats.parsed_documents <= 1000,
    ) ?? [];
  const selectedBucket = bucketId
    ? eligible.some((bucket) => bucket.id === bucketId)
      ? bucketId
      : ''
    : (eligible[0]?.id ?? '');
  const source = eligible.find((bucket) => bucket.id === selectedBucket);
  const unavailableSource = !!bucketId && !eligible.some((bucket) => bucket.id === bucketId);
  const validSplit =
    Number.isInteger(validationPercent) && validationPercent >= 1 && validationPercent <= 50;
  const validSeed = Number.isInteger(seed) && seed >= 0 && seed <= 4294967295;
  const canBuild =
    !!name.trim() && !!selectedBucket && validSplit && validSeed && !create.isPending;
  const saved = datasets.data?.items ?? [];
  const visible = saved.filter((dataset) =>
    `${dataset.name} ${dataset.description}`.toLowerCase().includes(search.trim().toLowerCase()),
  );

  return (
    <>
      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">Datasets</h1>
          <p className="lede">
            Save a frozen copy of one collection. You’ll get cleaned text, separate train and
            validation files, and a record of the sources.
          </p>
        </div>
        <ExplainToggle checked={explain} onChange={setExplain} />
      </div>
      {explain ? (
        <div className="help-panel">
          <p>
            <strong>Train</strong> is the text a model learns from. <strong>Validation</strong> is
            held aside to check it later. The split happens by whole document, with at least one in
            each file.
          </p>
          <p>Snapshots contain text. Tokenization and model training come later.</p>
        </div>
      ) : null}
      {buckets.error ? (
        <ErrorBox message={buckets.error.message} onRetry={() => void buckets.refetch()} />
      ) : buckets.isPending ? (
        <p className="muted" role="status">
          Loading buckets…
        </p>
      ) : (
        <Card>
          <CardHead title="Create a text snapshot" />
          <div className="snapshot-workspace">
            <form
              className="card-body stack"
              onSubmit={(event) => {
                event.preventDefault();
                if (canBuild) create.mutate();
              }}
            >
              <Field
                label="Name"
                value={name}
                onChange={setName}
                maxLength={64}
                placeholder="week-5-notes"
                disabled={create.isPending}
              />
              <Field
                label="Description"
                value={description}
                onChange={setDescription}
                maxLength={2000}
                placeholder="A quick note about this version"
                disabled={create.isPending}
              />
              <label className="field">
                <span className="label">Source bucket</span>
                <select
                  className="input"
                  aria-label="Source bucket"
                  aria-describedby="snapshot-source-hint"
                  value={selectedBucket}
                  onChange={(event) => {
                    setBucketId(event.target.value);
                    create.reset();
                  }}
                  disabled={!eligible.length || create.isPending}
                >
                  {unavailableSource ? (
                    <option value="">Choose another source</option>
                  ) : !eligible.length ? (
                    <option value="">Upload two parsed documents first</option>
                  ) : null}
                  {eligible.map((bucket) => (
                    <option key={bucket.id} value={bucket.id}>
                      {bucket.name} ({bucket.stats.parsed_documents} parsed)
                    </option>
                  ))}
                </select>
                <span className="hint" id="snapshot-source-hint">
                  A bucket is the collection your text came from.
                </span>
              </label>
              {unavailableSource ? (
                <Notice tone="warn">
                  The requested bucket needs 2 to 1,000 parsed documents or is no longer available.
                  Choose a source before building.
                </Notice>
              ) : null}
              <label className="field">
                <span className="label">Validation percent</span>
                <input
                  className="input"
                  type="number"
                  aria-label="Validation percent"
                  aria-describedby="snapshot-validation-hint"
                  min="1"
                  max="50"
                  value={Number.isFinite(validationPercent) ? validationPercent : ''}
                  disabled={create.isPending}
                  onChange={(event) =>
                    setValidationPercent(
                      event.target.value === '' ? NaN : Number(event.target.value),
                    )
                  }
                />
                <span className="hint" id="snapshot-validation-hint">
                  Keep 1% to 50% aside for validation.
                </span>
              </label>
              <details className="disclosure">
                <summary>Split settings</summary>
                <label className="field">
                  <span className="label">Split seed</span>
                  <input
                    className="input"
                    type="number"
                    aria-label="Split seed"
                    aria-describedby="snapshot-seed-hint"
                    min="0"
                    max="4294967295"
                    value={Number.isFinite(seed) ? seed : ''}
                    disabled={create.isPending}
                    onChange={(event) =>
                      setSeed(event.target.value === '' ? NaN : Number(event.target.value))
                    }
                  />
                  <span className="hint" id="snapshot-seed-hint">
                    The same source documents and seed give the same split.
                  </span>
                </label>
              </details>
              <div>
                <button className="btn btn-primary" type="submit" disabled={!canBuild}>
                  {create.isPending ? 'Building…' : 'Build snapshot'}
                </button>
                <p className="hint dataset-hint">
                  {!eligible.length
                    ? 'Add 2 to 1,000 parsed documents to one collection first.'
                    : !name.trim()
                      ? 'Give the snapshot a name first.'
                      : !validSplit || !validSeed
                        ? 'Check the split settings.'
                        : 'Up to 20 MB of cleaned text. Quality flags do not exclude a file.'}
                </p>
              </div>
              {create.error ? <Notice tone="bad">{create.error.message}</Notice> : null}
              {create.data?.status === 'ready' ? (
                <p role="status">
                  Saved {create.data.name}.{' '}
                  <a href={api.datasetDownloadUrl(create.data.id)}>Download your snapshot →</a>
                </p>
              ) : null}
            </form>
            <aside className="snapshot-preview" aria-label="Snapshot preview">
              <span className="eyebrow">Your snapshot</span>
              <h2 className="h2">{source?.name ?? 'Choose a source'}</h2>
              <p className="muted">
                {source?.description || 'One collection, saved as it is now.'}
              </p>
              {source ? (
                <>
                  <dl className="preview-counts">
                    <div>
                      <dt>Parsed documents</dt>
                      <dd>{source.stats.parsed_documents}</dd>
                    </div>
                    <div>
                      <dt>Characters</dt>
                      <dd>{compactNumber(source.stats.chars)}</dd>
                    </div>
                    <div>
                      <dt>Not parsed</dt>
                      <dd>{source.stats.documents - source.stats.parsed_documents}</dd>
                    </div>
                  </dl>
                  {validSplit ? (
                    <>
                      <div className="split-bar" aria-hidden="true">
                        <span style={{ width: `${100 - validationPercent}%` }} />
                        <span style={{ width: `${validationPercent}%` }} />
                      </div>
                      <p className="hint">
                        {100 - validationPercent}% train · {validationPercent}% validation target.
                        Whole documents can change the final percentages.
                      </p>
                    </>
                  ) : null}
                  <Link to={`/data/${source.id}`}>Review source files →</Link>
                </>
              ) : (
                <p className="hint">Choose a collection with at least two parsed documents.</p>
              )}
              <div className="archive-contents">
                <strong>Inside the ZIP</strong>
                <span>train.jsonl</span>
                <span>validation.jsonl</span>
                <span>manifest.json</span>
                <p className="hint">
                  Rejected and failed files stay out. Review quality flags before building.
                </p>
              </div>
            </aside>
          </div>
        </Card>
      )}
      <div className="overview-heading">
        <h2 className="h2">Saved snapshots</h2>
        <input
          className="input snapshot-search"
          aria-label="Find snapshots"
          placeholder="Search snapshots"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>
      {remove.error ? <Notice tone="bad">{remove.error.message}</Notice> : null}
      {datasets.error ? (
        <ErrorBox message={datasets.error.message} onRetry={() => void datasets.refetch()} />
      ) : datasets.isPending ? (
        <p className="muted" role="status">
          Loading datasets…
        </p>
      ) : !saved.length ? (
        <Empty title="No datasets yet" hint="Upload documents, then build a snapshot above." />
      ) : !visible.length ? (
        <Empty
          title="No matching snapshots"
          hint="Try another name."
          action={
            <button className="btn" type="button" onClick={() => setSearch('')}>
              Clear search
            </button>
          }
        />
      ) : (
        <div className="snapshot-grid">
          {visible.map((dataset) => (
            <article className="card snapshot-card" key={dataset.id} aria-label={dataset.name}>
              <div className="overview-heading">
                <h3 className="h2">{dataset.name}</h3>
                <Chip
                  tone={
                    dataset.status === 'ready' ? 'ok' : dataset.status === 'failed' ? 'bad' : 'idle'
                  }
                >
                  {dataset.status}
                </Chip>
              </div>
              <p className="muted">
                {dataset.description || 'Text snapshot'} · {relativeDate(dataset.created_at)}
              </p>
              <dl className="collection-counts">
                <div>
                  <dt>Train</dt>
                  <dd>{count(dataset, 'train_documents')}</dd>
                </div>
                <div>
                  <dt>Validation</dt>
                  <dd>{count(dataset, 'validation_documents')}</dd>
                </div>
                <div>
                  <dt>Characters</dt>
                  <dd>{compactNumber(dataset.char_count)}</dd>
                </div>
              </dl>
              <details className="disclosure">
                <summary>Snapshot details</summary>
                <dl className="preview-counts">
                  <div>
                    <dt>Source</dt>
                    <dd>
                      {buckets.data?.items.find((bucket) => bucket.id === dataset.bucket_id)
                        ?.name ??
                        dataset.bucket_id ??
                        'No source recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Split seed</dt>
                    <dd>{String(dataset.config.seed ?? 'Not recorded')}</dd>
                  </div>
                  <div>
                    <dt>Archive hash</dt>
                    <dd className="code">{String(dataset.config.sha256 ?? 'Not recorded')}</dd>
                  </div>
                </dl>
                <p className="hint">The manifest keeps each source document’s hash and split.</p>
              </details>
              <div className="collection-card-actions">
                {dataset.status === 'ready' ? (
                  <a className="btn" href={api.datasetDownloadUrl(dataset.id)}>
                    Download ZIP
                  </a>
                ) : null}
                <ConfirmButton
                  label="Delete"
                  confirmLabel={`Delete ${dataset.name}`}
                  disabled={remove.isPending}
                  onConfirm={() => remove.mutate(dataset.id)}
                />
              </div>
            </article>
          ))}
        </div>
      )}
      <p className="muted">
        Need more text? <Link to="/data">Open your collections</Link>.
      </p>
    </>
  );
}
