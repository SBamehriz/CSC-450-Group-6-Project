import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type Bucket } from '../api/client';
import {
  Card,
  CardHead,
  Chip,
  ConfirmButton,
  Empty,
  ErrorBox,
  Field,
  Icon,
  Notice,
} from '../components/ui';
import { compactNumber, plural, relativeDate } from '../lib/format';

export default function BucketsPage() {
  const queryClient = useQueryClient();
  const buckets = useQuery({ queryKey: ['buckets'], queryFn: api.listBuckets });
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [sort, setSort] = useState('newest');
  const [view, setView] = useState<'cards' | 'list'>('cards');
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['buckets'] });
    void queryClient.invalidateQueries({ queryKey: ['overview'] });
  };
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteBucket(id),
    onSuccess: refresh,
  });
  const all = buckets.data?.items ?? [];
  const visible = all
    .filter(
      (bucket) =>
        `${bucket.name} ${bucket.description}`
          .toLowerCase()
          .includes(search.trim().toLowerCase()) &&
        (filter === 'all' ||
          (filter === 'ready'
            ? bucket.stats.parsed_documents >= 2 && bucket.stats.parsed_documents <= 1000
            : bucket.stats.documents === 0)),
    )
    .sort((a, b) =>
      sort === 'name'
        ? a.name.localeCompare(b.name)
        : sort === 'documents'
          ? b.stats.documents - a.stats.documents || a.name.localeCompare(b.name)
          : b.created_at.localeCompare(a.created_at) || a.id.localeCompare(b.id),
    );

  return (
    <>
      <div className="page-intro">
        <h1 className="h1">Collections</h1>
        <p className="lede">
          Give each source of text its own home. Add files, check their text, and choose a source
          for your next snapshot.
        </p>
      </div>
      <NewCollection onCreated={refresh} />
      {buckets.isPending ? (
        <p className="muted" role="status">
          Loading your collections…
        </p>
      ) : buckets.error ? (
        <ErrorBox message={buckets.error.message} onRetry={() => void buckets.refetch()} />
      ) : !all.length ? (
        <Empty title="No collections yet" hint="Create one above, then open it to add files." />
      ) : (
        <>
          <div className="collection-summary">
            <span>{plural(all.length, 'collection')}</span>
            <span>
              {plural(
                all.reduce((sum, b) => sum + b.stats.documents, 0),
                'document',
              )}
            </span>
            <span>
              About {compactNumber(all.reduce((sum, b) => sum + b.stats.est_tokens, 0))} tokens
            </span>
          </div>
          <div className="collection-toolbar">
            <input
              className="input"
              aria-label="Find collections"
              placeholder="Search name or source"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <select
              className="input"
              aria-label="Filter collections"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="all">All collections</option>
              <option value="ready">Ready for a snapshot</option>
              <option value="empty">No files yet</option>
            </select>
            <select
              className="input"
              aria-label="Sort collections"
              value={sort}
              onChange={(event) => setSort(event.target.value)}
            >
              <option value="newest">Newest first</option>
              <option value="name">Name A to Z</option>
              <option value="documents">Most documents</option>
            </select>
            <div className="seg" aria-label="Collection view">
              <button
                type="button"
                className={view === 'cards' ? 'on' : undefined}
                aria-pressed={view === 'cards'}
                onClick={() => setView('cards')}
              >
                Cards
              </button>
              <button
                type="button"
                className={view === 'list' ? 'on' : undefined}
                aria-pressed={view === 'list'}
                onClick={() => setView('list')}
              >
                List
              </button>
            </div>
          </div>
          <p className="muted result-count" role="status">
            Showing {visible.length} of {all.length} collections
          </p>
          {remove.error ? <Notice tone="bad">{remove.error.message}</Notice> : null}
          {!visible.length ? (
            <Empty
              title="No matches"
              hint="Try another name or clear the filters."
              action={
                <button
                  className="btn"
                  type="button"
                  onClick={() => {
                    setSearch('');
                    setFilter('all');
                  }}
                >
                  Clear filters
                </button>
              }
            />
          ) : view === 'cards' ? (
            <div className="collection-grid">
              {visible.map((bucket) => (
                <CollectionCard
                  key={bucket.id}
                  bucket={bucket}
                  deleting={remove.isPending}
                  onDelete={() => remove.mutate(bucket.id)}
                />
              ))}
            </div>
          ) : (
            <Card>
              <div className="scroll-x">
                <table className="table collection-table">
                  <thead>
                    <tr>
                      <th>Name and source</th>
                      <th className="num">Documents</th>
                      <th className="num">Parsed</th>
                      <th className="num">Characters</th>
                      <th>Created</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.map((bucket) => (
                      <tr key={bucket.id}>
                        <td className="name">
                          <Link to={`/data/${bucket.id}`}>{bucket.name}</Link>
                          <p className="muted table-note">
                            {bucket.description || 'No source note yet.'}
                          </p>
                        </td>
                        <td className="num">{bucket.stats.documents}</td>
                        <td className="num">{bucket.stats.parsed_documents}</td>
                        <td className="num">{compactNumber(bucket.stats.chars)}</td>
                        <td className="muted">{relativeDate(bucket.created_at)}</td>
                        <td>
                          {bucket.stats.documents === 0 ? (
                            <ConfirmButton
                              label="Delete"
                              confirmLabel={`Delete ${bucket.name}`}
                              disabled={remove.isPending}
                              onConfirm={() => remove.mutate(bucket.id)}
                            />
                          ) : (
                            <Link to={`/data/${bucket.id}`}>Review files</Link>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </>
  );
}

function NewCollection({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [created, setCreated] = useState<Bucket | null>(null);
  const create = useMutation({
    mutationFn: () => api.createBucket(name.trim(), description.trim()),
    onSuccess: (bucket) => {
      setName('');
      setDescription('');
      setCreated(bucket);
      onCreated();
    },
  });
  return (
    <Card>
      <CardHead title="New collection" />
      <form
        className="card-body row-end"
        onSubmit={(event) => {
          event.preventDefault();
          if (name.trim() && !create.isPending) create.mutate();
        }}
      >
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="course-notes"
          maxLength={64}
          width={230}
          disabled={create.isPending}
        />
        <div className="grow">
          <Field
            label="Where it came from"
            value={description}
            onChange={setDescription}
            placeholder="A website, handbook, or class notes"
            maxLength={2000}
            disabled={create.isPending}
          />
        </div>
        <button
          className="btn btn-primary"
          type="submit"
          disabled={!name.trim() || create.isPending}
        >
          {create.isPending ? 'Creating…' : 'Create collection'}
        </button>
      </form>
      {create.error ? (
        <div className="card-note">
          <Notice tone="bad">{create.error.message}</Notice>
        </div>
      ) : null}
      {created ? (
        <p className="card-note" role="status">
          Created {created.name}. <Link to={`/data/${created.id}`}>Add files →</Link>
        </p>
      ) : null}
    </Card>
  );
}

function CollectionCard({
  bucket,
  deleting,
  onDelete,
}: {
  bucket: Bucket;
  deleting: boolean;
  onDelete: () => void;
}) {
  const parsed = bucket.stats.parsed_documents;
  const ready = parsed >= 2 && parsed <= 1000;
  return (
    <article className="card collection-card" aria-label={bucket.name}>
      <div className="collection-card-top">
        <Icon name="data" />
        <Chip tone={ready ? 'ok' : 'idle'}>
          {ready
            ? 'Ready for a snapshot'
            : parsed > 1000
              ? 'Over snapshot limit'
              : bucket.stats.documents
                ? 'Add more text'
                : 'No files yet'}
        </Chip>
      </div>
      <h2 className="h2">
        <Link to={`/data/${bucket.id}`}>{bucket.name}</Link>
      </h2>
      <p className="muted collection-description">{bucket.description || 'No source note yet.'}</p>
      <dl className="collection-counts">
        <div>
          <dt>Documents</dt>
          <dd>{bucket.stats.documents}</dd>
        </div>
        <div>
          <dt>Parsed</dt>
          <dd>{parsed}</dd>
        </div>
        <div>
          <dt>Characters</dt>
          <dd>{compactNumber(bucket.stats.chars)}</dd>
        </div>
      </dl>
      <div className="readiness-track" aria-hidden="true">
        <span
          style={{
            width: `${bucket.stats.documents ? (parsed / bucket.stats.documents) * 100 : 0}%`,
          }}
        />
      </div>
      <p className="muted hint">
        {bucket.stats.documents
          ? `${parsed} of ${bucket.stats.documents} parsed`
          : 'Upload your first file'}{' '}
        · {relativeDate(bucket.created_at)}
      </p>
      <div className="collection-card-actions">
        <Link className="btn" to={`/data/${bucket.id}`}>
          {bucket.stats.documents ? 'Review files' : 'Add files'}
        </Link>
        {ready ? (
          <Link to={`/datasets?bucket=${encodeURIComponent(bucket.id)}`}>Save snapshot →</Link>
        ) : bucket.stats.documents === 0 ? (
          <ConfirmButton
            label="Delete"
            confirmLabel={`Delete ${bucket.name}`}
            disabled={deleting}
            onConfirm={onDelete}
          />
        ) : null}
      </div>
    </article>
  );
}
