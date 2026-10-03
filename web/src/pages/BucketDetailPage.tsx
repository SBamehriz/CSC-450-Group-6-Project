import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';

import { DOCUMENT_PAGE, api, type Document } from '../api/client';
import {
  Card,
  CardHead,
  Chip,
  ConfirmButton,
  Empty,
  ErrorBox,
  ExplainToggle,
  Notice,
  type Tone,
} from '../components/ui';
import UploadPanel from '../components/UploadPanel';
import ExtractionDetails from '../components/ExtractionDetails';
import TextPreview from '../components/TextPreview';
import { compactNumber, plural } from '../lib/format';
import { flagText } from '../lib/document';

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'parsed', label: 'Parsed' },
  { key: 'failed', label: 'Failed' },
  { key: 'flagged', label: 'Flagged' },
  { key: 'rejected', label: 'Rejected' },
] as const;

type FilterKey = (typeof FILTERS)[number]['key'];

const STATUS_TONE: Record<string, Tone> = {
  parsed: 'ok',
  failed: 'bad',
  rejected: 'idle',
  pending: 'idle',
};

export default function BucketDetailPage() {
  const { bucketId = '' } = useParams();
  const queryClient = useQueryClient();
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState<FilterKey>(
    () => FILTERS.find((option) => option.key === params.get('filter'))?.key ?? 'all',
  );
  const [search, setSearch] = useState(params.get('q') ?? '');
  const [explain, setExplain] = useState(false);
  const openId = params.get('open');
  const [offset, setOffset] = useState(0);
  const openDocument = (id: string | null) => {
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (id) next.set('open', id);
        else next.delete('open');
        return next;
      },
      { replace: true },
    );
  };

  const bucket = useQuery({
    queryKey: ['bucket', bucketId],
    queryFn: () => api.getBucket(bucketId),
  });

  const documents = useQuery({
    queryKey: ['documents', bucketId, filter, search, offset],
    queryFn: () =>
      api.listDocuments({
        bucketId,
        status: filter === 'all' || filter === 'flagged' ? undefined : filter,
        flagged: filter === 'flagged',
        q: search || undefined,
        offset,
      }),
  });

  const selected = useQuery({
    queryKey: ['document', openId],
    queryFn: () => api.getDocument(openId!),
    enabled: !!openId,
  });

  const refresh = () => {
    setOffset(0);
    void queryClient.invalidateQueries({ queryKey: ['document'] });
    void queryClient.invalidateQueries({ queryKey: ['overview'] });
    void queryClient.invalidateQueries({ queryKey: ['recentUploads'] });
    void queryClient.invalidateQueries({ queryKey: ['documents', bucketId] });
    void queryClient.invalidateQueries({ queryKey: ['bucket', bucketId] });
    void queryClient.invalidateQueries({ queryKey: ['buckets'] });
  };

  if (bucket.isPending) return <p className="muted">Loading collection…</p>;
  if (bucket.error) {
    return <ErrorBox message={bucket.error.message} onRetry={() => void bucket.refetch()} />;
  }

  const stats = bucket.data.stats;
  const open = selected.data?.id === openId ? selected.data : null;

  return (
    <>
      <div className="crumb">
        <Link to="/data">Collections</Link>
        <span>/</span>
        <span>{bucket.data.name}</span>
      </div>

      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">{bucket.data.name}</h1>
          <p className="lede">
            {bucket.data.description ? `${bucket.data.description} ` : ''}
            {plural(stats.documents, 'document')}, {compactNumber(stats.chars)} characters, about{' '}
            {compactNumber(stats.est_tokens)} tokens.
          </p>
        </div>
        <ExplainToggle checked={explain} onChange={setExplain} />
      </div>
      {explain ? (
        <div className="help-panel">
          <p>
            Parsed means the file was read. Quality flags are hints, so check the cleaned text
            before using it. A snapshot includes every parsed file, even if it has flags.
          </p>
          <p>
            Reject keeps a file in this collection but leaves it out of future snapshots. Moving or
            deleting a file does not change a snapshot you already saved.
          </p>
        </div>
      ) : null}
      <div className="collection-summary">
        <span>{plural(stats.parsed_documents, 'parsed document')}</span>
        <span>{stats.documents - stats.parsed_documents} not parsed</span>
        <Link to={`/datasets?bucket=${encodeURIComponent(bucketId)}`}>
          Create a snapshot from this bucket →
        </Link>
      </div>

      <UploadPanel key={bucketId} bucketId={bucketId} onUploaded={() => setOffset(0)} />

      {openId && selected.isPending ? (
        <p className="muted" role="status">
          Loading document details…
        </p>
      ) : null}
      {openId && selected.error ? (
        <ErrorBox message={selected.error.message} onRetry={() => void selected.refetch()} />
      ) : null}
      {open && open.bucket_id !== bucketId ? (
        <Notice tone="warn">
          This document has moved.{' '}
          <Link to={`/data/${open.bucket_id}?open=${open.id}`}>Open its current bucket</Link>.
        </Notice>
      ) : null}
      {open && open.bucket_id === bucketId ? (
        <DocumentPanel
          key={open.id}
          document={open}
          onClose={() => openDocument(null)}
          onChanged={refresh}
        />
      ) : null}
      <Card>
        <CardHead
          title={documents.data ? plural(documents.data.total, 'document') : 'Documents'}
          end={
            <>
              <div className="seg">
                {FILTERS.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    className={filter === option.key ? 'on' : undefined}
                    aria-pressed={filter === option.key}
                    onClick={() => {
                      setFilter(option.key);
                      setOffset(0);
                    }}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <input
                className="input document-search"
                aria-label="Filter by filename"
                placeholder="Filter by filename"
                value={search}
                onChange={(event) => {
                  setSearch(event.target.value);
                  setOffset(0);
                }}
              />
            </>
          }
        />
        <DocumentTable
          isPending={documents.isPending}
          error={documents.error}
          documents={documents.data?.items ?? []}
          openId={openId}
          onOpen={(id) => openDocument(id === openId ? null : id)}
          onRetry={() => void documents.refetch()}
        />
        {documents.data && !documents.error && documents.data.total > 0 ? (
          <div className="card-body document-pager">
            <button
              type="button"
              className="btn"
              disabled={offset === 0 || documents.isFetching}
              onClick={() => {
                setOffset(Math.max(0, offset - DOCUMENT_PAGE));
                openDocument(null);
              }}
            >
              Previous documents
            </button>
            <span className="muted" role="status">
              {documents.data.items.length ? offset + 1 : 0} to{' '}
              {documents.data.items.length ? offset + documents.data.items.length : 0} of{' '}
              {documents.data.total}
            </span>
            <button
              type="button"
              className="btn"
              disabled={offset + DOCUMENT_PAGE >= documents.data.total || documents.isFetching}
              onClick={() => {
                setOffset(offset + DOCUMENT_PAGE);
                openDocument(null);
              }}
            >
              Next documents
            </button>
          </div>
        ) : null}
      </Card>
    </>
  );
}

function DocumentTable({
  isPending,
  error,
  documents,
  openId,
  onOpen,
  onRetry,
}: {
  isPending: boolean;
  error: Error | null;
  documents: Document[];
  openId: string | null;
  onOpen: (id: string) => void;
  onRetry: () => void;
}) {
  if (isPending) return <div className="card-body muted">Loading documents…</div>;
  if (error) {
    return (
      <div className="card-body">
        <ErrorBox message={error.message} onRetry={onRetry} />
      </div>
    );
  }
  if (documents.length === 0) {
    return (
      <div className="card-body">
        <Empty
          title="Nothing here yet"
          hint="Try another filter or upload files above. Failed files still show their reason."
        />
      </div>
    );
  }

  return (
    <div className="scroll-x">
      <table className="table document-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Format</th>
            <th>Status</th>
            <th>Quality</th>
            <th className="num">Characters</th>
            <th className="num">Words</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((document) => (
            <tr
              key={document.id}
              style={{
                background: openId === document.id ? 'var(--page)' : undefined,
              }}
            >
              <td className="name">
                <button
                  type="button"
                  className="btn btn-link file-button"
                  aria-expanded={openId === document.id}
                  onClick={() => onOpen(document.id)}
                >
                  {document.filename}
                </button>
              </td>
              <td className="muted">{document.source_format}</td>
              <td>
                <Chip tone={STATUS_TONE[document.status] ?? 'idle'}>{document.status}</Chip>
              </td>
              <td>
                {document.status === 'parsed' ? (
                  <QualityFlags flags={(document.quality.flags as string[]) ?? []} />
                ) : (
                  <span className="muted">{document.error}</span>
                )}
              </td>
              <td className="num">
                {document.status === 'parsed' || document.status === 'rejected'
                  ? compactNumber(document.char_count)
                  : 'Not available'}
              </td>
              <td className="num">
                {document.status === 'parsed' || document.status === 'rejected'
                  ? compactNumber(document.word_count)
                  : 'Not available'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QualityFlags({ flags }: { flags: string[] }) {
  if (flags.length === 0) return <span className="muted">None</span>;
  return (
    <span style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
      {flags.map((name) => (
        <span key={name} className="flag">
          {flagText(name).label}
        </span>
      ))}
    </span>
  );
}

function DocumentPanel({
  document,
  onClose,
  onChanged,
}: {
  document: Document;
  onClose: () => void;
  onChanged: () => void;
}) {
  const queryClient = useQueryClient();
  const panel = useRef<HTMLElement>(null);
  const [target, setTarget] = useState('');
  const [reason, setReason] = useState('');
  const buckets = useQuery({ queryKey: ['buckets'], queryFn: api.listBuckets });
  useEffect(() => {
    panel.current?.focus({ preventScroll: true });
    panel.current?.scrollIntoView?.({ block: 'start', behavior: 'instant' });
  }, []);
  const done = () => {
    void queryClient.invalidateQueries({ queryKey: ['documentText', document.id] });
    onChanged();
  };
  const reject = useMutation({
    mutationFn: () =>
      api.rejectDocument(document.id, reason.trim() || 'Excluded during document review.'),
    onSuccess: done,
  });
  const move = useMutation({
    mutationFn: (bucketId: string) => api.moveDocument(document.id, bucketId),
    onSuccess: () => {
      done();
      onClose();
    },
  });
  const remove = useMutation({
    mutationFn: () => api.deleteDocument(document.id),
    onSuccess: () => {
      done();
      onClose();
    },
  });
  const busy = reject.isPending || move.isPending || remove.isPending;
  const destinations = (buckets.data?.items ?? []).filter(
    (bucket) => bucket.id !== document.bucket_id,
  );
  const validTarget = destinations.some((bucket) => bucket.id === target);
  const flags = Array.isArray(document.quality.flags)
    ? document.quality.flags.filter((flag): flag is string => typeof flag === 'string')
    : [];
  const error = reject.error ?? move.error ?? remove.error;
  return (
    <section
      className="document-inspector"
      aria-label={`Review ${document.filename}`}
      tabIndex={-1}
      ref={panel}
    >
      <Card>
        <CardHead
          title={document.filename}
          end={
            <>
              <Chip tone={STATUS_TONE[document.status] ?? 'idle'}>{document.status}</Chip>
              <button type="button" className="btn btn-sm" disabled={busy} onClick={onClose}>
                Close
              </button>
            </>
          }
        />
        <div className="card-body stack">
          {error ? <Notice tone="bad">{error.message}</Notice> : null}
          {document.error ? <Notice tone="warn">{document.error}</Notice> : null}
          {flags.length ? (
            <div className="quality-review">
              <h3 className="h2">Worth a look</h3>
              <ul>
                {flags.map((flag) => (
                  <li key={flag}>
                    <strong>{flagText(flag).label}</strong>
                    <span>{flagText(flag).why}</span>
                  </li>
                ))}
              </ul>
              <p className="hint">These are automatic hints, not a pass or fail.</p>
            </div>
          ) : null}
          <div className="inspector-grid">
            <div>
              {document.status !== 'failed' && document.status !== 'pending' ? (
                <TextPreview key={document.id} documentId={document.id} />
              ) : (
                <Empty
                  title="No cleaned text to show"
                  hint="Check the saved error and try a readable source file."
                />
              )}
            </div>
            <aside className="inspector-details">
              {document.source_note ? (
                <div>
                  <h3 className="h2">Source note</h3>
                  <p className="muted">{document.source_note}</p>
                </div>
              ) : null}
              <ExtractionDetails quality={document.quality} />
              <details className="disclosure">
                <summary>Text quality details</summary>
                <dl className="extraction-grid">
                  <div>
                    <dt className="label">Characters</dt>
                    <dd>{document.char_count.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt className="label">Words</dt>
                    <dd>{document.word_count.toLocaleString()}</dd>
                  </div>
                  {['alpha_ratio', 'mean_line_len', 'line_count'].map((key) => (
                    <div key={key}>
                      <dt className="label">
                        {key === 'alpha_ratio'
                          ? 'Share of letters'
                          : key === 'mean_line_len'
                            ? 'Average line length'
                            : 'Lines'}
                      </dt>
                      <dd>
                        {typeof document.quality[key] === 'number'
                          ? String(document.quality[key])
                          : 'Not recorded'}
                      </dd>
                    </div>
                  ))}
                </dl>
              </details>
            </aside>
          </div>
          <details className="disclosure document-actions">
            <summary>Manage this file</summary>
            <div className="stack">
              {buckets.error ? (
                <ErrorBox message={buckets.error.message} onRetry={() => void buckets.refetch()} />
              ) : (
                <div className="move-controls">
                  <label className="field">
                    <span className="label">Move to another collection</span>
                    <select
                      className="input"
                      value={validTarget ? target : ''}
                      disabled={busy || buckets.isPending || !destinations.length}
                      onChange={(event) => setTarget(event.target.value)}
                    >
                      <option value="">Choose a destination</option>
                      {destinations.map((bucket) => (
                        <option key={bucket.id} value={bucket.id}>
                          {bucket.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="btn"
                    disabled={!validTarget || busy}
                    onClick={() => move.mutate(target)}
                  >
                    {move.isPending ? 'Moving…' : 'Move file'}
                  </button>
                </div>
              )}
              {document.status !== 'rejected' ? (
                <div className="move-controls">
                  <label className="field">
                    <span className="label">Reason to reject (optional)</span>
                    <input
                      className="input"
                      maxLength={2000}
                      value={reason}
                      disabled={busy}
                      onChange={(event) => setReason(event.target.value)}
                      placeholder="Why this text should stay out"
                    />
                  </label>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() => reject.mutate()}
                  >
                    {reject.isPending ? 'Rejecting…' : 'Reject'}
                  </button>
                </div>
              ) : (
                <p className="hint">Rejected files stay out of future snapshots.</p>
              )}
              <div className="delete-file">
                <p className="hint">Delete removes this document and its stored files.</p>
                <ConfirmButton
                  label="Delete"
                  confirmLabel="Delete this file"
                  disabled={busy}
                  onConfirm={() => remove.mutate()}
                />
              </div>
            </div>
          </details>
        </div>
      </Card>
    </section>
  );
}
