import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import { api, type Document, type UploadResult } from '../api/client';
import {
  Card,
  CardHead,
  Chip,
  Dropzone,
  Empty,
  ErrorBox,
  Notice,
  type Tone,
} from '../components/ui';
import { compactNumber, plural } from '../lib/format';

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
  const [filter, setFilter] = useState<FilterKey>('all');
  const [search, setSearch] = useState('');
  const [openId, setOpenId] = useState<string | null>(null);

  const bucket = useQuery({
    queryKey: ['bucket', bucketId],
    queryFn: () => api.getBucket(bucketId),
  });

  const documents = useQuery({
    queryKey: ['documents', bucketId, filter, search],
    queryFn: () =>
      api.listDocuments({
        bucketId,
        status: filter === 'all' || filter === 'flagged' ? undefined : filter,
        flagged: filter === 'flagged',
        q: search || undefined,
      }),
  });

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['documents', bucketId] });
    void queryClient.invalidateQueries({ queryKey: ['bucket', bucketId] });
    void queryClient.invalidateQueries({ queryKey: ['buckets'] });
  };

  if (bucket.isPending) return <p className="muted">Loading bucket.</p>;
  if (bucket.error) {
    return <ErrorBox message={bucket.error.message} onRetry={() => void bucket.refetch()} />;
  }

  const stats = bucket.data.stats;
  const open = documents.data?.items.find((document) => document.id === openId) ?? null;

  return (
    <>
      <div className="crumb">
        <Link to="/data">Data</Link>
        <span>,</span>
        <span>{bucket.data.name}</span>
      </div>

      <div>
        <h1 className="h1">{bucket.data.name}</h1>
        <p className="lede">
          {bucket.data.description || 'No source note yet.'}, {plural(stats.documents, 'document')},{' '}
          {compactNumber(stats.chars)} characters, about {compactNumber(stats.est_tokens)} tokens.
        </p>
      </div>

      <UploadPanel bucketId={bucketId} onUploaded={refresh} />

      <Card>
        <CardHead
          title={plural(documents.data?.total ?? 0, 'document')}
          end={
            <>
              <div className="seg">
                {FILTERS.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    className={filter === option.key ? 'on' : undefined}
                    onClick={() => setFilter(option.key)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <input
                className="input"
                style={{ width: 180, height: 25 }}
                placeholder="Filter by filename"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </>
          }
        />
        <DocumentTable
          isPending={documents.isPending}
          error={documents.error}
          documents={documents.data?.items ?? []}
          openId={openId}
          onOpen={(id) => setOpenId(id === openId ? null : id)}
          onRetry={() => void documents.refetch()}
        />
      </Card>

      {open ? (
        <DocumentPanel document={open} onClose={() => setOpenId(null)} onChanged={refresh} />
      ) : null}
    </>
  );
}

function UploadPanel({ bucketId, onUploaded }: { bucketId: string; onUploaded: () => void }) {
  const [sourceNote, setSourceNote] = useState('');
  const [result, setResult] = useState<UploadResult | null>(null);

  const upload = useMutation({
    mutationFn: (files: File[]) => api.uploadDocuments(bucketId, files, sourceNote),
    onSuccess: (data) => {
      setResult(data);
      onUploaded();
    },
  });

  return (
    <Card>
      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <label className="field">
          <span className="label">Where these files came from</span>
          <input
            className="input"
            value={sourceNote}
            placeholder="CourtListener public records, downloaded August 2026"
            maxLength={2000}
            onChange={(event) => setSourceNote(event.target.value)}
          />
        </label>

        <Dropzone onFiles={(files) => upload.mutate(files)} busy={upload.isPending} />
        <p className="muted" style={{ margin: '-4px 0 0', fontSize: 11.5 }}>
          Up to 50 MB each, and 200 files at a time. Each file is parsed when it arrives.
        </p>

        {upload.error ? <Notice tone="bad">{upload.error.message}</Notice> : null}
        {result ? <UploadReport result={result} /> : null}
      </div>
    </Card>
  );
}

function UploadReport({ result }: { result: UploadResult }) {
  return (
    <div>
      <p style={{ margin: '0 0 8px' }}>
        <Chip tone={result.failed ? 'warn' : 'ok'}>
          {result.parsed} parsed, {result.failed} failed
        </Chip>
      </p>
      {result.failed > 0 ? (
        <table className="table">
          <tbody>
            {result.outcomes
              .filter((outcome) => outcome.status === 'failed')
              .map((outcome) => (
                <tr key={outcome.filename}>
                  <td className="name" style={{ width: 220 }}>
                    {outcome.filename}
                  </td>
                  <td className="muted">{outcome.error}</td>
                </tr>
              ))}
          </tbody>
        </table>
      ) : null}
    </div>
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
  if (isPending) return <div className="card-body muted">Loading documents.</div>;
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
          hint="Add some files above. Failed files will still appear with the reason."
        />
      </div>
    );
  }

  return (
    <div className="scroll-x">
      <table className="table">
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
              onClick={() => onOpen(document.id)}
              style={{
                cursor: 'pointer',
                background: openId === document.id ? 'var(--page)' : undefined,
              }}
            >
              <td className="name">{document.filename}</td>
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
                {document.char_count ? compactNumber(document.char_count) : 'None'}
              </td>
              <td className="num">
                {document.word_count ? compactNumber(document.word_count) : 'None'}
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
          {name.replace(/_/g, ' ')}
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
  const buckets = useQuery({ queryKey: ['buckets'], queryFn: api.listBuckets });
  const text = useQuery({
    queryKey: ['documentText', document.id],
    queryFn: () => api.documentText(document.id),
    enabled: document.status !== 'failed',
  });

  const done = () => {
    void queryClient.invalidateQueries({ queryKey: ['documentText', document.id] });
    onChanged();
  };

  const reject = useMutation({
    mutationFn: () => api.rejectDocument(document.id, 'Rejected by hand from the inspector.'),
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

  const stats = document.quality as Record<string, number | string[]>;
  const flags = (stats.flags as string[]) ?? [];
  const error = reject.error ?? move.error ?? remove.error;

  return (
    <Card>
      <CardHead
        title={document.filename}
        end={
          <button type="button" className="btn btn-sm" onClick={onClose}>
            Close
          </button>
        }
      />
      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {error ? <Notice tone="bad">{error.message}</Notice> : null}

        {document.status === 'failed' ? (
          <Notice tone="warn">{document.error}</Notice>
        ) : flags.length > 0 ? (
          <Notice tone="warn">
            Flagged <strong>{flags.join(', ').replace(/_/g, ' ')}</strong>. Check the file before
            using it.
          </Notice>
        ) : null}

        {document.source_note ? (
          <div>
            <div className="label">Source</div>
            <div className="muted">{document.source_note}</div>
          </div>
        ) : null}

        {document.status !== 'failed' ? (
          <div>
            <div className="label" style={{ marginBottom: 5 }}>
              Cleaned text
              {text.data ? (
                <span className="muted" style={{ textTransform: 'none', letterSpacing: 0 }}>
                  {', first '}
                  {text.data.length.toLocaleString()} of {text.data.total.toLocaleString()}{' '}
                  characters.
                </span>
              ) : null}
            </div>
            <pre className="preview">
              {text.isPending ? 'Loading.' : (text.error?.message ?? text.data?.text)}
            </pre>
          </div>
        ) : null}

        <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap' }}>
          <Stat label="characters" value={document.char_count.toLocaleString()} />
          <Stat label="words" value={document.word_count.toLocaleString()} />
          <Stat label="alpha ratio" value={String(stats.alpha_ratio ?? 'None')} />
          <Stat label="mean line length" value={String(stats.mean_line_len ?? 'None')} />
          <Stat label="lines" value={String(stats.line_count ?? 'None')} />
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <select
            className="input"
            style={{ width: 200 }}
            value=""
            onChange={(event) => event.target.value && move.mutate(event.target.value)}
          >
            <option value="">Move to another bucket.</option>
            {(buckets.data?.items ?? [])
              .filter((candidate) => candidate.id !== document.bucket_id)
              .map((candidate) => (
                <option key={candidate.id} value={candidate.id}>
                  {candidate.name}
                </option>
              ))}
          </select>
          {document.status !== 'rejected' ? (
            <button
              type="button"
              className="btn"
              disabled={reject.isPending}
              onClick={() => reject.mutate()}
            >
              Reject
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-danger"
            disabled={remove.isPending}
            onClick={() => remove.mutate()}
          >
            Delete
          </button>
        </div>
      </div>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat-value">{value}</span>
      <span className="label">{label}</span>
    </div>
  );
}