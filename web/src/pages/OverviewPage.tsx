import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type Bucket } from '../api/client';
import UploadPanel from '../components/UploadPanel';
import DataSteps from '../components/DataSteps';
import { Card, CardHead, Chip, Empty, ErrorBox } from '../components/ui';
import { compactNumber, relativeDate } from '../lib/format';

export default function OverviewPage() {
  const stats = useQuery({
    queryKey: ['overview'],
    queryFn: api.overviewStats,
    refetchInterval: 5000,
  });
  const recent = useQuery({
    queryKey: ['recentUploads'],
    queryFn: api.recentUploads,
    refetchInterval: 5000,
  });
  const buckets = useQuery({
    queryKey: ['buckets'],
    queryFn: api.listBuckets,
    refetchInterval: 5000,
  });

  return (
    <>
      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">Overview</h1>
          <p className="lede">
            Your text, ready for the next step. Upload files, check what was extracted, and save a
            dataset.
          </p>
        </div>
        <Link className="btn" to="/data">
          Manage collections →
        </Link>
      </div>
      {stats.error ? (
        <ErrorBox message={stats.error.message} onRetry={() => void stats.refetch()} />
      ) : stats.isPending ? (
        <p className="muted" role="status">
          Loading library statistics…
        </p>
      ) : (
        <>
          <div className="overview-metrics">
            <Metric label="Buckets" value={stats.data.buckets} />
            <Metric label="Documents" value={stats.data.documents} />
            <Metric label="Parsed" value={stats.data.parsed_documents} />
            <Metric label="Flagged for review" value={stats.data.flagged_documents} />
            <Metric label="Failed" value={stats.data.failed_documents} />
            <Metric label="Ready snapshots" value={stats.data.ready_snapshots} />
          </div>
          <p className="muted">
            {compactNumber(stats.data.chars)} parsed characters · {compactNumber(stats.data.words)}{' '}
            words · about {compactNumber(stats.data.est_tokens)} tokens. Token counts here are
            estimates.
          </p>
          {buckets.data && !buckets.error ? (
            <DataSteps buckets={buckets.data.items} stats={stats.data} />
          ) : null}
        </>
      )}
      {buckets.error ? (
        <ErrorBox message={buckets.error.message} onRetry={() => void buckets.refetch()} />
      ) : buckets.isPending ? (
        <p className="muted" role="status">
          Loading buckets…
        </p>
      ) : buckets.data.items.length ? (
        <AddData buckets={buckets.data.items} />
      ) : (
        <Card>
          <div className="card-body stack">
            <h2 className="h2">Start with your data</h2>
            <p className="muted">Create a bucket for one source of text, then add files.</p>
            <div>
              <Link className="btn btn-primary" to="/data">
                Create a collection
              </Link>
            </div>
          </div>
        </Card>
      )}
      <Card>
        <CardHead
          title="Recent uploads"
          end={
            <button
              className="btn btn-sm"
              type="button"
              onClick={() => {
                void stats.refetch();
                void recent.refetch();
                void buckets.refetch();
              }}
            >
              Refresh
            </button>
          }
        />
        <div className="card-body">
          {recent.error ? (
            <ErrorBox message={recent.error.message} onRetry={() => void recent.refetch()} />
          ) : recent.isPending ? (
            <p className="muted" role="status">
              Loading recent uploads…
            </p>
          ) : !recent.data.items.length ? (
            <Empty title="No uploads yet" hint="Your latest documents will show up here." />
          ) : (
            <ul className="recent-uploads">
              {recent.data.items.map((document) => (
                <li key={document.id}>
                  <div className="recent-file">
                    <Link
                      to={`/data/${document.bucket_id}?open=${encodeURIComponent(document.id)}`}
                    >
                      {document.filename}
                    </Link>
                    <span className="muted">
                      {document.bucket_name} · {relativeDate(document.created_at)}
                    </span>
                  </div>
                  <Chip
                    tone={
                      document.status === 'parsed'
                        ? 'ok'
                        : document.status === 'failed'
                          ? 'bad'
                          : 'idle'
                    }
                  >
                    {document.status}
                  </Chip>
                  {document.error ? <p className="muted recent-error">{document.error}</p> : null}
                </li>
              ))}
            </ul>
          )}
          <p className="muted activity-note">
            Latest document records with their current bucket and status. Deleted records leave this
            list.
          </p>
        </div>
      </Card>
    </>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="card overview-metric">
      <span className="stat-value">{value.toLocaleString()}</span>
      <span className="label">{label}</span>
    </div>
  );
}

function AddData({ buckets }: { buckets: Bucket[] }) {
  const [selectedId, setSelectedId] = useState('');
  const [busy, setBusy] = useState(false);
  const bucket = buckets.find((item) => item.id === selectedId) ?? buckets[0];
  return (
    <div className="stack">
      <div className="upload-destination">
        <label className="field overview-bucket">
          <span className="label">Upload destination</span>
          <select
            className="input"
            value={bucket.id}
            disabled={busy}
            onChange={(event) => setSelectedId(event.target.value)}
          >
            {buckets.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <Link to={`/data/${bucket.id}`}>Review this collection →</Link>
      </div>
      <UploadPanel key={bucket.id} bucketId={bucket.id} onBusyChange={setBusy} />
    </div>
  );
}
