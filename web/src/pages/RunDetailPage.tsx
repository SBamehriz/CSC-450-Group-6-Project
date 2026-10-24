import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { api, type MetricPoint } from '../api/client';
import { Card, CardHead, Chip, ErrorBox, type Tone } from '../components/ui';
import { compactNumber, relativeDate } from '../lib/format';

const STATUS_TONE: Record<string, Tone> = {
  completed: 'ok',
  running: 'warn',
  stopped: 'warn',
  failed: 'bad',
  pending: 'idle',
};

export default function RunDetailPage() {
  const { runId = '' } = useParams();

  const run = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.getRun(runId),
    enabled: Boolean(runId),
  });

  const metrics = useQuery({
    queryKey: ['runMetrics', runId],
    queryFn: () => api.getRunMetrics(runId),
    enabled: Boolean(runId),
  });

  if (run.error) {
    return <ErrorBox message={run.error.message} onRetry={() => void run.refetch()} />;
  }

  if (run.isPending) {
    return (
      <p className="muted" role="status">
        Loading run details…
      </p>
    );
  }

  const r = run.data;
  const points = metrics.data?.items ?? [];

  return (
    <>
      <p>
        <Link to="/history">← Back to training history</Link>
      </p>

      <div className="page-intro">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <h1 className="h1">{r.name}</h1>
          <Chip tone={STATUS_TONE[r.status] ?? 'idle'}>{r.status}</Chip>
        </div>
        <p className="lede">
          Started {r.started_at ? relativeDate(r.started_at) : 'pending'} · Created{' '}
          {relativeDate(r.created_at)}
        </p>
      </div>

      <div className="preparation-grid">
        <Card>
          <CardHead title="Run configuration" />
          <div className="card-body stack">
            <dl className="preview-counts">
              <div>
                <dt>Model</dt>
                <dd>{r.model_name || r.model_id}</dd>
              </div>
              <div>
                <dt>Training set</dt>
                <dd>{r.dataset_name || (r.dataset_id ? r.dataset_id : 'None')}</dd>
              </div>
              <div>
                <dt>Device</dt>
                <dd>{r.device.toUpperCase()}</dd>
              </div>
              <div>
                <dt>Learning rate</dt>
                <dd>{r.learning_rate}</dd>
              </div>
              <div>
                <dt>Batch size</dt>
                <dd>{r.batch_size} sequences</dd>
              </div>
              <div>
                <dt>Steps</dt>
                <dd>
                  {r.current_step.toLocaleString()} / {compactNumber(r.max_steps)}
                </dd>
              </div>
            </dl>
          </div>
        </Card>

        <Card>
          <CardHead title="Run metrics" />
          <div className="card-body stack">
            <dl className="preview-counts">
              <div>
                <dt>Current loss</dt>
                <dd>{r.loss !== null && r.loss !== undefined ? r.loss.toFixed(4) : '—'}</dd>
              </div>
              <div>
                <dt>Best val loss</dt>
                <dd>
                  {r.best_val_loss !== null && r.best_val_loss !== undefined
                    ? r.best_val_loss.toFixed(4)
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>Training speed</dt>
                <dd>{r.tokens_per_sec ? `${compactNumber(r.tokens_per_sec)} tok/s` : '—'}</dd>
              </div>
              <div>
                <dt>Metric points</dt>
                <dd>{points.length.toLocaleString()}</dd>
              </div>
            </dl>

            {r.error ? (
              <p className="notice notice-bad" role="alert">
                {r.error}
              </p>
            ) : null}
          </div>
        </Card>
      </div>

      <Card>
        <CardHead title="Recorded metric points" />
        <div className="card-body">
          {metrics.isPending ? (
            <p className="muted">Loading measurements…</p>
          ) : !points.length ? (
            <p className="muted">No metric points recorded yet for this run.</p>
          ) : (
            <div className="scroll-x">
              <table className="table document-table">
                <thead>
                  <tr>
                    <th>Step</th>
                    <th className="num">Loss</th>
                    <th className="num">Val Loss</th>
                    <th className="num">Learning Rate</th>
                    <th className="num">Speed (tok/s)</th>
                    <th>Recorded</th>
                  </tr>
                </thead>
                <tbody>
                  {points.map((pt: MetricPoint) => (
                    <tr key={pt.id}>
                      <td className="code">{pt.step}</td>
                      <td className="num">{pt.loss !== null ? pt.loss.toFixed(4) : '—'}</td>
                      <td className="num">{pt.val_loss !== null ? pt.val_loss.toFixed(4) : '—'}</td>
                      <td className="num">
                        {pt.learning_rate !== null ? pt.learning_rate.toExponential(2) : '—'}
                      </td>
                      <td className="num">{pt.speed !== null ? compactNumber(pt.speed) : '—'}</td>
                      <td className="muted">{relativeDate(pt.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Card>
    </>
  );
}
