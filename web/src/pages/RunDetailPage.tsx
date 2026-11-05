import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import { api, type MetricPoint, type Run } from '../api/client';
import ExportButton from '../components/ExportButton';
import LossChart from '../components/LossChart';
import { Card, CardHead, Chip, Empty, ErrorBox, Notice, type Tone } from '../components/ui';
import { bytesToSize, compactNumber, relativeDate } from '../lib/format';
import { STATUS_WORDS, duration, isActive, secondsLeft } from '../lib/training';

const STATUS_TONE: Record<string, Tone> = {
  completed: 'ok',
  running: 'warn',
  stopped: 'warn',
  failed: 'bad',
  pending: 'idle',
};

const POLL_MS = 2000;

function number(value: number | null | undefined, digits = 4): string {
  return value !== null && value !== undefined && Number.isFinite(value)
    ? value.toFixed(digits)
    : 'Not yet';
}

function nextAction(run: Run, saves: number): 'start' | 'stop' | 'resume' | null {
  if (isActive(run)) return 'stop';
  if (run.status === 'completed') return null;
  if (saves > 0) return 'resume';
  return 'start';
}

const ACTION_LABEL = { start: 'Start training', stop: 'Stop', resume: 'Resume' } as const;

export default function RunDetailPage() {
  const { runId = '' } = useParams();
  const queryClient = useQueryClient();
  const [measuring, setMeasuring] = useState(false);
  const live = (run: Run | undefined) =>
    run && (isActive(run) || (measuring && (run.bpb === null || run.bpb === undefined)))
      ? POLL_MS
      : false;

  const run = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.getRun(runId),
    enabled: Boolean(runId),
    refetchInterval: (query) => live(query.state.data),
  });
  const polling = live(run.data);

  const metrics = useQuery({
    queryKey: ['runMetrics', runId],
    queryFn: () => api.getRunMetrics(runId),
    enabled: Boolean(runId),
    refetchInterval: polling,
  });

  const saves = useQuery({
    queryKey: ['runCheckpoints', runId],
    queryFn: () => api.listCheckpoints(runId),
    enabled: Boolean(runId),
    refetchInterval: polling,
  });

  const change = useMutation({
    mutationFn: (action: 'start' | 'stop' | 'resume') => api.changeRun(runId, action),
    onSuccess: (updated) => {
      queryClient.setQueryData(['run', runId], updated);
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
      void queryClient.invalidateQueries({ queryKey: ['runCheckpoints', runId] });
    },
  });

  const computeBpbMutation = useMutation({
    mutationFn: () => api.computeRunBpb(runId),
    onSuccess: () => {
      setMeasuring(true);
      void queryClient.invalidateQueries({ queryKey: ['run', runId] });
    },
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
  const savePoints = saves.data ?? [];
  const action = nextAction(r, savePoints.length);
  const left = isActive(r) ? secondsLeft(r) : null;
  const stopping = isActive(r) && change.isSuccess && change.variables === 'stop';

  return (
    <>
      <p>
        <Link to="/history">Back to training history</Link>
      </p>

      <div className="page-intro">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h1 className="h1">{r.name}</h1>
          <Chip tone={STATUS_TONE[r.status] ?? 'idle'}>{STATUS_WORDS[r.status] ?? r.status}</Chip>
          {action ? (
            <button
              type="button"
              className={action === 'stop' ? 'btn' : 'btn btn-primary'}
              disabled={change.isPending || stopping}
              onClick={() => change.mutate(action)}
            >
              {stopping ? 'Stopping…' : ACTION_LABEL[action]}
            </button>
          ) : null}
        </div>
        <p className="lede">
          {r.model_name ?? 'Model'} on {r.build_name ?? r.dataset_name ?? 'no training set'}.
          Created {relativeDate(r.created_at)}
          {r.started_at ? `, started ${relativeDate(r.started_at)}` : ''}.
        </p>
      </div>

      {change.error ? <Notice tone="bad">{change.error.message}</Notice> : null}
      {r.error ? <Notice tone="bad">{r.error}</Notice> : null}

      <Card>
        <CardHead title="Progress" />
        <div className="card-body stack">
          <dl className="run-numbers">
            <div>
              <dt>Step</dt>
              <dd>
                {r.current_step.toLocaleString()} of {r.max_steps.toLocaleString()}
              </dd>
            </div>
            <div>
              <dt>Word pieces seen</dt>
              <dd>{compactNumber(r.tokens_seen)}</dd>
            </div>
            <div>
              <dt>Speed</dt>
              <dd>
                {r.tokens_per_sec ? `${compactNumber(r.tokens_per_sec)} per second` : 'Not yet'}
              </dd>
            </div>
            <div>
              <dt>Time left</dt>
              <dd>
                {left !== null ? `About ${duration(left)}` : isActive(r) ? 'Measuring' : 'None'}
              </dd>
            </div>
            <div>
              <dt>Training loss</dt>
              <dd>{number(r.loss)}</dd>
            </div>
            <div>
              <dt>Best validation loss</dt>
              <dd>{number(r.best_val_loss)}</dd>
            </div>
            <div>
              <dt>Bits per byte</dt>
              <dd>
                {r.bpb !== null && r.bpb !== undefined ? r.bpb.toFixed(4) : 'Not measured yet'}
              </dd>
            </div>
          </dl>
          {(r.bpb === null || r.bpb === undefined) && !isActive(r) && savePoints.length ? (
            <div className="row">
              <button
                type="button"
                className="btn btn-sm"
                disabled={computeBpbMutation.isPending}
                onClick={() => computeBpbMutation.mutate()}
              >
                {computeBpbMutation.isPending ? 'Measuring…' : 'Measure bits per byte'}
              </button>
              <span className="muted">
                Lower is better. It scores the held-out text, so it compares models with different
                vocabularies.
              </span>
            </div>
          ) : null}
          {computeBpbMutation.isError ? (
            <Notice tone="bad">{computeBpbMutation.error.message}</Notice>
          ) : null}
          {computeBpbMutation.isSuccess && (r.bpb === null || r.bpb === undefined) ? (
            <p className="muted">Measuring in the background. The number shows here when done.</p>
          ) : null}
          <progress
            className="grow"
            aria-label="Steps done"
            max={r.max_steps}
            value={r.current_step}
            style={{ width: '100%' }}
          />
        </div>
      </Card>

      <Card>
        <CardHead title="Loss" />
        <div className="card-body">
          {metrics.isPending ? (
            <p className="muted">Loading measurements…</p>
          ) : (
            <LossChart points={points} />
          )}
          <p className="muted" style={{ marginBottom: 0 }}>
            Lower is better. Validation loss is measured on text the model never trains on, so if it
            rises while training loss falls, the model is memorising.
          </p>
        </div>
      </Card>

      <Card>
        <CardHead title="Save points" />
        <div className="card-body">
          {!savePoints.length ? (
            <Empty
              title="No save points yet"
              hint={`One is kept every ${(r.settings?.save_every ?? 200).toLocaleString()} steps, at the end, and when you stop.`}
            />
          ) : (
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Step</th>
                    <th className="num">Training loss</th>
                    <th className="num">Validation loss</th>
                    <th className="num">Size</th>
                    <th>Saved</th>
                    <th>Download</th>
                    <th>Other tools</th>
                  </tr>
                </thead>
                <tbody>
                  {savePoints.map((save) => (
                    <tr key={save.id}>
                      <td>
                        {save.step.toLocaleString()}{' '}
                        {save.is_best ? <Chip tone="ok">Best so far</Chip> : null}
                      </td>
                      <td className="num">{number(save.loss)}</td>
                      <td className="num">{number(save.val_loss)}</td>
                      <td className="num">{bytesToSize(save.size_bytes)}</td>
                      <td className="muted">{relativeDate(save.created_at)}</td>
                      <td>
                        <a
                          className="btn btn-sm"
                          href={api.checkpointDownloadUrl(save.id)}
                          download
                          aria-label={`Download the save point at step ${save.step}`}
                        >
                          Download
                        </a>
                      </td>
                      <td>
                        <ExportButton checkpointId={save.id} step={save.step} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Card>

      <details className="disclosure card">
        <summary>
          Every measurement <span className="muted">({points.length.toLocaleString()})</span>
        </summary>
        <div className="scroll-x">
          <table className="table document-table">
            <thead>
              <tr>
                <th>Step</th>
                <th className="num">Loss</th>
                <th className="num">Validation loss</th>
                <th className="num">Learning rate</th>
                <th className="num">Word pieces per second</th>
              </tr>
            </thead>
            <tbody>
              {points.map((pt: MetricPoint) => (
                <tr key={pt.id}>
                  <td>{pt.step}</td>
                  <td className="num">{number(pt.loss)}</td>
                  <td className="num">{pt.val_loss !== null ? pt.val_loss.toFixed(4) : ''}</td>
                  <td className="num">
                    {pt.learning_rate !== null ? pt.learning_rate.toExponential(2) : ''}
                  </td>
                  <td className="num">{pt.speed !== null ? compactNumber(pt.speed) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}
