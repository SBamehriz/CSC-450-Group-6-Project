import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type Run, RUN_PAGE } from '../api/client';
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
import { compactNumber, plural, relativeDate } from '../lib/format';

const STATUS_TONE: Record<string, Tone> = {
  completed: 'ok',
  running: 'warn',
  stopped: 'warn',
  failed: 'bad',
  pending: 'idle',
};

function formatValLoss(val: number | null | undefined): string {
  if (val === null || val === undefined || !Number.isFinite(val)) {
    return '—'; // Clear empty indicator when validation loss is unavailable
  }
  return val.toFixed(4);
}

export default function TrainingHistoryPage() {
  const queryClient = useQueryClient();
  const [offset, setOffset] = useState(0);
  const [statusFilter, setStatusFilter] = useState('');
  const [explain, setExplain] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const runs = useQuery({
    queryKey: ['runs', { offset, status: statusFilter }],
    queryFn: () =>
      api.listRuns({
        status: statusFilter || undefined,
        offset,
        limit: RUN_PAGE,
      }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteRun(id),
    onSuccess: () => {
      setDeleteError(null);
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
    },
    onError: (err: Error) => {
      setDeleteError(err.message);
    },
  });

  const runItems = runs.data?.items ?? [];
  const total = runs.data?.total ?? 0;
  const currentPage = Math.floor(offset / RUN_PAGE) + 1;
  const totalPages = Math.max(1, Math.ceil(total / RUN_PAGE));

  return (
    <>
      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">Training History</h1>
          <p className="lede">
            Track and review small language model training runs. Inspect loss curves, validation
            performance, and training metrics over time.
          </p>
        </div>
        <ExplainToggle checked={explain} onChange={setExplain} />
      </div>

      {explain ? (
        <div className="help-panel">
          <p>
            <strong>Best validation loss</strong>: The lowest evaluation loss recorded across all
            checkpoints and evaluation steps for this run. A clear dash (—) indicates that
            validation loss was not evaluated or recorded.
          </p>
          <p>
            <strong>Deletion protection</strong>: Runs cannot be deleted if they are currently
            running or if dependent saved checkpoint results exist. Remove or export checkpoints
            first before deleting a run.
          </p>
          <p>
            <strong>Run link</strong>: Click on any run name to view its detailed step progression
            and recorded metric points.
          </p>
        </div>
      ) : null}

      <div className="collection-toolbar">
        <select
          className="input"
          aria-label="Filter runs by status"
          value={statusFilter}
          onChange={(event) => {
            setStatusFilter(event.target.value);
            setOffset(0);
          }}
        >
          <option value="">All statuses</option>
          <option value="completed">Completed</option>
          <option value="running">Running</option>
          <option value="stopped">Stopped</option>
          <option value="failed">Failed</option>
          <option value="pending">Pending</option>
        </select>

        <span className="muted">{plural(total, 'run')} recorded</span>
      </div>

      {deleteError ? (
        <Notice tone="bad">
          <strong>Deletion blocked:</strong> {deleteError}
        </Notice>
      ) : null}

      {runs.error ? (
        <ErrorBox message={runs.error.message} onRetry={() => void runs.refetch()} />
      ) : runs.isPending ? (
        <p className="muted" role="status">
          Loading training runs…
        </p>
      ) : !runItems.length ? (
        <Empty
          title="No training runs found"
          hint={
            statusFilter
              ? 'Try changing the status filter to see other runs.'
              : 'Start a training run to see measurements and history.'
          }
        />
      ) : (
        <Card>
          <CardHead
            title="Training runs"
            end={
              <span className="muted">
                Page {currentPage} of {totalPages}
              </span>
            }
          />
          <div className="scroll-x">
            <table className="table document-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Model</th>
                  <th>Training set</th>
                  <th>Status</th>
                  <th className="num">Best val loss</th>
                  <th className="num">Step</th>
                  <th>Date</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {runItems.map((run: Run) => (
                  <tr key={run.id}>
                    <td className="name">
                      <Link
                        to={`/runs/${run.id}`}
                        className="file-button"
                        style={{ fontWeight: 600 }}
                        aria-label={`View run details for ${run.name}`}
                      >
                        {run.name}
                      </Link>
                    </td>

                    <td>
                      {run.model_name ? (
                        <span>
                          {run.model_name}{' '}
                          {run.model_preset ? (
                            <span className="muted">({run.model_preset})</span>
                          ) : null}
                        </span>
                      ) : (
                        <span className="muted">Unassigned</span>
                      )}
                    </td>

                    <td>
                      {run.dataset_name ? (
                        <span>{run.dataset_name}</span>
                      ) : (
                        <span className="muted">None</span>
                      )}
                    </td>

                    <td>
                      <Chip tone={STATUS_TONE[run.status] ?? 'idle'}>{run.status}</Chip>
                    </td>

                    <td className="num">
                      {run.best_val_loss !== null && run.best_val_loss !== undefined ? (
                        <strong className="code">{formatValLoss(run.best_val_loss)}</strong>
                      ) : (
                        <span className="muted" aria-label="Validation loss unavailable">
                          —
                        </span>
                      )}
                    </td>

                    <td className="num">
                      {run.current_step.toLocaleString()} / {compactNumber(run.max_steps)}
                    </td>

                    <td className="muted">{relativeDate(run.created_at)}</td>

                    <td>
                      <ConfirmButton
                        label="Delete"
                        confirmLabel={`Delete ${run.name}`}
                        disabled={remove.isPending}
                        onConfirm={() => remove.mutate(run.id)}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="document-pager" style={{ padding: '16px 24px' }}>
            <span className="muted">
              Showing {offset + 1}–{Math.min(offset + RUN_PAGE, total)} of {total} runs
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="btn btn-sm"
                type="button"
                disabled={offset === 0 || runs.isPending}
                onClick={() => setOffset((prev) => Math.max(0, prev - RUN_PAGE))}
              >
                ← Previous
              </button>
              <button
                className="btn btn-sm"
                type="button"
                disabled={offset + RUN_PAGE >= total || runs.isPending}
                onClick={() => setOffset((prev) => prev + RUN_PAGE)}
              >
                Next →
              </button>
            </div>
          </div>
        </Card>
      )}
    </>
  );
}
