import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { api, type EvalResult, type Model } from '../api/client';
import HubImport from '../components/HubImport';
import {
  Card,
  CardHead,
  Chip,
  ConfirmButton,
  Empty,
  ErrorBox,
  ExplainToggle,
  Notice,
} from '../components/ui';
import { bytesToSize, compactNumber, relativeDate } from '../lib/format';

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const models = useQuery({
    queryKey: ['models'],
    queryFn: api.listAllModels,
  });

  const [explain, setExplain] = useState(false);
  const [search, setSearch] = useState('');
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [selectedHistoryModelId, setSelectedHistoryModelId] = useState('');
  const [historyTypeFilter, setHistoryTypeFilter] = useState<
    'all' | 'prompt' | 'bpb' | 'benchmark'
  >('all');
  const [expandedResultIds, setExpandedResultIds] = useState<Record<string, boolean>>({});
  const [deleteResultError, setDeleteResultError] = useState<string | null>(null);

  const evalResults = useQuery({
    queryKey: ['evalResults', { model_id: selectedHistoryModelId || undefined }],
    queryFn: () =>
      typeof api.listEvalResults === 'function'
        ? api
            .listEvalResults({ model_id: selectedHistoryModelId || undefined })
            .catch(() => [] as EvalResult[])
        : Promise.resolve([] as EvalResult[]),
  });

  const removeResult = useMutation({
    mutationFn: (id: string) => api.deleteEvalResult(id),
    onSuccess: () => {
      setDeleteResultError(null);
      setDeleteError(null);
      void queryClient.invalidateQueries({ queryKey: ['evalResults'] });
      void queryClient.invalidateQueries({ queryKey: ['evaluations'] });
      void queryClient.invalidateQueries({ queryKey: ['bpbEvaluations'] });
    },
    onError: (err: Error) => {
      setDeleteResultError(err.message);
    },
  });

  const toggleResultExpand = (id: string) => {
    setExpandedResultIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteModel(id),
    onSuccess: () => {
      setDeleteError(null);
      void queryClient.invalidateQueries({ queryKey: ['models'] });
    },
    onError: (err: Error) => {
      setDeleteError(err.message);
    },
  });

  const savedModels = models.data?.items ?? [];
  const visibleModels = underParents(savedModels).filter((m: Model) =>
    `${m.name} ${m.description} ${m.preset ?? ''}`
      .toLowerCase()
      .includes(search.trim().toLowerCase()),
  );

  return (
    <>
      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">Models</h1>
          <p className="lede">
            Every model you trained, fine-tuned or brought in from Hugging Face. Make a new one on
            the <Link to="/train">Train</Link> page.
          </p>
        </div>
        <ExplainToggle checked={explain} onChange={setExplain} />
      </div>

      {explain ? (
        <div className="help-panel">
          <p>
            <strong>Blocks (layers)</strong>: Number of transformer blocks.
          </p>
          <p>
            <strong>Width (d_model)</strong>: Hidden vector size. Must divide evenly by attention
            heads.
          </p>
          <p>
            <strong>Vocabulary</strong>: Made from the text a model trains on. Token count directly
            sizes the embedding table and prediction head.
          </p>
          <p>
            <strong>Memory figures are estimates</strong>: Calculated using ~16 bytes/parameter
            (weights, gradients, and Adam optimizer states) plus activation memory for a batch of 8
            sequences. Figures are estimates based on architectural formulas.
          </p>
        </div>
      ) : null}

      <HubImport />

      {/* Saved Models List Card (SP8.9) */}
      <Card>
        <CardHead
          title="Saved models"
          end={
            <input
              type="search"
              className="input input-sm"
              placeholder="Filter models…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ maxWidth: 220 }}
            />
          }
        />
        <div className="card-body">
          {deleteError ? (
            <div style={{ marginBottom: 12 }}>
              <Notice tone="bad">{deleteError}</Notice>
            </div>
          ) : null}

          {models.error ? (
            <ErrorBox message={models.error.message} onRetry={() => void models.refetch()} />
          ) : models.isPending ? (
            <p role="status">Loading saved models…</p>
          ) : visibleModels.length === 0 ? (
            <Empty
              title={search ? 'No matching models' : 'No saved models yet'}
              hint={
                search
                  ? 'Try adjusting your search filter.'
                  : 'Train one on the Train page, or import one from Hugging Face above.'
              }
            />
          ) : (
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Source</th>
                    <th>Vocabulary</th>
                    <th className="num">Parameters</th>
                    <th className="num">Est. Memory</th>
                    <th>Shape</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleModels.map((m: Model) => {
                    const hub = hubFacts(m);
                    const vocabName = hub
                      ? 'Its own'
                      : typeof m.config?.dataset_name === 'string'
                        ? m.config.dataset_name
                        : 'Default vocab';
                    return (
                      <tr key={m.id} className={m.parent_id ? 'child-row' : undefined}>
                        <td>
                          <strong>{m.name}</strong>
                          {m.chat ? <span className="tag">chat</span> : null}{' '}
                          <Chip tone={m.ready ? 'ok' : 'idle'}>
                            {m.ready ? 'Ready' : 'Not trained yet'}
                          </Chip>
                          {m.parent_id ? (
                            <p className="muted" style={{ margin: '2px 0 0', fontSize: '0.85em' }}>
                              Fine-tuned from {m.parent_name ?? 'another model'}
                            </p>
                          ) : null}
                          {m.description ? (
                            <p className="muted" style={{ margin: '2px 0 0', fontSize: '0.85em' }}>
                              {m.description}
                            </p>
                          ) : null}
                          {hub ? (
                            <p className="muted" style={{ margin: '2px 0 0', fontSize: '0.85em' }}>
                              Licence {hub.license}, version {hub.commit}
                            </p>
                          ) : null}
                        </td>
                        <td>
                          {hub ? (
                            <Chip tone="idle">Imported</Chip>
                          ) : (
                            <Chip tone="idle">Made here</Chip>
                          )}
                        </td>
                        <td>
                          <span title={`Vocab size: ${m.vocab_size.toLocaleString()}`}>
                            {vocabName}
                          </span>
                        </td>
                        <td className="num">{compactNumber(m.param_count)}</td>
                        <td className="num">
                          {m.estimated_memory_bytes
                            ? bytesToSize(m.estimated_memory_bytes)
                            : 'None'}
                        </td>
                        <td>
                          {m.n_layer}L · {m.d_model}W · {m.n_head}H · {m.ctx_len}C
                        </td>
                        <td>{relativeDate(m.created_at)}</td>
                        <td>
                          <div
                            style={{
                              display: 'flex',
                              flexWrap: 'wrap',
                              gap: 6,
                              alignItems: 'center',
                            }}
                          >
                            {m.ready ? (
                              <Link className="btn btn-sm" to={chatLink(m)}>
                                Chat
                              </Link>
                            ) : null}
                            {m.ready && m.source === 'forge' ? (
                              <Link className="btn btn-sm" to={`/train?from=${m.id}`}>
                                Fine-tune
                              </Link>
                            ) : null}
                            <Link className="btn btn-sm" to={`/compare?model=${m.id}`}>
                              Compare
                            </Link>
                            <button
                              type="button"
                              className="btn btn-sm"
                              onClick={() => setSelectedHistoryModelId(m.id)}
                            >
                              History
                            </button>
                            <Link className="btn btn-sm" to={`/compare/measure?model=${m.id}`}>
                              Evaluate
                            </Link>
                            <ConfirmButton
                              label="Delete"
                              confirmLabel="Delete model"
                              onConfirm={() => remove.mutate(m.id)}
                              disabled={remove.isPending}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Card>

      {/* Model Evaluation History Card */}
      <Card>
        <CardHead
          title="Model evaluation history"
          end={
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <select
                className="input input-sm"
                aria-label="Filter history by model"
                value={selectedHistoryModelId}
                onChange={(e) => setSelectedHistoryModelId(e.target.value)}
                style={{ minWidth: 180 }}
              >
                <option value="">All models</option>
                {savedModels.map((m: Model) => (
                  <option key={m.id} value={m.id}>
                    Model: {m.name}
                  </option>
                ))}
              </select>
              <select
                className="input input-sm"
                aria-label="Filter history by result type"
                value={historyTypeFilter}
                onChange={(e) =>
                  setHistoryTypeFilter(e.target.value as 'all' | 'prompt' | 'bpb' | 'benchmark')
                }
                style={{ minWidth: 150 }}
              >
                <option value="all">All result types</option>
                <option value="prompt">Prompt answers</option>
                <option value="bpb">Bits per byte (BPB)</option>
                <option value="benchmark">Benchmarks</option>
              </select>
              <Link
                className="btn btn-sm"
                to={
                  selectedHistoryModelId ? `/compare?model=${selectedHistoryModelId}` : '/compare'
                }
              >
                Open Compare
              </Link>
            </div>
          }
        />
        <div className="card-body stack">
          {deleteResultError ? <Notice tone="bad">{deleteResultError}</Notice> : null}
          {evalResults.error ? (
            <ErrorBox
              message={evalResults.error.message}
              onRetry={() => void evalResults.refetch()}
            />
          ) : evalResults.isPending ? (
            <p role="status">Loading saved evaluation results…</p>
          ) : (
            (() => {
              const allItems: EvalResult[] = evalResults.data ?? [];
              const filtered =
                historyTypeFilter === 'all'
                  ? allItems
                  : allItems.filter((r) => r.result_type === historyTypeFilter);

              if (filtered.length === 0) {
                return (
                  <Empty
                    title="No saved evaluation results yet"
                    hint="Run a prompt set evaluation, bits-per-byte check, or benchmark to record history for this model."
                  />
                );
              }

              return (
                <div className="stack">
                  {filtered.map((r) => {
                    const isExpanded = Boolean(expandedResultIds[r.id]);
                    const isLongPrompt = (r.prompt?.length ?? 0) > 120;
                    const isLongAnswer = (r.answer?.length ?? 0) > 180;
                    const needsToggle =
                      r.result_type === 'prompt' && (isLongPrompt || isLongAnswer);

                    const displayPrompt =
                      !isExpanded && isLongPrompt
                        ? `${r.prompt?.slice(0, 120)}…`
                        : (r.prompt ?? '');
                    const displayAnswer =
                      !isExpanded && isLongAnswer
                        ? `${r.answer?.slice(0, 180)}…`
                        : (r.answer ?? '');

                    const identifier =
                      r.result_type === 'benchmark'
                        ? (r.benchmark_name ?? 'Benchmark')
                        : r.result_type === 'bpb'
                          ? ((r.settings?.dataset_name as string | undefined) ??
                            r.benchmark_name ??
                            'Dataset snapshot')
                          : ((r.settings?.prompt_set_name as string | undefined) ?? 'Prompt');

                    const checkpointText = r.checkpoint_id
                      ? `Checkpoint ${r.checkpoint_id.slice(0, 8)}`
                      : typeof r.settings?.checkpoint_step === 'number'
                        ? `Step ${r.settings.checkpoint_step}`
                        : 'Base / latest weights';

                    const settingsSummary = Object.entries(r.settings ?? {})
                      .filter(
                        ([k, v]) =>
                          v !== null &&
                          v !== undefined &&
                          !['prompt_set_name', 'dataset_name', 'evaluation_name'].includes(k),
                      )
                      .map(([k, v]) => `${k}: ${String(v)}`)
                      .join(' · ');

                    return (
                      <div key={r.id} className="subcard stack">
                        <div className="subcard-head">
                          <div
                            style={{
                              display: 'flex',
                              gap: 8,
                              alignItems: 'center',
                              flexWrap: 'wrap',
                            }}
                          >
                            <Chip
                              tone={
                                r.result_type === 'bpb'
                                  ? 'ok'
                                  : r.result_type === 'benchmark'
                                    ? 'warn'
                                    : 'idle'
                              }
                            >
                              {r.result_type === 'bpb'
                                ? 'Bits-per-byte'
                                : r.result_type === 'benchmark'
                                  ? 'Benchmark'
                                  : 'Prompt answer'}
                            </Chip>
                            <strong>{r.model_name}</strong>
                            <span className="muted" style={{ fontSize: '0.85em' }}>
                              ({checkpointText})
                            </span>
                            <span className="badge">{identifier}</span>
                          </div>
                          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                            <span className="muted" style={{ fontSize: '0.85em' }}>
                              {relativeDate(r.created_at)}
                            </span>
                            <Link className="btn btn-sm" to={`/compare?model=${r.model_id ?? ''}`}>
                              Compare result →
                            </Link>
                            <ConfirmButton
                              label="Delete"
                              confirmLabel="Delete result"
                              onConfirm={() => removeResult.mutate(r.id)}
                              disabled={removeResult.isPending}
                            />
                          </div>
                        </div>

                        {r.result_type === 'benchmark' ? (
                          <div style={{ display: 'flex', gap: 16, alignItems: 'baseline' }}>
                            <div>
                              <span className="muted" style={{ fontSize: '0.85em' }}>
                                Score:{' '}
                              </span>
                              <strong className="mono" style={{ fontSize: '1.05em' }}>
                                {r.score !== null && r.score !== undefined
                                  ? r.score.toFixed(4)
                                  : 'None'}
                              </strong>
                            </div>
                          </div>
                        ) : r.result_type === 'bpb' ? (
                          <div
                            style={{
                              display: 'flex',
                              gap: 16,
                              alignItems: 'baseline',
                              flexWrap: 'wrap',
                            }}
                          >
                            <div>
                              <span className="muted" style={{ fontSize: '0.85em' }}>
                                Bits per byte:{' '}
                              </span>
                              <strong className="mono" style={{ fontSize: '1.05em' }}>
                                {r.bpb !== null ? r.bpb.toFixed(4) : '—'}
                              </strong>
                            </div>
                            {r.total_nll !== null && r.scored_tokens ? (
                              <div>
                                <span className="muted" style={{ fontSize: '0.85em' }}>
                                  Val loss (nats):{' '}
                                </span>
                                <span className="mono">
                                  {(r.total_nll / r.scored_tokens).toFixed(4)}
                                </span>
                              </div>
                            ) : null}
                          </div>
                        ) : (
                          <div className="stack" style={{ gap: 6 }}>
                            <div>
                              <span className="muted" style={{ fontSize: '0.82em' }}>
                                Prompt:{' '}
                              </span>
                              <span style={{ fontWeight: 500 }}>{displayPrompt}</span>
                            </div>
                            <div
                              style={{
                                background: 'var(--bg-subtle, rgba(0,0,0,0.03))',
                                padding: '8px 10px',
                                borderRadius: 6,
                                fontFamily: 'var(--font-mono, monospace)',
                                fontSize: '0.88em',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                              }}
                            >
                              {displayAnswer}
                            </div>
                            {needsToggle ? (
                              <div>
                                <button
                                  type="button"
                                  className="btn btn-sm"
                                  onClick={() => toggleResultExpand(r.id)}
                                >
                                  {isExpanded ? 'Show less' : 'Show full prompt & answer'}
                                </button>
                              </div>
                            ) : null}
                          </div>
                        )}

                        <div
                          className="muted"
                          style={{
                            display: 'flex',
                            gap: 12,
                            flexWrap: 'wrap',
                            fontSize: '0.8em',
                          }}
                        >
                          <span>
                            Settings: <code>{settingsSummary || 'default'}</code>
                          </span>
                          <span>
                            Code version: <code>{r.code_version || 'Unavailable'}</code>
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()
          )}
        </div>
      </Card>
    </>
  );
}

// where an imported model came from, with its pinned version cut short
function hubFacts(model: Model): { license: string; commit: string } | null {
  if (model.source !== 'hub') return null;
  const hub = (model.config?.hub ?? {}) as { license?: string; commit?: string };
  return { license: hub.license ?? 'unknown', commit: (hub.commit ?? 'unknown').slice(0, 7) };
}

// each fine-tune right under the model it came from
function underParents(models: Model[]): Model[] {
  const ids = new Set(models.map((m) => m.id));
  const tops = models.filter((m) => !m.parent_id || !ids.has(m.parent_id));
  const placed: Model[] = [];
  const add = (m: Model) => {
    placed.push(m);
    for (const child of models) if (child.parent_id === m.id) add(child);
  };
  tops.forEach(add);
  return placed;
}

// a fine-tune opens beside the model it came from
function chatLink(m: Model): string {
  return m.parent_id ? `/chat?model=${m.id}&with=${m.parent_id}` : `/chat?model=${m.id}`;
}
