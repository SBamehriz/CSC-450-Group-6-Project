import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import { api, type EvalResult, type Model } from '../api/client';
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
import { relativeDate } from '../lib/format';

const TYPE_TONE: Record<string, Tone> = {
  prompt: 'ok',
  bpb: 'warn',
  benchmark: 'idle',
};

function formatSettings(settings: Record<string, unknown> | undefined): string {
  if (!settings || Object.keys(settings).length === 0) return 'Default settings';
  return Object.entries(settings)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k}: ${String(v)}`)
    .join(' · ');
}

function resultSummary(r: EvalResult): string {
  if (r.result_type === 'bpb') {
    return r.bpb !== null && r.bpb !== undefined ? `${r.bpb.toFixed(4)} bits/byte` : 'No score';
  }
  if (r.result_type === 'benchmark') {
    const name = r.benchmark_name || 'Benchmark';
    const score =
      r.benchmark_score !== null && r.benchmark_score !== undefined
        ? String(r.benchmark_score)
        : '—';
    return `${name}: ${score}`;
  }
  if (r.result_type === 'prompt') {
    const p = r.prompt ? (r.prompt.length > 50 ? `${r.prompt.slice(0, 50)}…` : r.prompt) : 'Prompt';
    return `"${p}"`;
  }
  return 'Saved result';
}

export default function ComparePage() {
  const queryClient = useQueryClient();
  const [params, setParams] = useSearchParams();
  const [explain, setExplain] = useState(false);
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [modelFilter, setModelFilter] = useState<string>(params.get('model') ?? '');
  const [search, setSearch] = useState('');
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const paramA = params.get('a') ?? '';
  const paramB = params.get('b') ?? '';
  const [selectedIdA, setSelectedIdA] = useState<string>(paramA);
  const [selectedIdB, setSelectedIdB] = useState<string>(paramB);

  // Queries
  const resultsQuery = useQuery({
    queryKey: ['evalResults'],
    queryFn: () => api.listEvalResults(),
  });

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: api.listAllModels,
  });

  const deleteResult = useMutation({
    mutationFn: (id: string) => api.deleteEvalResult(id),
    onSuccess: (_data, deletedId) => {
      setDeleteError(null);
      const next = new URLSearchParams(params);
      let changedUrl = false;
      if (selectedIdA === deletedId) {
        setSelectedIdA('');
        next.delete('a');
        changedUrl = true;
      }
      if (selectedIdB === deletedId) {
        setSelectedIdB('');
        next.delete('b');
        changedUrl = true;
      }
      if (changedUrl) {
        setParams(next, { replace: true });
      }
      void queryClient.invalidateQueries({ queryKey: ['evalResults'] });
      void queryClient.invalidateQueries({ queryKey: ['evaluations'] });
      void queryClient.invalidateQueries({ queryKey: ['bpbEvaluations'] });
      void queryClient.invalidateQueries({ queryKey: ['models'] });
    },
    onError: (err: Error) => {
      setDeleteError(err.message);
    },
  });

  const allResults = useMemo(() => resultsQuery.data ?? [], [resultsQuery.data]);
  const modelList: Model[] = useMemo(() => modelsQuery.data?.items ?? [], [modelsQuery.data]);

  // Filtered results for the selection list
  const filteredResults = useMemo(() => {
    return allResults.filter((r) => {
      if (typeFilter !== 'all' && r.result_type !== typeFilter) return false;
      if (modelFilter && r.model_id !== modelFilter) return false;
      if (search.trim()) {
        const q = search.trim().toLowerCase();
        const text =
          `${r.model_name} ${r.benchmark_name ?? ''} ${r.prompt ?? ''} ${r.answer ?? ''} ${r.result_type}`.toLowerCase();
        if (!text.includes(q)) return false;
      }
      return true;
    });
  }, [allResults, typeFilter, modelFilter, search]);

  // Lookup selected records directly from the loaded array
  const resultA = useMemo(() => {
    return allResults.find((r) => r.id === selectedIdA);
  }, [allResults, selectedIdA]);

  const resultB = useMemo(() => {
    return allResults.find((r) => r.id === selectedIdB);
  }, [allResults, selectedIdB]);

  const handleSelectA = (id: string) => {
    setSelectedIdA(id);
    const next = new URLSearchParams(params);
    if (id) next.set('a', id);
    else next.delete('a');
    setParams(next, { replace: true });
  };

  const handleSelectB = (id: string) => {
    setSelectedIdB(id);
    const next = new URLSearchParams(params);
    if (id) next.set('b', id);
    else next.delete('b');
    setParams(next, { replace: true });
  };

  const clearSelection = () => {
    setSelectedIdA('');
    setSelectedIdB('');
    const next = new URLSearchParams(params);
    next.delete('a');
    next.delete('b');
    setParams(next, { replace: true });
  };

  const bothSelected = Boolean(resultA && resultB);

  return (
    <>
      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">Compare evaluations</h1>
          <p className="lede">
            Compare saved benchmark scores, bits-per-byte measurements, and prompt outputs directly
            without rerunning evaluations or generating new model completions.
          </p>
        </div>
        <ExplainToggle checked={explain} onChange={setExplain} />
      </div>

      {explain ? (
        <div className="help-panel">
          <p>
            <strong>Direct Stored Data</strong>: This comparison loads saved numbers and generated
            answers directly from SQLite. It does not invoke PyTorch or trigger background jobs.
          </p>
          <p>
            <strong>Bits per byte (BPB)</strong>: Direct compression comparison on held-out text.
            Lower is better. When comparing models that evaluated the same validation hash, the test
            corpus was identical byte-for-byte.
          </p>
          <p>
            <strong>Prompt generation</strong>: Side-by-side prompt output review shows how
            different checkpoints, temperatures, or model architectures answered identical or
            different prompts.
          </p>
          <p>
            <strong>Code version & settings</strong>: Exact git commit SHAs and generation
            parameters are displayed alongside results so you can verify reproducibility.
          </p>
        </div>
      ) : null}

      {/* Comparison Workspace Card */}
      <Card>
        <CardHead
          title="Direct side-by-side comparison"
          end={
            bothSelected ? (
              <button type="button" className="btn btn-sm" onClick={clearSelection}>
                Clear comparison
              </button>
            ) : null
          }
        />
        <div className="card-body stack">
          {/* Missing records notice */}
          {selectedIdA && !resultA && !resultsQuery.isPending ? (
            <Notice tone="bad">
              <span>Selected Result A ({selectedIdA}) was not found or has been deleted. </span>
              <button
                type="button"
                className="btn btn-sm"
                style={{ marginLeft: 8 }}
                onClick={() => handleSelectA('')}
              >
                Clear A
              </button>
            </Notice>
          ) : null}
          {selectedIdB && !resultB && !resultsQuery.isPending ? (
            <Notice tone="bad">
              <span>Selected Result B ({selectedIdB}) was not found or has been deleted. </span>
              <button
                type="button"
                className="btn btn-sm"
                style={{ marginLeft: 8 }}
                onClick={() => handleSelectB('')}
              >
                Clear B
              </button>
            </Notice>
          ) : null}

          {/* Both selected: display the detailed comparison */}
          {bothSelected && resultA && resultB ? (
            <div className="stack" style={{ gap: 20 }}>
              {/* Top metadata overview */}
              <div className="preparation-grid">
                {/* Result A Card */}
                <div
                  className="card"
                  style={{
                    padding: 14,
                    background: 'var(--color-surface-soft)',
                    border: '1px solid var(--rule-soft)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                    }}
                  >
                    <span className="label" style={{ margin: 0, color: 'var(--accent)' }}>
                      Result A (Baseline)
                    </span>
                    <Chip tone={TYPE_TONE[resultA.result_type] ?? 'idle'}>
                      {resultA.result_type}
                    </Chip>
                  </div>
                  <h3 className="h3" style={{ margin: '4px 0 2px' }}>
                    {resultA.model_name}
                  </h3>
                  <p className="muted" style={{ margin: '0 0 8px', fontSize: '0.85em' }}>
                    {resultA.checkpoint_id ? `Checkpoint: ${resultA.checkpoint_id}` : 'Base model'}{' '}
                    · {relativeDate(resultA.created_at)}
                  </p>
                  <dl className="preview-counts" style={{ fontSize: '0.85rem' }}>
                    <div>
                      <dt>Code version</dt>
                      <dd>
                        {resultA.code_version ? (
                          <code title={resultA.code_version}>
                            {resultA.code_version.slice(0, 7)}
                          </code>
                        ) : (
                          <span className="muted">Unavailable</span>
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt>Settings</dt>
                      <dd style={{ fontSize: '0.8rem' }}>{formatSettings(resultA.settings)}</dd>
                    </div>
                  </dl>
                </div>

                {/* Result B Card */}
                <div
                  className="card"
                  style={{
                    padding: 14,
                    background: 'var(--color-surface-soft)',
                    border: '1px solid var(--rule-soft)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                    }}
                  >
                    <span className="label" style={{ margin: 0, color: 'var(--accent)' }}>
                      Result B (Comparison)
                    </span>
                    <Chip tone={TYPE_TONE[resultB.result_type] ?? 'idle'}>
                      {resultB.result_type}
                    </Chip>
                  </div>
                  <h3 className="h3" style={{ margin: '4px 0 2px' }}>
                    {resultB.model_name}
                  </h3>
                  <p className="muted" style={{ margin: '0 0 8px', fontSize: '0.85em' }}>
                    {resultB.checkpoint_id ? `Checkpoint: ${resultB.checkpoint_id}` : 'Base model'}{' '}
                    · {relativeDate(resultB.created_at)}
                  </p>
                  <dl className="preview-counts" style={{ fontSize: '0.85rem' }}>
                    <div>
                      <dt>Code version</dt>
                      <dd>
                        {resultB.code_version ? (
                          <code title={resultB.code_version}>
                            {resultB.code_version.slice(0, 7)}
                          </code>
                        ) : (
                          <span className="muted">Unavailable</span>
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt>Settings</dt>
                      <dd style={{ fontSize: '0.8rem' }}>{formatSettings(resultB.settings)}</dd>
                    </div>
                  </dl>
                </div>
              </div>

              {/* Context / Type difference notice */}
              {resultA.result_type !== resultB.result_type ? (
                <Notice tone="warn">
                  Selected results have different types: <strong>{resultA.result_type}</strong> vs{' '}
                  <strong>{resultB.result_type}</strong>. Showing respective metrics and contents
                  below.
                </Notice>
              ) : null}

              {/* BPB Comparison */}
              {resultA.result_type === 'bpb' && resultB.result_type === 'bpb' ? (
                <div
                  style={{
                    padding: 16,
                    borderRadius: 8,
                    background: 'var(--color-surface-soft)',
                    border: '1px solid var(--rule-soft)',
                  }}
                >
                  <h4 className="h4" style={{ margin: '0 0 12px' }}>
                    Numeric Bits-Per-Byte Comparison
                  </h4>
                  <div className="scroll-x">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Result A ({resultA.model_name})</th>
                          <th>Result B ({resultB.model_name})</th>
                          <th>Difference (B − A)</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>
                            <strong>Bits per byte (BPB)</strong>
                            <div className="muted" style={{ fontSize: '0.75rem' }}>
                              Lower is better
                            </div>
                          </td>
                          <td style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8' }}>
                            {resultA.bpb !== null && resultA.bpb !== undefined
                              ? resultA.bpb.toFixed(4)
                              : '—'}
                          </td>
                          <td style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8' }}>
                            {resultB.bpb !== null && resultB.bpb !== undefined
                              ? resultB.bpb.toFixed(4)
                              : '—'}
                          </td>
                          <td>
                            {resultA.bpb !== null &&
                            resultA.bpb !== undefined &&
                            resultB.bpb !== null &&
                            resultB.bpb !== undefined
                              ? (() => {
                                  const delta = resultB.bpb - resultA.bpb;
                                  const isBetter = delta < 0;
                                  return (
                                    <Chip tone={isBetter ? 'ok' : delta > 0 ? 'bad' : 'idle'}>
                                      {delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}{' '}
                                      {isBetter ? '(better)' : delta > 0 ? '(worse)' : '(equal)'}
                                    </Chip>
                                  );
                                })()
                              : '—'}
                          </td>
                        </tr>
                        <tr>
                          <td>Total NLL</td>
                          <td>
                            {resultA.total_nll !== null && resultA.total_nll !== undefined
                              ? `${resultA.total_nll.toFixed(2)} nats`
                              : '—'}
                          </td>
                          <td>
                            {resultB.total_nll !== null && resultB.total_nll !== undefined
                              ? `${resultB.total_nll.toFixed(2)} nats`
                              : '—'}
                          </td>
                          <td>
                            {resultA.total_nll !== null &&
                            resultA.total_nll !== undefined &&
                            resultB.total_nll !== null &&
                            resultB.total_nll !== undefined
                              ? (() => {
                                  const diff = resultB.total_nll - resultA.total_nll;
                                  return `${diff > 0 ? `+${diff.toFixed(2)}` : diff.toFixed(2)} nats`;
                                })()
                              : '—'}
                          </td>
                        </tr>
                        <tr>
                          <td>Scored UTF-8 bytes</td>
                          <td>{resultA.scored_bytes?.toLocaleString() ?? '—'}</td>
                          <td>{resultB.scored_bytes?.toLocaleString() ?? '—'}</td>
                          <td>
                            {resultA.scored_bytes === resultB.scored_bytes ? (
                              <Chip tone="ok">Identical corpus size</Chip>
                            ) : (
                              'Different byte count'
                            )}
                          </td>
                        </tr>
                        <tr>
                          <td>Scored tokens</td>
                          <td>{resultA.scored_tokens?.toLocaleString() ?? '—'}</td>
                          <td>{resultB.scored_tokens?.toLocaleString() ?? '—'}</td>
                          <td>
                            {resultA.scored_tokens && resultB.scored_tokens
                              ? `${resultB.scored_tokens - resultA.scored_tokens} tokens`
                              : '—'}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}

              {/* Benchmark Comparison */}
              {resultA.result_type === 'benchmark' && resultB.result_type === 'benchmark' ? (
                <div
                  style={{
                    padding: 16,
                    borderRadius: 8,
                    background: 'var(--color-surface-soft)',
                    border: '1px solid var(--rule-soft)',
                  }}
                >
                  <h4 className="h4" style={{ margin: '0 0 12px' }}>
                    Numeric Benchmark Score Comparison
                  </h4>
                  <div className="scroll-x">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Benchmark</th>
                          <th>Result A ({resultA.model_name})</th>
                          <th>Result B ({resultB.model_name})</th>
                          <th>Difference (B − A)</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>
                            <strong>{resultA.benchmark_name || 'Benchmark score'}</strong>
                            {resultA.benchmark_name !== resultB.benchmark_name ? (
                              <div className="muted" style={{ fontSize: '0.75rem' }}>
                                (Compared against {resultB.benchmark_name || 'Benchmark'})
                              </div>
                            ) : null}
                          </td>
                          <td style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                            {resultA.benchmark_score !== null &&
                            resultA.benchmark_score !== undefined
                              ? resultA.benchmark_score
                              : '—'}
                          </td>
                          <td style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                            {resultB.benchmark_score !== null &&
                            resultB.benchmark_score !== undefined
                              ? resultB.benchmark_score
                              : '—'}
                          </td>
                          <td>
                            {resultA.benchmark_score !== null &&
                            resultA.benchmark_score !== undefined &&
                            resultB.benchmark_score !== null &&
                            resultB.benchmark_score !== undefined
                              ? (() => {
                                  const diff = resultB.benchmark_score - resultA.benchmark_score;
                                  return (
                                    <Chip tone={diff > 0 ? 'ok' : diff < 0 ? 'bad' : 'idle'}>
                                      {diff > 0 ? `+${diff.toFixed(2)}` : diff.toFixed(2)}
                                    </Chip>
                                  );
                                })()
                              : '—'}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}

              {/* Prompt Answers Side-by-Side Comparison */}
              {resultA.result_type === 'prompt' || resultB.result_type === 'prompt' ? (
                <div
                  style={{
                    padding: 16,
                    borderRadius: 8,
                    background: 'var(--color-surface-soft)',
                    border: '1px solid var(--rule-soft)',
                  }}
                >
                  <h4 className="h4" style={{ margin: '0 0 12px' }}>
                    Side-by-Side Prompt & Generated Answer Comparison
                  </h4>

                  {resultA.prompt && resultB.prompt && resultA.prompt === resultB.prompt ? (
                    <div
                      style={{
                        padding: 12,
                        borderRadius: 6,
                        background: 'var(--surface)',
                        border: '1px solid var(--rule)',
                        marginBottom: 16,
                      }}
                    >
                      <div className="label" style={{ fontSize: '0.75rem', marginBottom: 4 }}>
                        Prompt evaluated on both models
                      </div>
                      <pre
                        style={{
                          margin: 0,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          fontFamily: 'inherit',
                          fontSize: '0.9rem',
                        }}
                      >
                        {resultA.prompt}
                      </pre>
                    </div>
                  ) : null}

                  <div className="preparation-grid">
                    {/* Column A Output */}
                    <div className="stack" style={{ gap: 8 }}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <strong>{resultA.model_name} output</strong>
                        <span className="muted" style={{ fontSize: '0.8rem' }}>
                          {resultA.answer ? `${resultA.answer.length} chars` : 'No answer'}
                        </span>
                      </div>
                      {resultA.prompt && resultA.prompt !== resultB.prompt ? (
                        <div
                          style={{
                            padding: 8,
                            borderRadius: 6,
                            background: 'var(--page)',
                            fontSize: '0.85rem',
                          }}
                        >
                          <span className="label" style={{ fontSize: '0.7rem' }}>
                            Prompt A
                          </span>
                          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {resultA.prompt}
                          </div>
                        </div>
                      ) : null}
                      <pre
                        style={{
                          margin: 0,
                          padding: 12,
                          borderRadius: 6,
                          background: 'var(--surface)',
                          border: '1px solid var(--rule)',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          minHeight: 140,
                          fontSize: '0.9rem',
                          lineHeight: 1.5,
                        }}
                      >
                        {resultA.answer || <span className="muted">No answer generated.</span>}
                      </pre>
                    </div>

                    {/* Column B Output */}
                    <div className="stack" style={{ gap: 8 }}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <strong>{resultB.model_name} output</strong>
                        <span className="muted" style={{ fontSize: '0.8rem' }}>
                          {resultB.answer ? `${resultB.answer.length} chars` : 'No answer'}
                        </span>
                      </div>
                      {resultB.prompt && resultA.prompt !== resultB.prompt ? (
                        <div
                          style={{
                            padding: 8,
                            borderRadius: 6,
                            background: 'var(--page)',
                            fontSize: '0.85rem',
                          }}
                        >
                          <span className="label" style={{ fontSize: '0.7rem' }}>
                            Prompt B
                          </span>
                          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {resultB.prompt}
                          </div>
                        </div>
                      ) : null}
                      <pre
                        style={{
                          margin: 0,
                          padding: 12,
                          borderRadius: 6,
                          background: 'var(--surface)',
                          border: '1px solid var(--rule)',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          minHeight: 140,
                          fontSize: '0.9rem',
                          lineHeight: 1.5,
                        }}
                      >
                        {resultB.answer || <span className="muted">No answer generated.</span>}
                      </pre>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <div
              style={{
                padding: '24px 16px',
                textAlign: 'center',
                borderRadius: 8,
                background: 'var(--color-surface-soft)',
                border: '1px dashed var(--rule)',
              }}
            >
              <p style={{ fontWeight: 600, margin: '0 0 6px' }}>
                Select two saved evaluation results to compare
              </p>
              <p className="muted" style={{ margin: 0, fontSize: '0.9rem' }}>
                Use the &quot;Compare as A&quot; and &quot;Compare as B&quot; buttons in the table
                below to select baseline and comparison runs.
              </p>
              <div style={{ marginTop: 12, display: 'flex', justifyContent: 'center', gap: 12 }}>
                <span className="badge">
                  Result A:{' '}
                  {resultA ? `${resultA.model_name} (${resultA.result_type})` : 'Not selected'}
                </span>
                <span className="badge">
                  Result B:{' '}
                  {resultB ? `${resultB.model_name} (${resultB.result_type})` : 'Not selected'}
                </span>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Saved Results Directory / Selection Table */}
      <Card>
        <CardHead
          title="Saved evaluation results"
          end={
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <input
                type="search"
                className="input input-sm"
                placeholder="Search results…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ maxWidth: 200 }}
              />
            </div>
          }
        />
        <div className="card-body stack">
          {/* Filter Bar */}
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 12,
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div className="seg" role="tablist" aria-label="Result type filter">
              <button
                type="button"
                className={typeFilter === 'all' ? 'on' : undefined}
                aria-pressed={typeFilter === 'all'}
                onClick={() => setTypeFilter('all')}
              >
                All types ({allResults.length})
              </button>
              <button
                type="button"
                className={typeFilter === 'prompt' ? 'on' : undefined}
                aria-pressed={typeFilter === 'prompt'}
                onClick={() => setTypeFilter('prompt')}
              >
                Prompts ({allResults.filter((r) => r.result_type === 'prompt').length})
              </button>
              <button
                type="button"
                className={typeFilter === 'bpb' ? 'on' : undefined}
                aria-pressed={typeFilter === 'bpb'}
                onClick={() => setTypeFilter('bpb')}
              >
                Bits per byte ({allResults.filter((r) => r.result_type === 'bpb').length})
              </button>
              <button
                type="button"
                className={typeFilter === 'benchmark' ? 'on' : undefined}
                aria-pressed={typeFilter === 'benchmark'}
                onClick={() => setTypeFilter('benchmark')}
              >
                Benchmarks ({allResults.filter((r) => r.result_type === 'benchmark').length})
              </button>
            </div>

            {modelList.length > 0 ? (
              <label
                className="field"
                style={{ margin: 0, flexDirection: 'row', alignItems: 'center', gap: 8 }}
              >
                <span className="label" style={{ margin: 0, whiteSpace: 'nowrap' }}>
                  Filter model:
                </span>
                <select
                  className="input input-sm"
                  value={modelFilter}
                  onChange={(e) => setModelFilter(e.target.value)}
                >
                  <option value="">All models</option>
                  {modelList.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>

          {deleteError ? <Notice tone="bad">{deleteError}</Notice> : null}

          {/* Table or Empty State */}
          {resultsQuery.error ? (
            <ErrorBox
              message={resultsQuery.error.message}
              onRetry={() => void resultsQuery.refetch()}
            />
          ) : resultsQuery.isPending ? (
            <p role="status">Loading saved results…</p>
          ) : filteredResults.length === 0 ? (
            <Empty
              title={
                search || typeFilter !== 'all' || modelFilter
                  ? 'No matching saved results'
                  : 'No evaluation results saved yet'
              }
              hint={
                search || typeFilter !== 'all' || modelFilter
                  ? 'Try adjusting your type, model, or text filters.'
                  : 'Run evaluations on the Models or Evaluation page to generate saved records.'
              }
            />
          ) : (
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Type</th>
                    <th>Model & Checkpoint</th>
                    <th>Identifier / Content</th>
                    <th>Score / Output</th>
                    <th>Settings</th>
                    <th>Code version</th>
                    <th>Date</th>
                    <th>Compare action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredResults.map((r) => {
                    const isA = r.id === selectedIdA;
                    const isB = r.id === selectedIdB;
                    return (
                      <tr
                        key={r.id}
                        style={{
                          background: isA || isB ? 'var(--color-surface-soft)' : undefined,
                        }}
                      >
                        <td>
                          <Chip tone={TYPE_TONE[r.result_type] ?? 'idle'}>{r.result_type}</Chip>
                        </td>
                        <td>
                          <strong>{r.model_name}</strong>
                          {r.checkpoint_id ? (
                            <div className="muted" style={{ fontSize: '0.75rem' }}>
                              ckpt: {r.checkpoint_id.slice(0, 8)}…
                            </div>
                          ) : null}
                        </td>
                        <td style={{ maxWidth: 220 }}>
                          <span style={{ fontSize: '0.85rem' }}>{resultSummary(r)}</span>
                        </td>
                        <td>
                          {r.result_type === 'bpb' ? (
                            <strong style={{ color: '#38bdf8' }}>
                              {r.bpb !== null && r.bpb !== undefined ? `${r.bpb.toFixed(4)}` : '—'}
                            </strong>
                          ) : r.result_type === 'benchmark' ? (
                            <strong>
                              {r.benchmark_score !== null && r.benchmark_score !== undefined
                                ? r.benchmark_score
                                : '—'}
                            </strong>
                          ) : (
                            <span className="muted" style={{ fontSize: '0.8rem' }}>
                              {r.answer ? `${r.answer.slice(0, 40)}…` : '—'}
                            </span>
                          )}
                        </td>
                        <td style={{ maxWidth: 180, fontSize: '0.75rem' }}>
                          <span className="muted">{formatSettings(r.settings)}</span>
                        </td>
                        <td>
                          {r.code_version ? (
                            <code title={r.code_version} style={{ fontSize: '0.8rem' }}>
                              {r.code_version.slice(0, 7)}
                            </code>
                          ) : (
                            <span className="muted" style={{ fontSize: '0.8rem' }}>
                              Unavailable
                            </span>
                          )}
                        </td>
                        <td style={{ whiteSpace: 'nowrap', fontSize: '0.85rem' }}>
                          {relativeDate(r.created_at)}
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                            <button
                              type="button"
                              className={`btn btn-sm ${isA ? 'btn-primary' : ''}`}
                              onClick={() => handleSelectA(isA ? '' : r.id)}
                            >
                              {isA ? 'Selected A' : 'Set A'}
                            </button>
                            <button
                              type="button"
                              className={`btn btn-sm ${isB ? 'btn-primary' : ''}`}
                              onClick={() => handleSelectB(isB ? '' : r.id)}
                            >
                              {isB ? 'Selected B' : 'Set B'}
                            </button>
                            <ConfirmButton
                              label="Delete"
                              confirmLabel="Delete result"
                              onConfirm={() => deleteResult.mutate(r.id)}
                              disabled={deleteResult.isPending}
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
    </>
  );
}
