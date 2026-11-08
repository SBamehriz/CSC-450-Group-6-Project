import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import {
  api,
  type BpbEvaluation,
  type BpbEvaluationCreate,
  type Dataset,
  type Model,
  type PromptSet,
  type PromptSetCreate,
} from '../api/client';
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
  type Tone,
} from '../components/ui';
import { relativeDate } from '../lib/format';

const EVAL_TONE: Record<string, Tone> = {
  completed: 'ok',
  running: 'warn',
  pending: 'idle',
  failed: 'bad',
  cancelled: 'bad',
};

export default function EvaluationsPage() {
  const queryClient = useQueryClient();
  const [params] = useSearchParams();
  const preselectedModelId = params.get('model') ?? '';
  const preselectedPromptSetId = params.get('prompt_set') ?? '';
  const initialTab = params.get('tab') === 'bpb' ? 'bpb' : 'prompts';

  const [tab, setTab] = useState<'prompts' | 'bpb'>(initialTab);
  const [explain, setExplain] = useState(false);
  const [selectedEvalId, setSelectedEvalId] = useState<string | null>(null);

  // Queries
  const models = useQuery({ queryKey: ['models'], queryFn: api.listAllModels });
  const promptSets = useQuery({ queryKey: ['promptSets'], queryFn: api.listPromptSets });
  const evaluations = useQuery({
    queryKey: ['evaluations'],
    queryFn: api.listEvaluations,
    refetchInterval: (query) =>
      query.state.data?.some((e) => e.status === 'pending' || e.status === 'running')
        ? 2000
        : false,
  });

  const datasets = useQuery({ queryKey: ['datasets'], queryFn: api.listDatasets });
  const bpbEvaluations = useQuery({
    queryKey: ['bpbEvaluations'],
    queryFn: api.listBpbEvaluations,
    refetchInterval: (query) =>
      query.state.data?.some((e) => e.status === 'pending' || e.status === 'running')
        ? 2000
        : false,
  });

  // Prompt Set Form State
  const [editingId, setEditingId] = useState<string | null>(null);
  const [psName, setPsName] = useState('');
  const [psDescription, setPsDescription] = useState('');
  const [promptsList, setPromptsList] = useState<string[]>(['']);
  const [psFormError, setPsFormError] = useState<string | null>(null);

  // Evaluation Form State
  const [selectedModelId, setSelectedModelId] = useState(preselectedModelId);
  const [selectedPromptSetId, setSelectedPromptSetId] = useState(preselectedPromptSetId);
  const [evalRunName, setEvalRunName] = useState('');
  const [maxTokens, setMaxTokens] = useState(32);
  const [temperature, setTemperature] = useState(0.7);
  const [evalFormError, setEvalFormError] = useState<string | null>(null);

  // BPB Evaluation Form State (SP9.11)
  const [bpbModelSource, setBpbModelSource] = useState<'native' | 'imported'>('native');
  const [bpbSelectedModelId, setBpbSelectedModelId] = useState(preselectedModelId);
  const [bpbImportedModelName, setBpbImportedModelName] = useState('gpt2');
  const [bpbSelectedDatasetId, setBpbSelectedDatasetId] = useState('');
  const [bpbRunName, setBpbRunName] = useState('');
  const [bpbCtxLen, setBpbCtxLen] = useState(512);
  const [bpbFormError, setBpbFormError] = useState<string | null>(null);

  const refreshAll = () => {
    void queryClient.invalidateQueries({ queryKey: ['promptSets'] });
    void queryClient.invalidateQueries({ queryKey: ['evaluations'] });
    void queryClient.invalidateQueries({ queryKey: ['bpbEvaluations'] });
    void queryClient.invalidateQueries({ queryKey: ['evalResults'] });
    void queryClient.invalidateQueries({ queryKey: ['datasets'] });
    void queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };

  // Prompt Set Mutations
  const createPromptSet = useMutation({
    mutationFn: (payload: PromptSetCreate) => api.createPromptSet(payload),
    onSuccess: () => {
      resetPromptSetForm();
      refreshAll();
    },
    onError: (err: Error) => setPsFormError(err.message),
  });

  const updatePromptSet = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: PromptSetCreate }) =>
      api.updatePromptSet(id, payload),
    onSuccess: () => {
      resetPromptSetForm();
      refreshAll();
    },
    onError: (err: Error) => setPsFormError(err.message),
  });

  const deletePromptSet = useMutation({
    mutationFn: (id: string) => api.deletePromptSet(id),
    onSuccess: refreshAll,
  });

  // Evaluation Mutations
  const createEvaluation = useMutation({
    mutationFn: () => {
      if (!selectedModelId) throw new Error('Please select a model to evaluate.');
      if (!selectedPromptSetId) throw new Error('Please select a prompt set.');
      return api.createEvaluation({
        model_id: selectedModelId,
        prompt_set_id: selectedPromptSetId,
        name: evalRunName.trim() || undefined,
        max_new_tokens: maxTokens,
        temperature,
      });
    },
    onSuccess: (ev) => {
      setEvalRunName('');
      setEvalFormError(null);
      setSelectedEvalId(ev.id);
      refreshAll();
    },
    onError: (err: Error) => setEvalFormError(err.message),
  });

  const deleteEvaluation = useMutation({
    mutationFn: (id: string) => api.deleteEvaluation(id),
    onSuccess: () => {
      setSelectedEvalId(null);
      refreshAll();
    },
  });

  // BPB Mutations (SP9.11)
  const createBpbEvaluation = useMutation({
    mutationFn: () => {
      setBpbFormError(null);
      if (bpbModelSource === 'native' && !bpbActiveModelId) {
        throw new Error('Please select a native model.');
      }
      if (bpbModelSource === 'imported' && !bpbImportedModelName.trim()) {
        throw new Error('Please enter an imported model name or path.');
      }
      if (!bpbActiveDatasetId) {
        throw new Error('Please select a dataset snapshot.');
      }
      const payload: BpbEvaluationCreate = {
        name: bpbRunName.trim() || undefined,
        model_source: bpbModelSource,
        model_id: bpbModelSource === 'native' ? bpbActiveModelId : undefined,
        imported_model_name:
          bpbModelSource === 'imported' ? bpbImportedModelName.trim() : undefined,
        dataset_id: bpbActiveDatasetId,
        ctx_len: bpbCtxLen,
      };
      return api.createBpbEvaluation(payload);
    },
    onSuccess: () => {
      setBpbRunName('');
      setBpbFormError(null);
      refreshAll();
    },
    onError: (err: Error) => setBpbFormError(err.message),
  });

  const deleteBpbEvaluation = useMutation({
    mutationFn: (id: string) => api.deleteBpbEvaluation(id),
    onSuccess: refreshAll,
  });

  const resetPromptSetForm = () => {
    setEditingId(null);
    setPsName('');
    setPsDescription('');
    setPromptsList(['']);
    setPsFormError(null);
  };

  const startEditPromptSet = (ps: PromptSet) => {
    setEditingId(ps.id);
    setPsName(ps.name);
    setPsDescription(ps.description);
    setPromptsList(ps.prompts.length ? [...ps.prompts] : ['']);
    setPsFormError(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handlePromptChange = (index: number, value: string) => {
    setPromptsList((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  };

  const addPromptField = () => {
    setPromptsList((prev) => [...prev, '']);
  };

  const removePromptField = (index: number) => {
    setPromptsList((prev) => {
      const next = prev.filter((_, i) => i !== index);
      return next.length ? next : [''];
    });
  };

  const handlePromptSetSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPsFormError(null);
    const cleanName = psName.trim();
    if (!cleanName) {
      setPsFormError('Please give the prompt set a name.');
      return;
    }
    const cleanPrompts = promptsList.map((p) => p.trim()).filter(Boolean);
    if (!cleanPrompts.length) {
      setPsFormError('A prompt set must contain at least one non-empty prompt.');
      return;
    }

    const payload: PromptSetCreate = {
      name: cleanName,
      description: psDescription.trim(),
      prompts: cleanPrompts,
    };

    if (editingId) {
      updatePromptSet.mutate({ id: editingId, payload });
    } else {
      createPromptSet.mutate(payload);
    }
  };

  const modelList = models.data?.items ?? [];
  const promptSetList = promptSets.data ?? [];
  const evalList = evaluations.data ?? [];

  const datasetList = datasets.data?.items ?? [];
  const readySnapshots = datasetList.filter((ds) => ds.status === 'ready' || ds.download_ready);
  const bpbActiveModelId = bpbSelectedModelId || modelList[0]?.id || '';
  const bpbActiveDatasetId = bpbSelectedDatasetId || readySnapshots[0]?.id || '';
  const bpbList = bpbEvaluations.data ?? [];

  // Effective selected model and prompt set for evaluation form
  const activeModelId = selectedModelId || modelList[0]?.id || '';
  const activePromptSetId = selectedPromptSetId || promptSetList[0]?.id || '';

  return (
    <>
      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">Model Evaluation</h1>
          <p className="lede">
            Evaluate small language models using named prompt sets or compute bits per byte (BPB) on
            held-out validation text from frozen snapshots.
          </p>
        </div>
        <ExplainToggle checked={explain} onChange={setExplain} />
      </div>

      {explain ? (
        <div className="help-panel">
          {tab === 'bpb' ? (
            <>
              <p>
                <strong>Bits Per Byte (BPB)</strong>: The cross-entropy loss in nats divided by{' '}
                <code>ln(2) * total_scored_UTF8_bytes</code>. Unlike token loss or perplexity, BPB
                allows direct, unbiased predictive compression comparisons across native and
                imported models with different tokenizers.
              </p>
              <p>
                <strong>Exact Validation Bytes</strong>: Both native and imported model paths score
                the exact same immutable validation text bytes from the selected frozen snapshot.
                Validation split SHA-256 digests are recorded.
              </p>
              <p>
                <strong>Sliding Window & Target Masking</strong>: Documents longer than the context
                window use a sliding window that retains preceding context while masking overlapping
                targets so each target byte is scored exactly once.
              </p>
            </>
          ) : (
            <>
              <p>
                <strong>Prompt Sets</strong>: Curate ordered collections of benchmark prompts to
                test reasoning, creativity, or instruction following across model architectures.
              </p>
              <p>
                <strong>Frozen Reproducibility</strong>: When you start an evaluation, Forge
                captures an independent copy of the exact prompts and their order. Later edits or
                deletions of the prompt set leave historical evaluation results completely intact.
              </p>
              <p>
                <strong>Background Jobs</strong>: Evaluations execute as background jobs with
                real-time progress reporting and cancellation support.
              </p>
            </>
          )}
        </div>
      ) : null}

      <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <div className="seg" role="tablist" aria-label="Evaluation mode">
          <button
            type="button"
            className={tab === 'prompts' ? 'on' : undefined}
            aria-pressed={tab === 'prompts'}
            onClick={() => setTab('prompts')}
          >
            Prompt sets
          </button>
          <button
            type="button"
            className={tab === 'bpb' ? 'on' : undefined}
            aria-pressed={tab === 'bpb'}
            onClick={() => setTab('bpb')}
          >
            Bits per byte (BPB)
          </button>
          <Link to="/compare">Compare</Link>
        </div>
      </div>

      {tab === 'prompts' ? (
        <div className="preparation-grid">
          {/* Left Column: Prompt Sets Management */}
          <div className="stack">
            <Card>
              <CardHead
                title={editingId ? 'Edit prompt set' : 'Create prompt set'}
                end={
                  editingId ? (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={resetPromptSetForm}
                      disabled={updatePromptSet.isPending}
                    >
                      Cancel
                    </button>
                  ) : null
                }
              />
              <form className="card-body stack" onSubmit={handlePromptSetSubmit}>
                <Field
                  label="Name"
                  value={psName}
                  onChange={setPsName}
                  placeholder="reasoning-bench-v1"
                  maxLength={64}
                  disabled={createPromptSet.isPending || updatePromptSet.isPending}
                />
                <Field
                  label="Description"
                  value={psDescription}
                  onChange={setPsDescription}
                  placeholder="Benchmark testing logic, common sense, and formatting"
                  maxLength={500}
                  disabled={createPromptSet.isPending || updatePromptSet.isPending}
                />

                <div className="field">
                  <span className="label">Prompts (ordered collection)</span>
                  <div className="stack" style={{ gap: 8 }}>
                    {promptsList.map((prompt, index) => (
                      <div
                        key={index}
                        style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}
                      >
                        <span
                          className="muted"
                          style={{
                            minWidth: 24,
                            paddingTop: 8,
                            fontSize: '0.85em',
                            textAlign: 'right',
                          }}
                        >
                          {index + 1}.
                        </span>
                        <textarea
                          className="input"
                          rows={2}
                          value={prompt}
                          placeholder={`Enter prompt #${index + 1}…`}
                          onChange={(e) => handlePromptChange(index, e.target.value)}
                          disabled={createPromptSet.isPending || updatePromptSet.isPending}
                        />
                        {promptsList.length > 1 ? (
                          <button
                            type="button"
                            className="btn btn-sm"
                            title="Remove prompt"
                            aria-label={`Remove prompt ${index + 1}`}
                            onClick={() => removePromptField(index)}
                            disabled={createPromptSet.isPending || updatePromptSet.isPending}
                          >
                            ✕
                          </button>
                        ) : null}
                      </div>
                    ))}
                    <button
                      type="button"
                      className="btn btn-sm"
                      style={{ alignSelf: 'flex-start', marginTop: 4 }}
                      onClick={addPromptField}
                      disabled={createPromptSet.isPending || updatePromptSet.isPending}
                    >
                      + Add prompt
                    </button>
                  </div>
                </div>

                {psFormError ? <Notice tone="bad">{psFormError}</Notice> : null}

                <div>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={createPromptSet.isPending || updatePromptSet.isPending}
                  >
                    {editingId
                      ? updatePromptSet.isPending
                        ? 'Saving changes…'
                        : 'Update prompt set'
                      : createPromptSet.isPending
                        ? 'Saving…'
                        : 'Save prompt set'}
                  </button>
                </div>
              </form>
            </Card>

            {/* Saved Prompt Sets List */}
            <Card>
              <CardHead title="Saved prompt sets" />
              <div className="card-body">
                {promptSets.error ? (
                  <ErrorBox
                    message={promptSets.error.message}
                    onRetry={() => void promptSets.refetch()}
                  />
                ) : promptSets.isPending ? (
                  <p role="status">Loading prompt sets…</p>
                ) : !promptSetList.length ? (
                  <Empty
                    title="No prompt sets yet"
                    hint="Create a named prompt set above to evaluate models."
                  />
                ) : (
                  <div className="stack" style={{ gap: 12 }}>
                    {promptSetList.map((ps) => (
                      <article
                        key={ps.id}
                        className="card"
                        style={{ padding: 12, background: 'var(--color-surface-soft)' }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'baseline',
                          }}
                        >
                          <h3 className="h3" style={{ margin: 0 }}>
                            {ps.name}
                          </h3>
                          <Chip tone="ok">
                            {ps.prompt_count} {ps.prompt_count === 1 ? 'prompt' : 'prompts'}
                          </Chip>
                        </div>
                        {ps.description ? (
                          <p className="muted" style={{ margin: '4px 0 0', fontSize: '0.9em' }}>
                            {ps.description}
                          </p>
                        ) : null}
                        <p className="muted" style={{ margin: '4px 0 8px', fontSize: '0.8em' }}>
                          Updated {relativeDate(ps.updated_at)}
                        </p>

                        <div
                          style={{
                            display: 'flex',
                            gap: 8,
                            alignItems: 'center',
                            marginTop: 6,
                          }}
                        >
                          <button
                            type="button"
                            className="btn btn-sm btn-primary"
                            onClick={() => {
                              setSelectedPromptSetId(ps.id);
                              window.scrollTo({ top: 0, behavior: 'smooth' });
                            }}
                          >
                            Use in eval
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm"
                            onClick={() => startEditPromptSet(ps)}
                          >
                            Edit
                          </button>
                          <ConfirmButton
                            label="Delete"
                            confirmLabel={`Delete ${ps.name}`}
                            onConfirm={() => deletePromptSet.mutate(ps.id)}
                            disabled={deletePromptSet.isPending}
                          />
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Right Column: Run Evaluation & Results */}
          <div className="stack">
            {/* Start Evaluation Card */}
            <Card>
              <CardHead title="Start a model evaluation" />
              <form
                className="card-body stack"
                onSubmit={(e) => {
                  e.preventDefault();
                  createEvaluation.mutate();
                }}
              >
                <label className="field">
                  <span className="label">Target model</span>
                  <select
                    className="input"
                    aria-label="Target model"
                    value={activeModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    disabled={!modelList.length || createEvaluation.isPending}
                  >
                    {!modelList.length ? (
                      <option value="">No saved models available</option>
                    ) : (
                      modelList.map((m: Model) => (
                        <option key={m.id} value={m.id}>
                          {m.name} ({m.preset ?? 'custom'} · {m.n_layer}L/{m.d_model}W)
                        </option>
                      ))
                    )}
                  </select>
                  <span className="hint">
                    {!modelList.length ? (
                      <Link to="/models">Create a model first →</Link>
                    ) : (
                      'Choose a model architecture to evaluate.'
                    )}
                  </span>
                </label>

                <label className="field">
                  <span className="label">Prompt set</span>
                  <select
                    className="input"
                    aria-label="Prompt set"
                    value={activePromptSetId}
                    onChange={(e) => setSelectedPromptSetId(e.target.value)}
                    disabled={!promptSetList.length || createEvaluation.isPending}
                  >
                    {!promptSetList.length ? (
                      <option value="">No prompt sets available</option>
                    ) : (
                      promptSetList.map((ps) => (
                        <option key={ps.id} value={ps.id}>
                          {ps.name} ({ps.prompt_count} prompts)
                        </option>
                      ))
                    )}
                  </select>
                  <span className="hint">
                    The prompts and their order will be frozen with this evaluation.
                  </span>
                </label>

                <Field
                  label="Evaluation run name (optional)"
                  value={evalRunName}
                  onChange={setEvalRunName}
                  placeholder="e.g. baseline-nano-eval"
                  maxLength={128}
                  disabled={createEvaluation.isPending}
                />

                <details className="disclosure">
                  <summary>Generation settings</summary>
                  <div className="stack" style={{ gap: 12, marginTop: 8 }}>
                    <label className="field">
                      <span className="label">Max new tokens per prompt</span>
                      <input
                        className="input"
                        type="number"
                        min={1}
                        max={256}
                        value={maxTokens}
                        onChange={(e) => setMaxTokens(Number(e.target.value) || 32)}
                        disabled={createEvaluation.isPending}
                      />
                    </label>
                    <label className="field">
                      <span className="label">Temperature</span>
                      <input
                        className="input"
                        type="number"
                        step={0.1}
                        min={0.1}
                        max={2.0}
                        value={temperature}
                        onChange={(e) => setTemperature(Number(e.target.value) || 0.7)}
                        disabled={createEvaluation.isPending}
                      />
                    </label>
                  </div>
                </details>

                {evalFormError ? <Notice tone="bad">{evalFormError}</Notice> : null}

                <div>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={
                      !modelList.length || !promptSetList.length || createEvaluation.isPending
                    }
                  >
                    {createEvaluation.isPending ? 'Starting evaluation…' : 'Start evaluation'}
                  </button>
                </div>
              </form>
            </Card>

            {/* Evaluations History List */}
            <Card>
              <CardHead title="Evaluation results" />
              <div className="card-body">
                {evaluations.error ? (
                  <ErrorBox
                    message={evaluations.error.message}
                    onRetry={() => void evaluations.refetch()}
                  />
                ) : evaluations.isPending ? (
                  <p role="status">Loading evaluation history…</p>
                ) : !evalList.length ? (
                  <Empty
                    title="No evaluations yet"
                    hint="Select a model and prompt set above to start your first evaluation."
                  />
                ) : (
                  <div className="stack" style={{ gap: 12 }}>
                    {evalList.map((ev) => {
                      const isSelected = ev.id === selectedEvalId;
                      return (
                        <article
                          key={ev.id}
                          className="card"
                          style={{
                            padding: 12,
                            background: isSelected
                              ? 'var(--color-surface-selected, rgba(255,255,255,0.06))'
                              : 'var(--color-surface-soft)',
                            border: isSelected
                              ? '1px solid var(--color-primary, #646cff)'
                              : undefined,
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'baseline',
                            }}
                          >
                            <h3 className="h3" style={{ margin: 0 }}>
                              {ev.name}
                            </h3>
                            <Chip tone={EVAL_TONE[ev.status] ?? 'idle'}>{ev.status}</Chip>
                          </div>
                          <p className="muted" style={{ margin: '4px 0 0', fontSize: '0.85em' }}>
                            Model: <strong>{ev.model_name}</strong> &bull; Prompt set:{' '}
                            <strong>{ev.prompt_set_name || 'Prompt set removed'}</strong> &bull;{' '}
                            {ev.captured_prompts.length} prompts &bull;{' '}
                            {relativeDate(ev.created_at)}
                          </p>
                          {ev.error ? (
                            <div style={{ marginTop: 6 }}>
                              <Notice tone="bad">{ev.error}</Notice>
                            </div>
                          ) : null}

                          <div
                            style={{
                              display: 'flex',
                              gap: 8,
                              alignItems: 'center',
                              marginTop: 8,
                            }}
                          >
                            <button
                              type="button"
                              className="btn btn-sm"
                              onClick={() => setSelectedEvalId(isSelected ? null : ev.id)}
                            >
                              {isSelected ? 'Hide details' : 'View prompts & results'}
                            </button>
                            <ConfirmButton
                              label="Delete"
                              confirmLabel="Delete result"
                              onConfirm={() => deleteEvaluation.mutate(ev.id)}
                              disabled={deleteEvaluation.isPending}
                            />
                          </div>

                          {/* Display Prompts Actually Used (Requirement 8) */}
                          {isSelected ? (
                            <div
                              style={{
                                marginTop: 16,
                                paddingTop: 12,
                                borderTop: '1px solid var(--color-border, rgba(255,255,255,0.1))',
                              }}
                            >
                              <h4 className="h4" style={{ margin: '0 0 8px' }}>
                                Prompts used in this evaluation
                              </h4>
                              {!ev.prompt_set_id ? (
                                <Notice tone="warn">
                                  The original prompt set was deleted, but this evaluation’s
                                  captured prompts and outputs are preserved.
                                </Notice>
                              ) : null}

                              <div className="stack" style={{ gap: 12, marginTop: 8 }}>
                                {ev.captured_prompts.map((promptText, idx) => {
                                  const resultItem = ev.results?.[idx] as
                                    Record<string, unknown> | undefined;
                                  const outputStr =
                                    typeof resultItem?.output === 'string'
                                      ? resultItem.output
                                      : resultItem?.output
                                        ? String(resultItem.output)
                                        : '';
                                  return (
                                    <div
                                      key={idx}
                                      style={{
                                        padding: 10,
                                        borderRadius: 6,
                                        background: 'var(--color-surface, #1e1e24)',
                                        border:
                                          '1px solid var(--color-border, rgba(255,255,255,0.08))',
                                      }}
                                    >
                                      <div
                                        style={{
                                          fontSize: '0.8em',
                                          fontWeight: 600,
                                          color: 'var(--color-primary, #646cff)',
                                          marginBottom: 4,
                                        }}
                                      >
                                        Prompt {idx + 1}
                                      </div>
                                      <pre
                                        style={{
                                          margin: '0 0 8px',
                                          whiteSpace: 'pre-wrap',
                                          wordBreak: 'break-word',
                                          fontFamily: 'inherit',
                                        }}
                                      >
                                        {promptText}
                                      </pre>
                                      {resultItem ? (
                                        <div>
                                          <div
                                            style={{
                                              fontSize: '0.8em',
                                              fontWeight: 600,
                                              color: 'var(--color-text-muted, #a1a1aa)',
                                              marginBottom: 2,
                                            }}
                                          >
                                            Model output:
                                          </div>
                                          <pre
                                            style={{
                                              margin: 0,
                                              padding: 8,
                                              borderRadius: 4,
                                              background: 'rgba(0,0,0,0.25)',
                                              whiteSpace: 'pre-wrap',
                                              wordBreak: 'break-word',
                                              fontSize: '0.9em',
                                            }}
                                          >
                                            {outputStr}
                                          </pre>
                                        </div>
                                      ) : ev.status === 'running' || ev.status === 'pending' ? (
                                        <p
                                          className="muted"
                                          style={{ margin: 0, fontSize: '0.85em' }}
                                        >
                                          Waiting for model generation…
                                        </p>
                                      ) : null}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                )}
              </div>
            </Card>
          </div>
        </div>
      ) : (
        <div className="preparation-grid">
          {/* Left Column: Launch BPB Evaluation Form */}
          <div className="stack">
            <Card>
              <CardHead title="Launch BPB evaluation" />
              <form
                className="card-body stack"
                onSubmit={(e) => {
                  e.preventDefault();
                  createBpbEvaluation.mutate();
                }}
              >
                <div className="field">
                  <span className="label">Model type</span>
                  <div className="seg" role="radiogroup" aria-label="Model type">
                    <button
                      type="button"
                      className={bpbModelSource === 'native' ? 'on' : undefined}
                      aria-pressed={bpbModelSource === 'native'}
                      onClick={() => setBpbModelSource('native')}
                    >
                      Native Forge model
                    </button>
                    <button
                      type="button"
                      className={bpbModelSource === 'imported' ? 'on' : undefined}
                      aria-pressed={bpbModelSource === 'imported'}
                      onClick={() => setBpbModelSource('imported')}
                    >
                      Imported model (SP9.8)
                    </button>
                  </div>
                </div>

                {bpbModelSource === 'native' ? (
                  <label className="field">
                    <span className="label">Select model</span>
                    <select
                      className="input"
                      aria-label="Select native model"
                      value={bpbActiveModelId}
                      onChange={(e) => setBpbSelectedModelId(e.target.value)}
                      disabled={!modelList.length || createBpbEvaluation.isPending}
                    >
                      {!modelList.length ? (
                        <option value="">No models available</option>
                      ) : (
                        modelList.map((m: Model) => (
                          <option key={m.id} value={m.id}>
                            {m.name} ({m.preset ?? 'custom'} · {m.n_layer}L/{m.d_model}W)
                          </option>
                        ))
                      )}
                    </select>
                    <span className="hint">
                      Scores negative log likelihood using our native causal transformer.
                    </span>
                  </label>
                ) : (
                  <Field
                    label="Hugging Face model identifier or path"
                    value={bpbImportedModelName}
                    onChange={setBpbImportedModelName}
                    placeholder="e.g. gpt2 or meta-llama/Llama-3.2-1B"
                    hint="Requires SP9.8 imported model runtime. Evaluates true token likelihoods."
                    disabled={createBpbEvaluation.isPending}
                  />
                )}

                <label className="field">
                  <span className="label">Frozen validation snapshot</span>
                  <select
                    className="input"
                    aria-label="Frozen validation snapshot"
                    value={bpbActiveDatasetId}
                    onChange={(e) => setBpbSelectedDatasetId(e.target.value)}
                    disabled={!readySnapshots.length || createBpbEvaluation.isPending}
                  >
                    {!readySnapshots.length ? (
                      <option value="">No ready snapshots available</option>
                    ) : (
                      readySnapshots.map((ds: Dataset) => (
                        <option key={ds.id} value={ds.id}>
                          {ds.name} ({ds.doc_count?.toLocaleString()} docs)
                        </option>
                      ))
                    )}
                  </select>
                  <span className="hint">
                    Scores the exact immutable validation text bytes of this frozen snapshot.
                  </span>
                </label>

                <Field
                  label="Evaluation name (optional)"
                  value={bpbRunName}
                  onChange={setBpbRunName}
                  placeholder="e.g. gpt2-val-bpb"
                  maxLength={128}
                  disabled={createBpbEvaluation.isPending}
                />

                <label className="field">
                  <span className="label">Context window size</span>
                  <input
                    type="number"
                    className="input"
                    min={64}
                    max={4096}
                    value={bpbCtxLen}
                    onChange={(e) => setBpbCtxLen(Number(e.target.value))}
                    disabled={createBpbEvaluation.isPending}
                  />
                  <span className="hint">
                    Texts longer than context length use bounded sliding window with target masking.
                  </span>
                </label>

                {bpbFormError ? <Notice tone="bad">{bpbFormError}</Notice> : null}

                <div>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={
                      createBpbEvaluation.isPending ||
                      (bpbModelSource === 'native' && !modelList.length) ||
                      !readySnapshots.length
                    }
                  >
                    {createBpbEvaluation.isPending
                      ? 'Launching evaluation…'
                      : 'Launch BPB evaluation'}
                  </button>
                </div>
              </form>
            </Card>
          </div>

          {/* Right Column: BPB Evaluations Results List */}
          <div className="stack">
            <Card>
              <CardHead title="Held-out text BPB evaluations" />
              <div className="card-body">
                {bpbEvaluations.error ? (
                  <ErrorBox
                    message={bpbEvaluations.error.message}
                    onRetry={() => void bpbEvaluations.refetch()}
                  />
                ) : bpbEvaluations.isPending ? (
                  <p role="status">Loading BPB evaluation history…</p>
                ) : !bpbList.length ? (
                  <Empty
                    title="No BPB evaluations yet"
                    hint="Select a model and snapshot on the left to evaluate predictive bits per byte."
                  />
                ) : (
                  <div className="stack" style={{ gap: 12 }}>
                    {bpbList.map((ev: BpbEvaluation) => (
                      <article
                        key={ev.id}
                        className="card"
                        style={{ padding: 14, background: 'var(--color-surface-soft)' }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'baseline',
                          }}
                        >
                          <h3 className="h3" style={{ margin: 0 }}>
                            {ev.name}
                          </h3>
                          <Chip tone={EVAL_TONE[ev.status] ?? 'idle'}>{ev.status}</Chip>
                        </div>

                        <p className="muted" style={{ margin: '4px 0 10px', fontSize: '0.85em' }}>
                          Model: <strong>{ev.model_name}</strong>{' '}
                          <Chip tone={ev.model_source === 'imported' ? 'warn' : 'idle'}>
                            {ev.model_source}
                          </Chip>{' '}
                          &bull; Snapshot: <strong>{ev.dataset_name}</strong> &bull;{' '}
                          {relativeDate(ev.created_at)}
                        </p>

                        <div
                          style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                            gap: 10,
                            padding: 10,
                            borderRadius: 6,
                            background: 'rgba(0, 0, 0, 0.2)',
                            marginBottom: 10,
                          }}
                        >
                          <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                              Bits per byte
                            </div>
                            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#38bdf8' }}>
                              {ev.bpb !== null && ev.bpb !== undefined ? ev.bpb.toFixed(4) : '—'}
                            </div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                              (lower is better)
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                              Total NLL
                            </div>
                            <div style={{ fontSize: '1rem', fontWeight: 600 }}>
                              {ev.total_nll !== null && ev.total_nll !== undefined
                                ? `${ev.total_nll.toFixed(2)} nats`
                                : '—'}
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                              Scored UTF-8 bytes
                            </div>
                            <div style={{ fontSize: '1rem', fontWeight: 600 }}>
                              {ev.scored_bytes?.toLocaleString() ?? '—'}
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                              Scored tokens
                            </div>
                            <div style={{ fontSize: '1rem', fontWeight: 600 }}>
                              {ev.scored_tokens?.toLocaleString() ?? '—'}
                            </div>
                          </div>
                        </div>

                        <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
                          <div>
                            Tokenizer: <code>{ev.tokenizer_identity}</code>
                          </div>
                          <div style={{ wordBreak: 'break-all', marginTop: 2 }}>
                            Validation file hash (SHA-256):{' '}
                            <code>{ev.validation_file_hash || '—'}</code>
                          </div>
                        </div>

                        {ev.error ? (
                          <div style={{ marginTop: 8 }}>
                            <Notice tone="bad">{ev.error}</Notice>
                          </div>
                        ) : null}

                        <div
                          style={{
                            display: 'flex',
                            gap: 8,
                            alignItems: 'center',
                            marginTop: 10,
                          }}
                        >
                          <ConfirmButton
                            label="Delete"
                            confirmLabel="Delete BPB result"
                            onConfirm={() => deleteBpbEvaluation.mutate(ev.id)}
                            disabled={deleteBpbEvaluation.isPending}
                          />
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </div>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
