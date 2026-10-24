import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useState } from 'react';

import { api, type GpuCheck, type Model, type ModelCreate, type Preset } from '../api/client';
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
} from '../components/ui';
import { bytesToSize, compactNumber, gigabytes, relativeDate } from '../lib/format';

interface PresetConfig {
  n_layer: number;
  d_model: number;
  n_head: number;
  ctx_len: number;
}

const PRESETS: Record<string, PresetConfig> = {
  nano: { n_layer: 4, d_model: 128, n_head: 4, ctx_len: 512 },
  micro: { n_layer: 6, d_model: 256, n_head: 8, ctx_len: 512 },
  small: { n_layer: 8, d_model: 512, n_head: 8, ctx_len: 512 },
};

const STEPS = [1, 2, 4, 8, 16];

function calculateFormulaParams(
  n_layer: number,
  d_model: number,
  ctx_len: number,
  vocab_size: number,
): number {
  const embeddings = vocab_size * d_model + ctx_len * d_model;
  const attn = 4 * d_model * d_model + 4 * d_model;
  const mlp = 8 * d_model * d_model + 5 * d_model;
  const norms = 4 * d_model;
  const per_block = attn + mlp + norms;
  const final_norm = 2 * d_model;
  return embeddings + n_layer * per_block + final_norm;
}

function calculateEstimatedVram(
  params: number,
  n_layer: number,
  d_model: number,
  ctx_len: number,
  micro_batch_seqs = 8,
): number {
  const optimizerAndWeights = 16 * params;
  const activations = micro_batch_seqs * ctx_len * d_model * n_layer * 16;
  return optimizerAndWeights + activations;
}

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const presets = useQuery({ queryKey: ['presets'], queryFn: api.listPresets });
  const machine = useQuery({ queryKey: ['gpu'], queryFn: api.checkGpu });
  const [sequences, setSequences] = useState(8);
  const card = machine.data?.cards[0];

  const vocabularies = useQuery({
    queryKey: ['vocabularies'],
    queryFn: api.listVocabularies,
  });
  const models = useQuery({
    queryKey: ['models'],
    queryFn: api.listAllModels,
  });

  const [explain, setExplain] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [preset, setPreset] = useState<'nano' | 'micro' | 'small' | 'custom'>('nano');
  const [nLayer, setNLayer] = useState(4);
  const [dModel, setDModel] = useState(128);
  const [nHead, setNHead] = useState(4);
  const [ctxLen, setCtxLen] = useState(512);
  const [search, setSearch] = useState('');
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Selected vocabulary details
  const vocabs = vocabularies.data ?? [];
  const selectedVocab = vocabs.find((v) => v.dataset_id === selectedDatasetId);
  const actualVocabSize = selectedVocab ? selectedVocab.vocab_size : 16384;

  const handlePresetChange = (newPreset: 'nano' | 'micro' | 'small' | 'custom') => {
    setPreset(newPreset);
    if (newPreset !== 'custom') {
      const cfg = PRESETS[newPreset];
      setNLayer(cfg.n_layer);
      setDModel(cfg.d_model);
      setNHead(cfg.n_head);
      setCtxLen(cfg.ctx_len);
    }
  };

  // Plain-language shape validation using model code's actual validation rules
  const shapeError = useMemo(() => {
    if (nHead <= 0) return 'Number of attention heads must be at least 1.';
    if (dModel % nHead !== 0) {
      const headDim = (dModel / nHead).toFixed(1);
      return (
        `Model width (${dModel}) must divide evenly by attention heads (${nHead}). ` +
        `Each head would get ${headDim} dimensions, but head dimensions must be integers.`
      );
    }
    if (nLayer < 1 || nLayer > 24) {
      return `Number of blocks (layers) must be between 1 and 24 (got ${nLayer}).`;
    }
    if (dModel < 1 || dModel > 768) {
      return `Model width (d_model) must be between 1 and 768 (got ${dModel}).`;
    }
    if (![256, 512, 1024].includes(ctxLen)) {
      return `Context length must be 256, 512, or 1024 tokens (got ${ctxLen}).`;
    }
    if (actualVocabSize < 2) {
      return `Vocabulary size must be at least 2 tokens (got ${actualVocabSize}).`;
    }
    return null;
  }, [nLayer, dModel, nHead, ctxLen, actualVocabSize]);

  // Parameter count and estimated training memory
  const preview = useMemo(() => {
    if (shapeError) return null;
    const params = calculateFormulaParams(nLayer, dModel, ctxLen, actualVocabSize);
    const vramBytes = calculateEstimatedVram(params, nLayer, dModel, ctxLen, 8);
    return {
      paramCount: params,
      estimatedMemoryBytes: vramBytes,
    };
  }, [nLayer, dModel, ctxLen, actualVocabSize, shapeError]);

  const canSave = !shapeError && name.trim().length > 0;

  const create = useMutation({
    mutationFn: () => {
      const payload: ModelCreate = {
        name: name.trim(),
        description: description.trim(),
        preset: preset === 'custom' ? null : preset,
        dataset_id: selectedDatasetId || undefined,
        n_layer: nLayer,
        d_model: dModel,
        n_head: nHead,
        ctx_len: ctxLen,
        vocab_size: actualVocabSize,
        dropout: 0.0,
      };
      return api.createModel(payload);
    },
    onSuccess: () => {
      setName('');
      setDescription('');
      setDeleteError(null);
      void queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });

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
  const visibleModels = savedModels.filter((m: Model) =>
    `${m.name} ${m.description} ${m.preset ?? ''}`
      .toLowerCase()
      .includes(search.trim().toLowerCase()),
  );

  return (
    <>
      <div className="overview-heading page-intro">
        <div>
          <h1 className="h1">Model sizes</h1>
          <p className="lede">
            Every size the model can be built at, how big it is, and how much graphics card memory
            training it takes on this computer.
          </p>
        </div>
        <ExplainToggle checked={explain} onChange={setExplain} />
      </div>

      {explain ? (
        <div className="help-panel">
          <p>
            <strong>Blocks (layers)</strong>: Number of transformer blocks. Valid range is 1 to 24.
          </p>
          <p>
            <strong>Width (d_model)</strong>: Hidden vector size. Must divide evenly by attention
            heads.
          </p>
          <p>
            <strong>Vocabulary</strong>: Obtained from prepared datasets. Token count directly sizes
            the embedding table and prediction head.
          </p>
          <p>
            <strong>Memory figures are estimates</strong>: Calculated using ~16 bytes/parameter
            (weights, gradients, and Adam optimizer states) plus activation memory for a batch of 8
            sequences. Figures are estimates based on architectural formulas.
          </p>
        </div>
      ) : null}

      {/* GPU Details Card (Sprint 7) */}
      <Card>
        <CardHead
          title="This computer's graphics card"
          end={
            <button
              className="btn btn-sm"
              type="button"
              disabled={machine.isFetching}
              onClick={() => void machine.refetch()}
            >
              Check again
            </button>
          }
        />
        <div className="card-body stack">
          {machine.error ? (
            <ErrorBox message={machine.error.message} onRetry={() => void machine.refetch()} />
          ) : machine.isPending ? (
            <p role="status">Checking for a graphics card…</p>
          ) : (
            <GpuDetails seen={machine.data} />
          )}
        </div>
      </Card>

      {/* Preset Model Sizes Card (Sprint 7) */}
      <Card>
        <CardHead
          title="Model sizes"
          end={
            <label className="field">
              <span className="label">Sequences per step</span>
              <select
                className="input"
                aria-label="Sequences per step"
                value={sequences}
                onChange={(event) => setSequences(Number(event.target.value))}
              >
                {STEPS.map((step) => (
                  <option key={step} value={step}>
                    {step}
                  </option>
                ))}
              </select>
            </label>
          }
        />
        {presets.error ? (
          <div className="card-body">
            <ErrorBox message={presets.error.message} onRetry={() => void presets.refetch()} />
          </div>
        ) : presets.isPending ? (
          <p className="card-body" role="status">
            Loading model sizes…
          </p>
        ) : (
          <>
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Size</th>
                    <th className="num">Parameters</th>
                    <th className="num">Blocks</th>
                    <th className="num">Width</th>
                    <th className="num">Heads</th>
                    <th className="num">Context</th>
                    <th className="num">Memory to train</th>
                    {card ? <th>On {card.name}</th> : null}
                  </tr>
                </thead>
                <tbody>
                  {presets.data.map((p) => (
                    <SizeRow
                      key={p.name}
                      preset={p}
                      sequences={sequences}
                      cardBytes={card?.memory_bytes}
                    />
                  ))}
                </tbody>
              </table>
            </div>
            <p className="card-body hint">
              Memory is an estimate for training every weight with a vocabulary of{' '}
              {presets.data[0]?.vocab_size.toLocaleString() ?? 0} word pieces. Fewer sequences per
              step need less memory and train more slowly.
            </p>
          </>
        )}
      </Card>

      {/* Create & Save Model Builder Card (SP8.9) */}
      <div className="preparation-grid">
        <Card>
          <CardHead title="Create a model architecture" />
          <form
            className="card-body stack"
            onSubmit={(event) => {
              event.preventDefault();
              if (canSave && !create.isPending) create.mutate();
            }}
          >
            <Field
              label="Name"
              value={name}
              onChange={setName}
              maxLength={64}
              placeholder="e.g. tiny-gpt-nano"
              disabled={create.isPending}
            />

            <Field
              label="Description (optional)"
              value={description}
              onChange={setDescription}
              maxLength={255}
              placeholder="e.g. 4-layer model trained on math corpus"
              disabled={create.isPending}
            />

            <label className="field">
              <span className="label">Vocabulary source</span>
              <select
                className="input"
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                disabled={create.isPending}
              >
                <option value="">Default (16,384 tokens)</option>
                {vocabs.map((v) => (
                  <option key={v.dataset_id} value={v.dataset_id}>
                    {v.dataset_name} ({v.vocab_size.toLocaleString()} tokens)
                  </option>
                ))}
              </select>
              <span className="field-hint">
                Vocabulary size is taken directly from prepared token snapshots.
              </span>
            </label>

            <label className="field">
              <span className="label">Architecture preset</span>
              <select
                className="input"
                value={preset}
                onChange={(e) =>
                  handlePresetChange(e.target.value as 'nano' | 'micro' | 'small' | 'custom')
                }
                disabled={create.isPending}
              >
                <option value="nano">Nano (4 layers, 128 width, 4 heads, 512 context)</option>
                <option value="micro">Micro (6 layers, 256 width, 8 heads, 512 context)</option>
                <option value="small">Small (8 layers, 512 width, 8 heads, 512 context)</option>
                <option value="custom">Custom shape…</option>
              </select>
            </label>

            {preset === 'custom' ? (
              <div className="grid grid-2">
                <label className="field">
                  <span className="label">Blocks (n_layer)</span>
                  <input
                    type="number"
                    className="input"
                    min={1}
                    max={24}
                    value={nLayer}
                    onChange={(e) => setNLayer(Number(e.target.value))}
                    disabled={create.isPending}
                  />
                </label>
                <label className="field">
                  <span className="label">Model width (d_model)</span>
                  <input
                    type="number"
                    className="input"
                    min={1}
                    max={768}
                    step={1}
                    value={dModel}
                    onChange={(e) => setDModel(Number(e.target.value))}
                    disabled={create.isPending}
                  />
                </label>
                <label className="field">
                  <span className="label">Attention heads (n_head)</span>
                  <input
                    type="number"
                    className="input"
                    min={1}
                    max={32}
                    value={nHead}
                    onChange={(e) => setNHead(Number(e.target.value))}
                    disabled={create.isPending}
                  />
                </label>
                <label className="field">
                  <span className="label">Context length (tokens)</span>
                  <select
                    className="input"
                    value={ctxLen}
                    onChange={(e) => setCtxLen(Number(e.target.value))}
                    disabled={create.isPending}
                  >
                    <option value={256}>256 tokens</option>
                    <option value={512}>512 tokens</option>
                    <option value={1024}>1024 tokens</option>
                  </select>
                </label>
              </div>
            ) : null}

            {shapeError ? <Notice tone="bad">{shapeError}</Notice> : null}

            {create.error ? <Notice tone="bad">{create.error.message}</Notice> : null}

            <div>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!canSave || create.isPending}
              >
                {create.isPending ? 'Saving model…' : 'Save model'}
              </button>
            </div>
          </form>
        </Card>

        {/* Live Architecture Preview Card */}
        <Card>
          <CardHead title="Architecture preview" />
          <div className="card-body stack">
            <dl className="preview-counts">
              <div>
                <dt>Parameters</dt>
                <dd>
                  {preview ? preview.paramCount.toLocaleString() : '—'}
                  {preview ? (
                    <span className="muted" style={{ fontSize: '0.85em', marginLeft: 6 }}>
                      ({compactNumber(preview.paramCount)})
                    </span>
                  ) : null}
                </dd>
              </div>
              <div>
                <dt>Estimated training memory</dt>
                <dd>
                  {preview ? bytesToSize(preview.estimatedMemoryBytes) : '—'}
                  <span className="badge" style={{ marginLeft: 6 }}>
                    Estimate
                  </span>
                </dd>
              </div>
              <div>
                <dt>Vocabulary size</dt>
                <dd>{actualVocabSize.toLocaleString()} tokens</dd>
              </div>
              <div>
                <dt>Shape</dt>
                <dd>
                  {nLayer} layers · {dModel} width · {nHead} heads · {ctxLen} ctx
                </dd>
              </div>
            </dl>

            <p className="hint">
              Estimated memory covers weights, gradients, optimizer states (16 bytes/param) and
              activations for 8 sequences per step. Figures are estimates based on architectural
              formulas.
            </p>

            {shapeError ? (
              <div className="notice notice-bad">
                <strong>Invalid shape:</strong> {shapeError}
              </div>
            ) : (
              <div className="notice notice-ok">
                Shape satisfies architectural validation constraints.
              </div>
            )}
          </div>
        </Card>
      </div>

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
                  : 'Configure and save a model above to use for pre-training.'
              }
            />
          ) : (
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Preset</th>
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
                    const vocabName =
                      typeof m.config?.dataset_name === 'string'
                        ? m.config.dataset_name
                        : 'Default vocab';
                    return (
                      <tr key={m.id}>
                        <td>
                          <strong>{m.name}</strong>
                          {m.description ? (
                            <p className="muted" style={{ margin: '2px 0 0', fontSize: '0.85em' }}>
                              {m.description}
                            </p>
                          ) : null}
                        </td>
                        <td>
                          <Chip tone={m.preset ? 'ok' : 'idle'}>{m.preset ?? 'custom'}</Chip>
                        </td>
                        <td>
                          <span title={`Vocab size: ${m.vocab_size.toLocaleString()}`}>
                            {vocabName}
                          </span>
                        </td>
                        <td className="num">{compactNumber(m.param_count)}</td>
                        <td className="num">
                          {m.estimated_memory_bytes ? bytesToSize(m.estimated_memory_bytes) : '—'}
                        </td>
                        <td>
                          {m.n_layer}L · {m.d_model}W · {m.n_head}H · {m.ctx_len}C
                        </td>
                        <td>{relativeDate(m.created_at)}</td>
                        <td>
                          <ConfirmButton
                            label="Delete"
                            confirmLabel="Delete model"
                            onConfirm={() => remove.mutate(m.id)}
                            disabled={remove.isPending}
                          />
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

function SizeRow({
  preset,
  sequences,
  cardBytes,
}: {
  preset: Preset;
  sequences: number;
  cardBytes: number | undefined;
}) {
  const needed = preset.memory.find((row) => row.sequences_per_step === sequences)?.bytes;
  return (
    <tr>
      <td className="name">{preset.name}</td>
      <td className="num">{compactNumber(preset.parameters)}</td>
      <td className="num">{preset.n_layer}</td>
      <td className="num">{preset.d_model}</td>
      <td className="num">{preset.n_head}</td>
      <td className="num">{preset.ctx_len.toLocaleString()}</td>
      <td className="num">{needed === undefined ? 'None' : gigabytes(needed)}</td>
      {cardBytes === undefined ? null : (
        <td>
          {needed === undefined ? null : needed <= cardBytes ? (
            <Chip tone="ok">Fits</Chip>
          ) : (
            <Chip tone="bad">Too big</Chip>
          )}
        </td>
      )}
    </tr>
  );
}

function GpuDetails({ seen }: { seen: GpuCheck }) {
  return (
    <>
      {seen.cards.length === 0 ? (
        <>
          <p>
            No NVIDIA graphics card was found. Training would run on the processor, which is much
            slower.
          </p>
          {seen.tool_said ? (
            <p className="muted">The card's driver said: {seen.tool_said}</p>
          ) : null}
        </>
      ) : (
        seen.cards.map((card) => (
          <dl className="preview-counts" key={card.index}>
            <div>
              <dt>{seen.cards.length > 1 ? `Card ${card.index}` : 'Card'}</dt>
              <dd>{card.name}</dd>
            </div>
            <div>
              <dt>Memory</dt>
              <dd>{gigabytes(card.memory_bytes)}</dd>
            </div>
            <div>
              <dt>Driver</dt>
              <dd>{card.driver}</dd>
            </div>
            <div>
              <dt>Compute capability</dt>
              <dd>{card.compute_capability ?? 'Unknown'}</dd>
            </div>
          </dl>
        ))
      )}
      <TorchNote seen={seen} />
    </>
  );
}

function TorchNote({ seen }: { seen: GpuCheck }) {
  if (seen.torch === 'missing') {
    return (
      <Notice tone="warn">
        PyTorch isn't installed, so this computer can't train yet. Add the train extra from the
        README.
      </Notice>
    );
  }
  if (seen.torch === 'cpu' && seen.cards.length > 0) {
    return (
      <Notice tone="warn">
        PyTorch {seen.torch_version} is installed without graphics card support, so training would
        use the processor. Reinstall PyTorch with CUDA from the README.
      </Notice>
    );
  }
  if (seen.torch === 'cuda') {
    return (
      <p className="muted">
        PyTorch {seen.torch_version} with CUDA {seen.torch_cuda}
      </p>
    );
  }
  return seen.torch_version ? <p className="muted">PyTorch {seen.torch_version}</p> : null;
}
