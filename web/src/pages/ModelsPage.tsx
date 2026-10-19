import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { api, type GpuCheck, type Preset } from '../api/client';
import { Card, CardHead, Chip, ErrorBox, Notice } from '../components/ui';
import { compactNumber, gigabytes } from '../lib/format';

const STEPS = [1, 2, 4, 8, 16];

export default function ModelsPage() {
  const presets = useQuery({ queryKey: ['presets'], queryFn: api.listPresets });
  const machine = useQuery({ queryKey: ['gpu'], queryFn: api.checkGpu });
  const [sequences, setSequences] = useState(8);
  const card = machine.data?.cards[0];

  return (
    <>
      <div className="page-intro">
        <h1 className="h1">Model sizes</h1>
        <p className="lede">
          Every size the model can be built at, how big it is, and how much graphics card memory
          training it takes on this computer.
        </p>
      </div>
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
                  {presets.data.map((preset) => (
                    <SizeRow
                      key={preset.name}
                      preset={preset}
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
              <dd>{card.compute_capability}</dd>
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
