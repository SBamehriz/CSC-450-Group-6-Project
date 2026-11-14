import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

import { api, type GpuCheck, type SpeedTest, type SpeedTestCreate } from '../api/client';
import JobStatus, { useJob } from '../components/JobStatus';
import { Card, CardHead, Empty, ErrorBox, Notice } from '../components/ui';
import { APPEARANCES, useAppearance } from '../lib/appearance';
import { compactNumber, gigabytes, relativeDate } from '../lib/format';
import { jobBusy } from '../lib/measure';

const SEQUENCES = [1, 2, 4, 8, 16];

function shapeWords(test: SpeedTest): string {
  if (test.preset) return test.preset;
  const shape = test.shape as Record<string, number> | null;
  return shape ? `${shape.n_layer} blocks, ${shape.d_model} wide` : 'A standard small model';
}

export default function ComputerPage() {
  const machine = useQuery({ queryKey: ['gpu'], queryFn: api.checkGpu });

  return (
    <>
      <div className="page-intro">
        <h1 className="h1">Settings</h1>
        <p className="lede">
          How Forge looks, and what this computer can train: its graphics card and how fast it
          really is.
        </p>
      </div>

      <AppearanceCard />

      <Card>
        <CardHead
          title="Graphics card"
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

      <SpeedTestCard />
    </>
  );
}

function AppearanceCard() {
  const [appearance, setAppearance] = useAppearance();
  return (
    <Card>
      <CardHead title="Appearance" />
      <div className="card-body">
        <div className="seg" role="group" aria-label="Appearance">
          {APPEARANCES.map((one) => (
            <button
              key={one.value}
              type="button"
              className={appearance === one.value ? 'on' : undefined}
              aria-pressed={appearance === one.value}
              onClick={() => setAppearance(one.value)}
            >
              {one.label}
            </button>
          ))}
        </div>
      </div>
    </Card>
  );
}

function SpeedTestCard() {
  const queryClient = useQueryClient();
  const tests = useQuery({ queryKey: ['speedTests'], queryFn: api.listSpeedTests });
  const [sequences, setSequences] = useState(4);
  const [device, setDevice] = useState<SpeedTestCreate['device']>('auto');
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, () => {
    void queryClient.invalidateQueries({ queryKey: ['speedTests'] });
    void queryClient.invalidateQueries({ queryKey: ['estimate'] });
  });
  const start = useMutation({
    mutationFn: () => api.startSpeedTest({ sequences_per_step: sequences, device }),
    onSuccess: (started) => setJobId(started.id),
  });
  const busy = start.isPending || jobBusy(job.data);
  const rows = tests.data ?? [];

  return (
    <Card>
      <CardHead title="Speed test" />
      <form
        className="card-body stack"
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy) start.mutate();
        }}
      >
        <p className="muted" style={{ margin: 0 }}>
          Trains a standard small model for a few steps on random text and times it, so the hours on
          the Train page come from this computer and not a guess. It takes under two minutes.
        </p>
        <div className="dataset-form">
          <label className="field">
            <span className="label">Sequences per step</span>
            <select
              className="input"
              aria-label="Sequences per step"
              value={sequences}
              onChange={(event) => setSequences(Number(event.target.value))}
            >
              {SEQUENCES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="label">Where it runs</span>
            <select
              className="input"
              aria-label="Where it runs"
              value={device}
              onChange={(event) => setDevice(event.target.value as SpeedTestCreate['device'])}
            >
              <option value="auto">The graphics card if there is one</option>
              <option value="cuda">The graphics card</option>
              <option value="cpu">The CPU</option>
            </select>
          </label>
        </div>
        {start.error ? <Notice tone="bad">{start.error.message}</Notice> : null}
        <JobStatus job={job.data} doneText="Speed test finished." />
        <div>
          <button type="submit" className="btn btn-primary" disabled={busy}>
            {busy ? 'Timing…' : 'Run speed test'}
          </button>
        </div>
      </form>
      {tests.error ? (
        <div className="card-body">
          <ErrorBox message={tests.error.message} onRetry={() => void tests.refetch()} />
        </div>
      ) : tests.isPending ? null : !rows.length ? (
        <div className="card-body">
          <Empty title="No speed tests yet" hint="Run one above to see real hours." />
        </div>
      ) : (
        <div className="scroll-x">
          <table className="table">
            <thead>
              <tr>
                <th>Size</th>
                <th className="num">Sequences per step</th>
                <th>Where</th>
                <th className="num">Word pieces a second</th>
                <th className="num">Peak memory</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((test) => (
                <tr key={test.id}>
                  <td>{shapeWords(test)}</td>
                  <td className="num">{test.sequences_per_step}</td>
                  <td>{test.device_name}</td>
                  <td className="num">{compactNumber(Math.round(test.tokens_per_sec))}</td>
                  <td className="num">
                    {test.peak_memory_bytes === null
                      ? 'Not measured'
                      : gigabytes(test.peak_memory_bytes)}
                  </td>
                  <td className="muted">{relativeDate(test.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export function GpuDetails({ seen }: { seen: GpuCheck }) {
  return (
    <>
      {seen.cards.length === 0 ? (
        <>
          <p>
            No NVIDIA graphics card was found. Training would run on the processor, which is much
            slower.
          </p>
          {seen.tool_said ? (
            <p className="muted">The card&apos;s driver said: {seen.tool_said}</p>
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
              <dt>Free right now</dt>
              <dd>{gigabytes(card.free_bytes)}</dd>
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
        PyTorch isn&apos;t installed, so this computer can&apos;t train yet. Add the train extra
        from the README.
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
