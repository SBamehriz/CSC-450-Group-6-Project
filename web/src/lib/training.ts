import type { DatasetBuild, Model, Run, Tokenizer, TrainSettings } from '../api/client';

export const DEFAULT_SETTINGS: TrainSettings = {
  batch_size: 16,
  micro_batch_size: 0,
  max_steps: 1000,
  learning_rate: 6e-4,
  warmup_steps: -1,
  weight_decay: 0.1,
  grad_clip: 1.0,
  eval_every: 100,
  eval_tokens: 100_000,
  save_every: 200,
};

export function usableBuilds(
  model: Model | undefined,
  builds: DatasetBuild[],
  vocabularies: Tokenizer[],
): DatasetBuild[] {
  if (!model) return [];
  const size = new Map(vocabularies.map((v) => [v.id, v.vocab_size]));
  return builds.filter((build) => {
    if (build.status !== 'ready') return false;
    if (model.tokenizer_id) return build.tokenizer_id === model.tokenizer_id;
    return size.get(build.tokenizer_id) === model.vocab_size;
  });
}

export function trainingTokens(build: DatasetBuild | undefined): number | null {
  const tokens = build?.stats?.train?.tokens;
  return typeof tokens === 'number' ? tokens : null;
}

export function isActive(run: Run): boolean {
  return run.job_status === 'pending' || run.job_status === 'running' || run.status === 'running';
}

export function secondsLeft(run: Run): number | null {
  if (!run.tokens_per_sec || run.current_step <= 0 || run.tokens_seen <= 0) return null;
  const perStep = run.tokens_seen / run.current_step;
  const steps = Math.max(0, run.max_steps - run.current_step);
  return (steps * perStep) / run.tokens_per_sec;
}

export function duration(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} h ${rest} min` : `${hours} h`;
}

export const STATUS_WORDS: Record<string, string> = {
  pending: 'Waiting',
  running: 'Training',
  completed: 'Finished',
  stopped: 'Stopped',
  failed: 'Stopped with a problem',
};
