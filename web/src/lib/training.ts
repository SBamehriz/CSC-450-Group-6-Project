import type { Run } from '../api/client';

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
