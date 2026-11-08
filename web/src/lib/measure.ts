import type { Estimate, Job } from '../api/client';
import { compactNumber, gigabytes } from './format';
import { duration } from './training';

export function jobProgress(job: Job): { completed: number; total: number; message: string } {
  const value = job.progress as { completed?: number; total?: number; message?: string };
  return {
    completed: value.completed ?? 0,
    total: value.total ?? 0,
    message: value.message ?? '',
  };
}

export function jobBusy(job: Job | undefined): boolean {
  return job?.status === 'pending' || job?.status === 'running';
}

export function speedWords(estimate: Estimate): string {
  const where = estimate.device_name ? ` (${estimate.device_name})` : '';
  if (estimate.speed_source === 'measured') {
    return `Speed measured on this computer${where}.`;
  }
  if (estimate.speed_source === 'scaled') {
    return `Speed scaled from a speed test of another size${where}. A speed test of this size gives a closer number.`;
  }
  return 'Run a speed test to see hours.';
}

export function timeWords(estimate: Estimate): string | null {
  if (estimate.hours === null) return null;
  const pace = estimate.tokens_per_sec
    ? ` at about ${compactNumber(Math.round(estimate.tokens_per_sec))} word pieces a second`
    : '';
  return `About ${duration(estimate.hours * 3600)} to train ${compactNumber(estimate.train_tokens)} word pieces${pace}.`;
}

export function memoryWords(estimate: Estimate): string {
  const needed = gigabytes(estimate.memory_bytes);
  if (estimate.fits === null || estimate.free_bytes === null) {
    return `Needs about ${needed} of graphics card memory. No graphics card was found, so it would train on the processor.`;
  }
  const free = gigabytes(estimate.free_bytes);
  return estimate.fits
    ? `Needs about ${needed}, and ${free} is free on the graphics card right now. It fits.`
    : `Needs about ${needed}, and only ${free} is free on the graphics card right now.`;
}

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
