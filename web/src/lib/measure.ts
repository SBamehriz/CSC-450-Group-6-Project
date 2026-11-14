import type { Job } from '../api/client';

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

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
