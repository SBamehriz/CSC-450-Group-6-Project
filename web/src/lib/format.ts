export function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return 'None';
  const abs = Math.abs(value);
  if (abs < 1_000) return String(value);
  if (abs < 1_000_000) return `${trim(value / 1_000)}K`;
  if (abs < 1_000_000_000) return `${trim(value / 1_000_000)}M`;
  return `${trim(value / 1_000_000_000)}B`;
}

function trim(value: number): string {
  return value.toFixed(1).replace(/\.0$/, '');
}

export function plural(count: number, word: string, pluralForm?: string): string {
  const label = count === 1 ? word : (pluralForm ?? `${word}s`);
  return `${count.toLocaleString()} ${label}`;
}

export function relativeDate(iso: string, now: Date = new Date()): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return 'None';
  const seconds = Math.round((now.getTime() - then.getTime()) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return then.toLocaleDateString();
}