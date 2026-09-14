import { describe, expect, it } from 'vitest';

import { compactNumber, plural, relativeDate } from './format';

describe('compactNumber', () => {
  it('leaves small numbers alone', () => {
    expect(compactNumber(0)).toBe('0');
    expect(compactNumber(999)).toBe('999');
  });

  it('shortens big numbers', () => {
    expect(compactNumber(1_000)).toBe('1K');
    expect(compactNumber(1_234)).toBe('1.2K');
    expect(compactNumber(4_500_000)).toBe('4.5M');
    expect(compactNumber(2_000_000_000)).toBe('2B');
  });

  it('handles negative and invalid numbers', () => {
    expect(compactNumber(-1_500)).toBe('-1.5K');
    expect(compactNumber(NaN)).toBe('None');
  });
});

describe('plural', () => {
  it('uses the right form', () => {
    expect(plural(1, 'document')).toBe('1 document');
    expect(plural(2, 'document')).toBe('2 documents');
    expect(plural(0, 'document')).toBe('0 documents');
  });

  it('supports an irregular plural', () => {
    expect(plural(3, 'index', 'indices')).toBe('3 indices');
  });
});

describe('relativeDate', () => {
  const now = new Date('2026-08-25T12:00:00Z');

  it('describes recent times', () => {
    expect(relativeDate('2026-08-25T11:59:30Z', now)).toBe('just now');
    expect(relativeDate('2026-08-25T11:30:00Z', now)).toBe('30m ago');
    expect(relativeDate('2026-08-25T06:00:00Z', now)).toBe('6h ago');
    expect(relativeDate('2026-08-20T12:00:00Z', now)).toBe('5d ago');
  });

  it('uses a regular date for old times', () => {
    expect(relativeDate('2020-01-01T12:00:00Z', now)).not.toContain('ago');
  });

  it('handles an invalid date', () => {
    expect(relativeDate('not a date', now)).toBe('None');
  });
});