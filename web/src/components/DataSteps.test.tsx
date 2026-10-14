// @vitest-environment jsdom
import { cleanup, screen } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';

import { bucket, open, stats } from '../test-utils';
import DataSteps from './DataSteps';

afterEach(cleanup);
it('needs two parsed documents in the same collection and does not claim manual review', () => {
  const first = { ...bucket, stats: { ...bucket.stats, parsed_documents: 1 } };
  const second = { ...first, id: 'second', name: 'Second' };
  open(
    <DataSteps
      buckets={[first, second]}
      stats={{ ...stats, parsed_documents: 2, ready_snapshots: 0 }}
    />,
  );
  expect(screen.queryByRole('link', { name: 'Create a snapshot →' })).toBeNull();
  expect(screen.getByText('Needs more text')).toBeTruthy();
  expect(screen.getByText('Ready to review')).toBeTruthy();
  expect(screen.queryByText('Review complete')).toBeNull();
});
it('links to an eligible source and leaves an oversized collection out', () => {
  const large = { ...bucket, id: 'large', stats: { ...bucket.stats, parsed_documents: 1001 } };
  open(<DataSteps buckets={[large, bucket]} stats={stats} />);
  expect(screen.getByRole('link', { name: 'Create a snapshot →' }).getAttribute('href')).toBe(
    '/datasets?bucket=bucket-1',
  );
});
