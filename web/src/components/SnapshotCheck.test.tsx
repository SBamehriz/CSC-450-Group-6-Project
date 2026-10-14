// @vitest-environment jsdom
import { cleanup, fireEvent, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { open } from '../test-utils';
import SnapshotCheck from './SnapshotCheck';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('checks the saved ZIP and shows the result', async () => {
  const verify = vi.spyOn(api, 'verifyDataset').mockResolvedValue({
    valid: true,
    message: 'Both files passed.',
    train_documents: 3,
    validation_documents: 1,
  });
  open(<SnapshotCheck datasetId="saved" />);
  fireEvent.click(screen.getByRole('button', { name: 'Check snapshot' }));
  expect(await screen.findByText('Both files passed.')).toBeTruthy();
  expect(verify).toHaveBeenCalledWith('saved');
});

it('shows a damaged snapshot response', async () => {
  vi.spyOn(api, 'verifyDataset').mockResolvedValue({
    valid: false,
    message: 'The snapshot has changed.',
    train_documents: 0,
    validation_documents: 0,
  });
  open(<SnapshotCheck datasetId="saved" />);
  fireEvent.click(screen.getByRole('button', { name: 'Check snapshot' }));
  expect(await screen.findByText('The snapshot has changed.')).toBeTruthy();
});
