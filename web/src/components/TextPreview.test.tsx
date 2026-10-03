// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { open } from '../test-utils';
import TextPreview from './TextPreview';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('reads text beyond the first window and can return to it', async () => {
  vi.spyOn(api, 'documentText').mockImplementation(async (_id, offset = 0) => ({
    text: offset ? 'Last window.' : 'First window.',
    offset,
    length: offset ? 500 : 4000,
    total: 4500,
  }));
  open(<TextPreview documentId="doc-1" />);
  expect(await screen.findByText('First window.')).toBeTruthy();
  expect(
    (screen.getByRole('button', { name: 'Previous text' }) as HTMLButtonElement).disabled,
  ).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: 'Next text' }));
  expect(await screen.findByText('Last window.')).toBeTruthy();
  expect(api.documentText).toHaveBeenLastCalledWith('doc-1', 4000, 4000);
  expect((screen.getByRole('button', { name: 'Next text' }) as HTMLButtonElement).disabled).toBe(
    true,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Previous text' }));
  expect(await screen.findByText('First window.')).toBeTruthy();
});

it('shows a failed text request and retries it', async () => {
  vi.spyOn(api, 'documentText')
    .mockRejectedValueOnce(new Error('Text unavailable.'))
    .mockResolvedValue({ text: 'Recovered text.', offset: 0, length: 15, total: 15 });
  open(<TextPreview documentId="doc-1" />);
  expect(await screen.findByText('Text unavailable.')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  await waitFor(() => expect(screen.getByText('Recovered text.')).toBeTruthy());
});
