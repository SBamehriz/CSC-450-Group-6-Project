// @vitest-environment jsdom
import { cleanup, fireEvent, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { open } from '../test-utils';
import Layout from './Layout';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
it('keeps Collections active on a document route and links the wordmark home', async () => {
  vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', database: 'ok' });
  open(<Layout>Files</Layout>, '/data/bucket-1?open=doc-1');
  await screen.findByText('Connected');
  expect(screen.getByRole('link', { name: 'Collections' }).getAttribute('aria-current')).toBe(
    'page',
  );
  expect(screen.getByRole('link', { name: 'Forge home' }).getAttribute('href')).toBe('/');
  expect(screen.getByRole('link', { name: 'Datasets' }).getAttribute('href')).toBe('/datasets');
});
it('reconnects after a failed health check', async () => {
  vi.spyOn(api, 'health')
    .mockRejectedValueOnce(new Error('Offline'))
    .mockResolvedValue({ status: 'ok', database: 'ok' });
  open(<Layout>Files</Layout>);
  await screen.findByText('Server unavailable');
  fireEvent.click(screen.getByRole('button', { name: 'Reconnect' }));
  await screen.findByText('Connected');
  expect(screen.queryByRole('button', { name: 'Reconnect' })).toBeNull();
});
