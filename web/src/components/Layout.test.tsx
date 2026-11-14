// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import { open } from '../test-utils';
import Layout from './Layout';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});
it('has the seven places, keeps Data lit on a document route and links the name home', async () => {
  vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', database: 'ok' });
  open(<Layout>Files</Layout>, '/data/bucket-1?open=doc-1');
  const nav = screen.getByRole('navigation', { name: 'Main navigation' });
  expect(
    Array.from(nav.querySelectorAll('a')).map((link) => [
      link.textContent,
      link.getAttribute('href'),
    ]),
  ).toEqual([
    ['Home', '/'],
    ['Data', '/data'],
    ['Train', '/train'],
    ['Models', '/models'],
    ['Chat', '/chat'],
    ['Compare', '/compare'],
  ]);
  expect(screen.getByRole('link', { name: 'Settings' }).getAttribute('href')).toBe('/settings');
  expect(screen.getByRole('link', { name: 'Data' }).getAttribute('aria-current')).toBe('page');
  expect(screen.getByRole('link', { name: 'Forge home' }).getAttribute('href')).toBe('/');
  expect(screen.getByRole('link', { name: 'Skip to content' }).getAttribute('href')).toBe(
    '#main-content',
  );
  // a healthy server says nothing
  await waitFor(() => expect(api.health).toHaveBeenCalled());
  expect(screen.queryByText('Server unavailable')).toBeNull();
});
it('keeps Train lit on a run page', () => {
  vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', database: 'ok' });
  open(<Layout>Run</Layout>, '/runs/run-1');
  expect(screen.getByRole('link', { name: 'Train' }).getAttribute('aria-current')).toBe('page');
});
it('folds to icons and remembers it', () => {
  vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', database: 'ok' });
  open(<Layout>Files</Layout>);
  fireEvent.click(screen.getByRole('button', { name: 'Fold the side panel' }));
  expect(localStorage.getItem('forge.sidebar-folded')).toBe('yes');
  expect(screen.getByRole('link', { name: 'Models' }).getAttribute('title')).toBe('Models');
  cleanup();
  open(<Layout>Files</Layout>);
  expect(screen.getByRole('button', { name: 'Unfold the side panel' })).toBeTruthy();
});
it('opens the menu over the page and closes it with Escape', () => {
  vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', database: 'ok' });
  open(<Layout>Files</Layout>);
  const menu = screen.getByRole('button', { name: 'Open the menu' });
  fireEvent.click(menu);
  expect(menu.getAttribute('aria-expanded')).toBe('true');
  expect(document.getElementById('main-content')?.hasAttribute('inert')).toBe(true);
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(menu.getAttribute('aria-expanded')).toBe('false');
  expect(document.activeElement).toBe(menu);
});
it('reconnects after a failed health check', async () => {
  vi.spyOn(api, 'health')
    .mockRejectedValueOnce(new Error('Offline'))
    .mockResolvedValue({ status: 'ok', database: 'ok' });
  open(<Layout>Files</Layout>);
  await screen.findByText('Server unavailable');
  fireEvent.click(screen.getByRole('button', { name: 'Reconnect' }));
  await waitFor(() => expect(screen.queryByText('Server unavailable')).toBeNull());
  expect(screen.queryByRole('button', { name: 'Reconnect' })).toBeNull();
});
