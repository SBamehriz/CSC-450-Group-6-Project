// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { ApiError, api, type Model } from '../api/client';
import { open } from '../test-utils';
import ModelsPage from './ModelsPage';

const mockModel: Model = {
  id: 'model-1',
  source: 'forge',
  name: 'nano-science',
  description: 'Nano model for science papers',
  preset: 'nano',
  n_layer: 4,
  d_model: 128,
  n_head: 4,
  ctx_len: 512,
  vocab_size: 4096,
  dropout: 0.0,
  param_count: 920448,
  config: {
    dataset_id: 'vocab-1',
    dataset_name: 'Science papers',
    vocab_size: 4096,
  },
  estimated_memory_bytes: 15400000,
  chat: false,
  ready: true,
  created_at: '2026-10-23T12:00:00Z',
};

beforeEach(() => {
  vi.spyOn(api, 'listAllModels').mockResolvedValue({ items: [mockModel], total: 1 });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function showPage() {
  open(<ModelsPage />);
}

it('shows why the hub refused a model', async () => {
  vi.spyOn(api, 'importModel').mockRejectedValue(
    new ApiError(409, 'import_refused', 'This model only has pickle weights.'),
  );
  open(<ModelsPage />);
  fireEvent.change(await screen.findByLabelText('Model on the hub'), {
    target: { value: 'someone/old-model' },
  });
  fireEvent.click(screen.getByLabelText('Run it on the CPU (slower)'));
  fireEvent.click(screen.getByRole('button', { name: 'Import model' }));
  expect(await screen.findByText('This model only has pickle weights.')).toBeTruthy();
  expect(api.importModel).toHaveBeenCalledWith({
    repo_id: 'someone/old-model',
    revision: 'main',
    name: null,
    on_cpu: true,
  });
});

it('labels an imported model with its licence and version', async () => {
  vi.mocked(api.listAllModels).mockResolvedValue({
    items: [
      {
        ...mockModel,
        id: 'model-hub',
        name: 'gpt2',
        source: 'hub',
        preset: null,
        config: {
          hub: { repo_id: 'openai-community/gpt2', commit: 'c0ffee12345', license: 'mit' },
        },
      },
    ],
    total: 1,
  });
  open(<ModelsPage />);
  expect(await screen.findByText('gpt2')).toBeTruthy();
  expect(screen.getByText('Imported')).toBeTruthy();
  expect(screen.getByText('Licence mit, version c0ffee1')).toBeTruthy();
});

it('makes models on the Train page, not here', async () => {
  showPage();
  expect(await screen.findByText('nano-science')).toBeTruthy();
  expect(screen.queryByText('Create a model architecture')).toBeNull();
  expect(screen.queryByLabelText('Sequences per step')).toBeNull();
  expect(screen.getByRole('link', { name: 'Train' }).getAttribute('href')).toBe('/train');
});

it('shows a fine-tune under its parent, with chat, fine-tune and compare links', async () => {
  const child: Model = {
    ...mockModel,
    id: 'model-2',
    name: 'science-chat',
    parent_id: 'model-1',
    parent_name: 'nano-science',
    chat: true,
  };
  const later: Model = { ...mockModel, id: 'model-3', name: 'not-yet', ready: false };
  vi.mocked(api.listAllModels).mockResolvedValue({
    items: [child, later, mockModel],
    total: 3,
  });
  showPage();
  await screen.findByText('science-chat');
  const rows = screen.getAllByRole('row').slice(1);
  const names = rows.map((row) => row.querySelector('strong')?.textContent);
  expect(names).toEqual(['not-yet', 'nano-science', 'science-chat']);
  const childRow = within(rows[2]);
  expect(childRow.getByText('chat')).toBeTruthy();
  expect(childRow.getByText('Fine-tuned from nano-science')).toBeTruthy();
  expect(childRow.getByRole('link', { name: 'Chat' }).getAttribute('href')).toBe(
    '/chat?model=model-2&with=model-1',
  );
  const parentRow = within(rows[1]);
  expect(parentRow.getByRole('link', { name: 'Chat' }).getAttribute('href')).toBe(
    '/chat?model=model-1',
  );
  expect(parentRow.getByRole('link', { name: 'Fine-tune' }).getAttribute('href')).toBe(
    '/train?from=model-1',
  );
  expect(parentRow.getByRole('link', { name: 'Compare' }).getAttribute('href')).toBe(
    '/compare?model=model-1',
  );
  const laterRow = within(rows[0]);
  expect(laterRow.getByText('Not trained yet')).toBeTruthy();
  expect(laterRow.queryByRole('link', { name: 'Chat' })).toBeNull();
  expect(laterRow.queryByRole('link', { name: 'Fine-tune' })).toBeNull();
});

it('renders saved models list and handles deletion', async () => {
  const deleteSpy = vi.spyOn(api, 'deleteModel').mockResolvedValue(undefined);

  showPage();
  expect(await screen.findByText('nano-science')).toBeTruthy();
  expect(screen.getByText('Science papers')).toBeTruthy();

  const deleteBtn = screen.getByRole('button', { name: 'Delete' });
  fireEvent.click(deleteBtn);

  const confirmBtn = await screen.findByRole('button', { name: 'Delete model' });
  fireEvent.click(confirmBtn);

  await waitFor(() => {
    expect(deleteSpy).toHaveBeenCalledWith('model-1');
  });
});
