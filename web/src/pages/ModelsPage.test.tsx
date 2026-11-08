// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import {
  ApiError,
  api,
  type GpuCheck,
  type Model,
  type Preset,
  type Vocabulary,
} from '../api/client';
import { open } from '../test-utils';
import ModelsPage from './ModelsPage';

const GB = 2 ** 30;

function preset(name: string, parameters: number, perSequence: number): Preset {
  return {
    name,
    n_layer: 4,
    d_model: 128,
    n_head: 4,
    ctx_len: 512,
    vocab_size: 16384,
    parameters,
    memory: [1, 2, 4, 8, 16].map((sequences) => ({
      sequences_per_step: sequences,
      bytes: sequences * perSequence,
    })),
  };
}

const sizes = [preset('nano', 2_956_032, 0.5 * GB), preset('135m', 133_864_704, 3 * GB)];

const oneCard: GpuCheck = {
  cards: [
    {
      index: 0,
      name: 'NVIDIA GeForce RTX 3090',
      memory_bytes: 24 * GB,
      free_bytes: 20 * GB,
      driver: '595.79',
      compute_capability: '8.6',
    },
  ],
  tool_said: null,
  torch: 'cuda',
  torch_version: '2.14.1+cu130',
  torch_cuda: '13.0',
};

const mockVocab: Vocabulary = {
  dataset_id: 'vocab-1',
  dataset_name: 'Science papers',
  vocab_size: 4096,
  requested_vocab_size: 4096,
  train_tokens: 50000,
  trained_on_documents: 12,
};

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
  created_at: '2026-10-23T12:00:00Z',
};

beforeEach(() => {
  vi.spyOn(api, 'listPresets').mockResolvedValue(sizes);
  vi.spyOn(api, 'checkGpu').mockResolvedValue(oneCard);
  vi.spyOn(api, 'listVocabularies').mockResolvedValue([mockVocab]);
  vi.spyOn(api, 'listAllModels').mockResolvedValue({ items: [mockModel], total: 1 });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function rowOf(name: string) {
  const cells = screen.getAllByRole('cell', { name });
  const cell = cells.find((c) => c.classList.contains('name')) ?? cells[0];
  return within(cell.closest('tr') as HTMLElement);
}

function showPage() {
  open(<ModelsPage />);
}

it('judges each size against the memory free on the card right now', async () => {
  open(<ModelsPage />);
  expect(
    await screen.findByRole('columnheader', { name: 'On NVIDIA GeForce RTX 3090' }),
  ).toBeTruthy();
  expect(
    screen.getByText(
      /Fits is judged against the 20.0 GB free right now on NVIDIA GeForce RTX 3090/,
    ),
  ).toBeTruthy();
  expect(rowOf('nano').getByText('4.0 GB')).toBeTruthy();
  expect(rowOf('nano').getByText('Fits')).toBeTruthy();
  expect(rowOf('135m').getByText('24.0 GB')).toBeTruthy();
  expect(rowOf('135m').getByText('Too big')).toBeTruthy();

  fireEvent.change(screen.getByLabelText('Sequences per step'), { target: { value: '2' } });
  expect(rowOf('135m').getByText('6.0 GB')).toBeTruthy();
  expect(rowOf('135m').getByText('Fits')).toBeTruthy();
});

it('judges nothing when there is no card', async () => {
  vi.mocked(api.checkGpu).mockResolvedValue({
    cards: [],
    tool_said: null,
    torch: 'missing',
    torch_version: null,
    torch_cuda: null,
  });
  open(<ModelsPage />);
  expect(await screen.findByText(/No graphics card was found/)).toBeTruthy();
  expect(await screen.findByRole('cell', { name: '135m' })).toBeTruthy();
  expect(screen.queryByText('Fits')).toBeNull();
  expect(screen.queryByText('Too big')).toBeNull();
});

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

it('displays parameter count and estimated memory before saving', async () => {
  showPage();
  expect(await screen.findByText('Create a model architecture')).toBeTruthy();
  expect(await screen.findByText('Architecture preview')).toBeTruthy();
  expect(await screen.findByText(/Figures are estimates/i)).toBeTruthy();
  const estimateBadges = await screen.findAllByText(/Estimate/i);
  expect(estimateBadges.length).toBeGreaterThan(0);
});

it('explains invalid shape in plain language', async () => {
  showPage();
  expect(await screen.findByText('Create a model architecture')).toBeTruthy();

  fireEvent.change(screen.getByLabelText('Architecture preset'), {
    target: { value: 'custom' },
  });

  const widthInput = screen.getByLabelText(/Model width/i);
  const headsInput = screen.getByLabelText(/Attention heads/i);

  fireEvent.change(widthInput, { target: { value: '128' } });
  fireEvent.change(headsInput, { target: { value: '6' } });

  const errorMatches = await screen.findAllByText(
    /Model width \(128\) must divide evenly by attention heads \(6\)/i,
  );
  expect(errorMatches.length).toBeGreaterThan(0);
});

it('creates and saves a model successfully', async () => {
  const createSpy = vi.spyOn(api, 'createModel').mockResolvedValue({
    ...mockModel,
    id: 'model-2',
    name: 'custom-math-gpt',
  });

  showPage();
  expect(await screen.findByText('Create a model architecture')).toBeTruthy();

  const nameInput = screen.getByLabelText(/^Name/);
  fireEvent.change(nameInput, { target: { value: 'custom-math-gpt' } });

  const saveBtn = screen.getByRole('button', { name: 'Save model' });
  expect(saveBtn.hasAttribute('disabled')).toBe(false);
  fireEvent.click(saveBtn);

  await waitFor(() => {
    expect(createSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'custom-math-gpt',
        n_layer: 4,
        d_model: 128,
        n_head: 4,
        ctx_len: 512,
      }),
    );
  });
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
