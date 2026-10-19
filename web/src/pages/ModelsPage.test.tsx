// @vitest-environment jsdom
import { cleanup, fireEvent, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type GpuCheck, type Preset } from '../api/client';
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
      driver: '595.79',
      compute_capability: '8.6',
    },
  ],
  tool_said: null,
  torch: 'cuda',
  torch_version: '2.14.1+cu130',
  torch_cuda: '13.0',
};

beforeEach(() => {
  vi.spyOn(api, 'listPresets').mockResolvedValue(sizes);
  vi.spyOn(api, 'checkGpu').mockResolvedValue(oneCard);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function rowOf(name: string) {
  const cell = screen.getByRole('cell', { name });
  return within(cell.closest('tr') as HTMLElement);
}

it('judges each size against the card this computer has', async () => {
  open(<ModelsPage />);
  expect(await screen.findByText('NVIDIA GeForce RTX 3090', { selector: 'dd' })).toBeTruthy();
  expect(screen.getByText('24.0 GB', { selector: 'dd' })).toBeTruthy();
  expect(screen.getByText('PyTorch 2.14.1+cu130 with CUDA 13.0')).toBeTruthy();
  expect(
    await screen.findByRole('columnheader', { name: 'On NVIDIA GeForce RTX 3090' }),
  ).toBeTruthy();
  expect(rowOf('nano').getByText('4.0 GB')).toBeTruthy();
  expect(rowOf('nano').getByText('Fits')).toBeTruthy();
  expect(rowOf('135m').getByText('24.0 GB')).toBeTruthy();
  expect(rowOf('135m').getByText('Fits')).toBeTruthy();

  fireEvent.change(screen.getByLabelText('Sequences per step'), { target: { value: '16' } });
  expect(rowOf('135m').getByText('48.0 GB')).toBeTruthy();
  expect(rowOf('135m').getByText('Too big')).toBeTruthy();
  expect(rowOf('nano').getByText('Fits')).toBeTruthy();
});

it('says when there is no card and judges nothing', async () => {
  vi.mocked(api.checkGpu).mockResolvedValue({
    cards: [],
    tool_said: 'NVIDIA-SMI has failed.',
    torch: 'missing',
    torch_version: null,
    torch_cuda: null,
  });
  open(<ModelsPage />);
  expect(await screen.findByText(/No NVIDIA graphics card was found/)).toBeTruthy();
  expect(screen.getByText("The card's driver said: NVIDIA-SMI has failed.")).toBeTruthy();
  expect(screen.getByText(/PyTorch isn't installed/)).toBeTruthy();
  expect(await screen.findByRole('cell', { name: '135m' })).toBeTruthy();
  expect(screen.queryByText('Fits')).toBeNull();
  expect(screen.queryByText('Too big')).toBeNull();
});

it('warns when PyTorch cannot use the card that is there', async () => {
  vi.mocked(api.checkGpu).mockResolvedValue({
    ...oneCard,
    torch: 'cpu',
    torch_version: '2.14.1+cpu',
    torch_cuda: null,
  });
  open(<ModelsPage />);
  expect(await screen.findByText(/installed without graphics card support/)).toBeTruthy();
});

it('recovers when a check fails', async () => {
  vi.mocked(api.checkGpu).mockRejectedValueOnce(new Error('Server unavailable.'));
  open(<ModelsPage />);
  expect(await screen.findByText('Server unavailable.')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(await screen.findByText('NVIDIA GeForce RTX 3090', { selector: 'dd' })).toBeTruthy();
});
