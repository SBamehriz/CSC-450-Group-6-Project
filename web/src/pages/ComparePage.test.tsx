// @vitest-environment jsdom
import { cleanup, fireEvent, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type CompareModel, type Model } from '../api/client';
import { open } from '../test-utils';
import ComparePage from './ComparePage';

const model = (id: string, source = 'forge'): Model => ({
  id,
  name: `${id} model`,
  description: '',
  preset: null,
  n_layer: 12,
  d_model: 768,
  n_head: 12,
  ctx_len: 1024,
  vocab_size: 50257,
  dropout: 0,
  param_count: 150_000_000,
  config: {},
  tokenizer_id: null,
  source,
  created_at: '2026-10-22T00:00:00Z',
});

const side = (id: string, change: Partial<CompareModel> = {}): CompareModel => ({
  id,
  name: `${id} model`,
  source: 'forge',
  parameters: 150_000_000,
  vocab_size: 50257,
  ctx_len: 1024,
  license: null,
  tokens_seen: 2_000_000,
  best_val_loss: 3.21,
  bits_per_byte: null,
  benchmark: null,
  ...change,
});

beforeEach(() => {
  vi.spyOn(api, 'listAllModels').mockResolvedValue({
    items: [model('ours'), model('theirs', 'hub')],
    total: 2,
  });
  vi.spyOn(api, 'benchmarkPlan').mockResolvedValue({
    parameters: 150_000_000,
    allowed: [{ task: 'hellaswag', name: 'HellaSwag', chance: 0.25 }],
    refused: [
      {
        task: 'mmlu',
        name: 'MMLU',
        reason: 'MMLU needs a model of at least about 1 billion parameters.',
      },
    ],
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('shows each score with its interval and the chance level', async () => {
  vi.spyOn(api, 'compare').mockResolvedValue({
    models: [
      side('ours', {
        benchmark: {
          job_id: 'job-1',
          finished_at: null,
          checkpoint_id: null,
          limit: 200,
          harness_version: 'lm-eval 0.4.13',
          scores: [
            {
              task: 'hellaswag',
              name: 'HellaSwag',
              metric: 'acc_norm',
              score: 0.31,
              low: 0.29,
              high: 0.33,
              chance: 0.25,
              questions: 200,
            },
          ],
          refused: [],
        },
      }),
    ],
    prompt_runs: [],
  });
  open(<ComparePage />);
  fireEvent.change(await screen.findByLabelText('First model'), { target: { value: 'ours' } });
  const row = (await screen.findByRole('cell', { name: 'HellaSwag' })).closest('tr');
  const cells = within(row as HTMLElement);
  expect(cells.getByText('31.0%')).toBeTruthy();
  expect(cells.getByText('29.0% to 33.0%')).toBeTruthy();
  expect(cells.getByText('25%')).toBeTruthy();
  expect(screen.getByText(/Quick look: at most 200 questions each/)).toBeTruthy();
  expect(await screen.findByText(/MMLU needs a model of at least about 1 billion/)).toBeTruthy();
  expect(screen.getByText('Not measured yet')).toBeTruthy();
  expect(screen.getByText('3.210')).toBeTruthy();
});

it('shows two models answering the same prompts side by side', async () => {
  vi.spyOn(api, 'compare').mockResolvedValue({
    models: [side('ours'), side('theirs', { source: 'hub', license: 'mit', tokens_seen: null })],
    prompt_runs: [
      {
        job_id: 'job-2',
        finished_at: null,
        settings: { temperature: 0.8, top_k: 40, max_new_tokens: 16, seed: 3 },
        model_names: ['ours model', 'theirs model'],
        rows: [{ prompt: 'The river', answers: ['runs past the mill', 'is very wide'] }],
      },
    ],
  });
  open(<ComparePage />);
  fireEvent.change(await screen.findByLabelText('First model'), { target: { value: 'ours' } });
  fireEvent.change(screen.getByLabelText('Second model'), { target: { value: 'theirs' } });
  const row = (await screen.findByRole('cell', { name: 'The river' })).closest('tr');
  const cells = within(row as HTMLElement);
  expect(cells.getByText('runs past the mill')).toBeTruthy();
  expect(cells.getByText('is very wide')).toBeTruthy();
  expect(screen.getByText(/Temperature 0.8, top-k 40, up to 16 word pieces, seed 3/)).toBeTruthy();
  expect(screen.getByRole('cell', { name: 'mit' })).toBeTruthy();

  const ask = vi.spyOn(api, 'comparePrompts').mockResolvedValue({
    id: 'job-3',
    job_type: 'prompt_compare',
    status: 'pending',
    payload: {},
    progress: {},
    error: null,
    attempts: 0,
    cancel_requested: false,
    result: {},
    started_at: null,
    completed_at: null,
    created_at: '2026-11-01T00:00:00Z',
  });
  vi.spyOn(api, 'getJob').mockResolvedValue({
    id: 'job-3',
    job_type: 'prompt_compare',
    status: 'running',
    payload: {},
    progress: { completed: 1, total: 4, message: 'ours model: prompt 1 of 2' },
    error: null,
    attempts: 1,
    cancel_requested: false,
    result: {},
    started_at: null,
    completed_at: null,
    created_at: '2026-11-01T00:00:00Z',
  });
  fireEvent.change(screen.getByLabelText('Prompts, one per line'), {
    target: { value: 'Hello\n\nOnce upon a time' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Compare answers' }));
  expect(await screen.findByText('ours model: prompt 1 of 2')).toBeTruthy();
  expect(ask).toHaveBeenCalledWith(
    expect.objectContaining({
      model_ids: ['ours', 'theirs'],
      prompts: ['Hello', 'Once upon a time'],
      seed: 0,
    }),
  );
});
