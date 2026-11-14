// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, ApiError, type Bucket, type Model, type Plan, type Run } from '../api/client';
import { bucket, open } from '../test-utils';
import TrainPlanner, { parseAmount } from './TrainPlanner';

const WHY = {
  parameters: 'A size that suits your collection.',
  n_layer: 'Deeper models can learn harder patterns.',
  train_tokens: 'About 20 word pieces for every parameter.',
};

const plan = (over: Partial<Plan> = {}): Plan => ({
  fine_tune: false,
  shape: { n_layer: 6, d_model: 256, n_head: 8, ctx_len: 256, vocab_size: 16384 },
  dropout: 0,
  parameters: 4_200_000,
  train_tokens: 84_000_000,
  settings: {
    batch_size: 32,
    micro_batch_size: 0,
    max_steps: 10_000,
    learning_rate: 0.0006,
    warmup_steps: 200,
    weight_decay: 0.1,
    grad_clip: 1,
    eval_every: 500,
    eval_tokens: 100_000,
    save_every: 2500,
  },
  why: WHY,
  data: { tokens: 30_000_000, exact: false, passes: 2.8, enough: true },
  memory: {
    needed_bytes: 2 ** 30,
    free_bytes: 3 * 2 ** 30,
    cards: 1,
    card_name: 'RTX 3050',
    sequences_per_pass: 16,
    fits: true,
  },
  time: { tokens_per_sec: 40_000, hours: 1.5, source: 'measured' },
  warnings: [],
  blockers: [],
  ...over,
});

const letters: Bucket = { ...bucket, name: 'Letters', kind: 'text' };
const talks: Bucket = { ...bucket, id: 'bucket-2', name: 'Talks', kind: 'chat' };

const model = (over: Partial<Model> = {}): Model => ({
  id: 'model-1',
  name: 'Letters model',
  description: '',
  preset: null,
  n_layer: 6,
  d_model: 256,
  n_head: 8,
  ctx_len: 256,
  vocab_size: 16384,
  dropout: 0,
  param_count: 4_200_000,
  config: {},
  tokenizer_id: null,
  source: 'forge',
  chat: false,
  ready: true,
  created_at: '2026-10-22T00:00:00Z',
  ...over,
});

const started = { id: 'run-9' } as Run;

const page = (path = '/train') =>
  open(
    <Routes>
      <Route path="/train" element={<TrainPlanner />} />
      <Route path="/runs/:runId" element={<p>Run page</p>} />
    </Routes>,
    path,
  );

const asked = (spy: { mock: { calls: unknown[][] } }) => spy.mock.calls.at(-1)?.[0];

beforeEach(() => {
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [letters, talks], total: 2 });
  vi.spyOn(api, 'listAllModels').mockResolvedValue({ items: [], total: 0 });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('reads sizes as people write them', () => {
  expect(parseAmount('135M')).toBe(135_000_000);
  expect(parseAmount('2b')).toBe(2e9);
  expect(parseAmount('1,000')).toBe(1000);
  expect(parseAmount('6e-4')).toBe(0.0006);
  expect(parseAmount('lots')).toBeNull();
});

it('plans with Forge choosing everything, and starts with that plan', async () => {
  const planned = vi.spyOn(api, 'plan').mockResolvedValue(plan());
  const train = vi.spyOn(api, 'train').mockResolvedValue(started);
  page();
  expect(await screen.findByText(WHY.n_layer)).toBeTruthy();
  expect(asked(planned)).toEqual({ bucket_id: 'bucket-1', base_model_id: null, device: 'auto' });
  expect(screen.getByRole('textbox', { name: 'Layers' })).toHaveProperty('value', '6');
  expect(screen.getByRole('textbox', { name: 'Word pieces to read' })).toHaveProperty(
    'value',
    '84M',
  );
  expect(screen.getByRole('textbox', { name: 'Parameters' })).toHaveProperty('value', '4.2M');
  expect(screen.getByText('About 1 h 30 min')).toBeTruthy();
  expect(screen.getByText('1.0 GB of 3.0 GB free')).toBeTruthy();
  expect(screen.getByText('Training reads it 2.8 times.')).toBeTruthy();
  const go = screen.getByRole('button', { name: 'Start training' });
  await waitFor(() => expect(go).toHaveProperty('disabled', false));
  fireEvent.click(go);
  expect(await screen.findByText('Run page')).toBeTruthy();
  expect(train).toHaveBeenCalledWith({
    name: 'Letters 4.2M',
    bucket_id: 'bucket-1',
    base_model_id: null,
    shape: plan().shape,
    dropout: 0,
    settings: plan().settings,
    seed: 0,
    device: 'auto',
  });
});

it('points to Data when there is no collection', async () => {
  vi.spyOn(api, 'listBuckets').mockResolvedValue({ items: [], total: 0 });
  vi.spyOn(api, 'plan').mockResolvedValue(plan());
  page();
  expect(await screen.findByText('No collections yet')).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Add text' }).getAttribute('href')).toBe('/data');
  await screen.findByText(WHY.n_layer);
  expect(screen.getByRole('button', { name: 'Start training' })).toHaveProperty('disabled', true);
});

it('picks the collection from the address and shows its kind', async () => {
  const planned = vi.spyOn(api, 'plan').mockResolvedValue(plan());
  page('/train?data=bucket-2');
  await waitFor(() => expect(asked(planned)).toMatchObject({ bucket_id: 'bucket-2' }));
  expect(screen.getByRole('option', { name: 'Talks (chat, 2.5K word pieces)' })).toBeTruthy();
  expect(screen.getByText('Conversations: the model learns to answer.')).toBeTruthy();
});

it('takes any size typed, and hands it back to Forge on undo', async () => {
  const planned = vi.spyOn(api, 'plan').mockResolvedValue(plan());
  page();
  const size = await screen.findByRole('textbox', { name: 'Parameters' });
  fireEvent.change(size, { target: { value: '2B' } });
  await waitFor(() => expect(asked(planned)).toMatchObject({ parameters: 2e9 }));
  fireEvent.click(screen.getByRole('button', { name: 'Undo Parameters' }));
  await waitFor(() => expect(asked(planned)).not.toHaveProperty('parameters'));
});

it('plans by the hours you have', async () => {
  const planned = vi.spyOn(api, 'plan').mockResolvedValue(plan());
  page();
  fireEvent.click(await screen.findByRole('button', { name: 'By time' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Hours you have' }), {
    target: { value: '3' },
  });
  await waitFor(() => expect(asked(planned)).toMatchObject({ hours: 3 }));
  expect(asked(planned)).not.toHaveProperty('parameters');
});

it('marks a number you change, sends only that, and hands it back on undo', async () => {
  const planned = vi.spyOn(api, 'plan').mockResolvedValue(plan());
  page();
  fireEvent.change(await screen.findByRole('textbox', { name: 'Layers' }), {
    target: { value: '12' },
  });
  await waitFor(() => expect(asked(planned)).toMatchObject({ n_layer: 12 }));
  expect(asked(planned)).not.toHaveProperty('d_model');
  expect(screen.getByText('Changed')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Undo Layers' }));
  await waitFor(() => expect(asked(planned)).not.toHaveProperty('n_layer'));
  expect(screen.queryByText('Changed')).toBeNull();
});

it('keeps the last plan on screen while the next one loads, and waits for it', async () => {
  vi.spyOn(api, 'plan')
    .mockResolvedValueOnce(plan())
    .mockImplementation(() => new Promise(() => {}));
  page();
  const layers = await screen.findByRole('textbox', { name: 'Layers' });
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'Start training' })).toHaveProperty(
      'disabled',
      false,
    ),
  );
  fireEvent.change(layers, { target: { value: '12' } });
  expect(await screen.findByText('Planning…')).toBeTruthy();
  expect(screen.getByText(WHY.n_layer)).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Start training' })).toHaveProperty('disabled', true);
});

it('keeps the rarer settings under Advanced settings, still editable', async () => {
  const planned = vi.spyOn(api, 'plan').mockResolvedValue(plan());
  page();
  const heads = await screen.findByRole('textbox', { name: 'Attention heads' });
  expect(heads.closest('details')).not.toBeNull();
  expect(screen.getByRole('textbox', { name: 'Sequences per step' }).closest('details')).toBeNull();
  fireEvent.change(heads, { target: { value: '4' } });
  await waitFor(() => expect(asked(planned)).toMatchObject({ n_head: 4 }));
});

it('fine-tunes a ready Forge model with its shape fixed', async () => {
  vi.spyOn(api, 'listAllModels').mockResolvedValue({
    items: [
      model(),
      model({ id: 'hub-1', name: 'Hub model', source: 'hub' }),
      model({ id: 'new-1', name: 'Untrained model', ready: false }),
    ],
    total: 3,
  });
  const planned = vi.spyOn(api, 'plan').mockResolvedValue(plan({ fine_tune: true }));
  const train = vi.spyOn(api, 'train').mockResolvedValue(started);
  page('/train?from=model-1');
  await waitFor(() => expect(asked(planned)).toMatchObject({ base_model_id: 'model-1' }));
  expect(await screen.findByText('Set by Letters model, the model it starts from.')).toBeTruthy();
  expect(screen.queryByRole('option', { name: /Hub model/ })).toBeNull();
  expect(screen.queryByRole('option', { name: /Untrained model/ })).toBeNull();
  expect(screen.queryByRole('textbox', { name: 'Parameters' })).toBeNull();
  expect(screen.queryByRole('textbox', { name: 'Width' })).toBeNull();
  expect(screen.queryByRole('textbox', { name: 'Attention heads' })).toBeNull();
  fireEvent.click(await screen.findByRole('button', { name: 'Start fine-tuning' }));
  await screen.findByText('Run page');
  expect(asked(train)).toMatchObject({
    base_model_id: 'model-1',
    name: 'Letters model on Letters',
  });
});

it('says when the time is a guess and points to the speed test', async () => {
  vi.spyOn(api, 'plan').mockResolvedValue(
    plan({
      data: { tokens: null, exact: false, passes: null, enough: null },
      time: { tokens_per_sec: null, hours: null, source: 'none' },
    }),
  );
  page();
  expect(await screen.findByText('Not known yet')).toBeTruthy();
  expect(screen.getByText('Not counted yet')).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Settings' }).getAttribute('href')).toBe('/settings');
});

it('shows warnings, and will not start past a blocker', async () => {
  vi.spyOn(api, 'plan').mockResolvedValue(
    plan({ warnings: ['The time is a rough guess.'], blockers: ['This model needs 9 GB.'] }),
  );
  page();
  expect(await screen.findByText('This model needs 9 GB.')).toBeTruthy();
  expect(screen.getByText('The time is a rough guess.')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Start training' })).toHaveProperty('disabled', true);
});

it("shows the server's reason when it will not start", async () => {
  vi.spyOn(api, 'plan').mockResolvedValue(plan());
  vi.spyOn(api, 'train').mockRejectedValue(
    new ApiError(409, 'name_taken', 'A model called Letters 4.2M already exists.'),
  );
  page();
  await screen.findByText(WHY.n_layer);
  const go = screen.getByRole('button', { name: 'Start training' });
  await waitFor(() => expect(go).toHaveProperty('disabled', false));
  fireEvent.click(go);
  expect(await screen.findByText('A model called Letters 4.2M already exists.')).toBeTruthy();
});

it('says when Forge could not plan', async () => {
  vi.spyOn(api, 'plan').mockRejectedValue(new ApiError(422, 'invalid', 'Width is too small.'));
  page();
  expect(await screen.findByText('Width is too small.')).toBeTruthy();
});
