// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { useLocation } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, type MockInstance, vi } from 'vitest';

import * as client from '../api/client';
import { api, ApiError, type ChatEvent, type Model } from '../api/client';
import { open } from '../test-utils';
import ChatPage from './ChatPage';
import { fresh, KEY } from './chat-state';

const model = (over: Partial<Model> = {}): Model => ({
  id: 'base',
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

const base = model();
const tuned = model({
  id: 'tuned',
  name: 'Letters chat',
  chat: true,
  parent_id: 'base',
  parent_name: 'Letters model',
});

let stream: MockInstance<typeof client.chatStream>;
beforeEach(() => {
  window.localStorage.clear();
  stream = vi.spyOn(client, 'chatStream');
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function ready(items: Model[] = [base, tuned]) {
  return vi.spyOn(api, 'listAllModels').mockResolvedValue({ items, total: items.length });
}

// fake stream, events picked by model
function answering(by: Record<string, ChatEvent[]>) {
  stream.mockImplementation(async (request, onEvent) => {
    for (const event of by[request.model_id]) onEvent(event);
  });
}

function Where() {
  return <output data-testid="where">{useLocation().search}</output>;
}

const done = (tokens = 3): ChatEvent => ({ done: true, tokens, seconds: 1.25, stopped: 'end' });

function say(text: string) {
  const box = screen.getByRole('textbox', { name: 'Your message' });
  fireEvent.change(box, { target: { value: text } });
  fireEvent.keyDown(box, { key: 'Enter' });
}

it('points to Train and Models when no model is ready', async () => {
  ready([model({ ready: false })]);
  open(<ChatPage />, '/chat');
  expect(await screen.findByText('No model to talk to yet')).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Train a model' }).getAttribute('href')).toBe('/train');
  expect(screen.getByRole('link', { name: 'Models' }).getAttribute('href')).toBe('/models');
});

it("shows the server's words when the models do not load, and tries again", async () => {
  const list = vi
    .spyOn(api, 'listAllModels')
    .mockRejectedValueOnce(new ApiError(500, 'error', 'The models could not be read.'))
    .mockResolvedValue({ items: [base], total: 1 });
  open(<ChatPage />, '/chat');
  expect(await screen.findByText('The models could not be read.')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(await screen.findByRole('combobox', { name: 'Who answers' })).toBeTruthy();
  expect(list).toHaveBeenCalledTimes(2);
});

it('starts with a chat model and streams its answer in', async () => {
  ready();
  answering({ tuned: [{ text: 'Hello' }, { text: ' there.\nHow are you?' }, done(12)] });
  open(
    <>
      <ChatPage />
      <Where />
    </>,
    '/chat',
  );
  const pick = (await screen.findByRole('combobox', { name: 'Who answers' })) as HTMLSelectElement;
  expect(pick.value).toBe('tuned');
  expect(
    screen.getByRole('option', { name: 'Letters chat (4.2M weights, chat, from Letters model)' }),
  );
  expect(screen.getByTestId('where').textContent).toBe('?model=tuned');

  say('Hi');
  expect(await screen.findByText('12 word pieces in 1.3 seconds')).toBeTruthy();
  const log = screen.getByRole('log', { name: 'Conversation' });
  expect(within(log).getByText('Hi')).toBeTruthy();
  expect(log.textContent).toContain('Hello there.\nHow are you?');
  expect(stream.mock.calls[0][0]).toMatchObject({
    model_id: 'tuned',
    messages: [{ role: 'user', content: 'Hi' }],
    temperature: 0.7,
    top_p: 0.95,
    max_new_tokens: 256,
  });
  expect(document.activeElement).toBe(screen.getByRole('textbox', { name: 'Your message' }));
});

it('makes a new line on Shift+Enter instead of sending', async () => {
  ready();
  open(<ChatPage />, '/chat');
  await screen.findByRole('combobox', { name: 'Who answers' });
  const box = screen.getByRole('textbox', { name: 'Your message' });
  fireEvent.change(box, { target: { value: 'Hi' } });
  fireEvent.keyDown(box, { key: 'Enter', shiftKey: true });
  expect(stream).not.toHaveBeenCalled();
});

it('says when a model has not learned to chat, and when it hit the length limit', async () => {
  ready([base]);
  answering({ base: [{ text: 'the the the' }, { ...done(256), stopped: 'length' }] });
  open(<ChatPage />, '/chat');
  expect(await screen.findByText(/has not learned to chat/)).toBeTruthy();
  say('Hi');
  expect(
    await screen.findByText('256 word pieces in 1.3 seconds. Stopped at the length limit.'),
  ).toBeTruthy();
});

it('puts a fine-tune beside its parent, one message to both', async () => {
  ready();
  answering({
    tuned: [{ text: 'Hello! How can I help?' }, done()],
    base: [{ text: 'Hi hi the the' }, done()],
  });
  open(
    <>
      <ChatPage />
      <Where />
    </>,
    '/chat',
  );
  fireEvent.click(await screen.findByRole('switch', { name: 'Side by side' }));
  const second = screen.getByRole('combobox', { name: 'Second model' }) as HTMLSelectElement;
  expect(second.value).toBe('base');
  expect(screen.getByTestId('where').textContent).toBe('?model=tuned&with=base');

  say('Hi');
  expect(await screen.findByText('Hello! How can I help?')).toBeTruthy();
  expect(await screen.findByText('Hi hi the the')).toBeTruthy();
  expect(stream.mock.calls.map(([request]) => request.model_id)).toEqual(['tuned', 'base']);
  const seeds = stream.mock.calls.map(([request]) => request.seed);
  expect(seeds[0]).toBe(seeds[1]);

  say('Thanks');
  await waitFor(() => expect(stream).toHaveBeenCalledTimes(4));
  expect(stream.mock.calls[3][0].messages).toEqual([
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hi hi the the' },
    { role: 'user', content: 'Thanks' },
  ]);
});

it('turns Send into Stop, and stops every answer being written', async () => {
  ready();
  const signals: AbortSignal[] = [];
  stream.mockImplementation(
    (_, onEvent, signal) =>
      new Promise((resolve) => {
        signals.push(signal!);
        onEvent({ text: 'Half an' });
        signal!.addEventListener('abort', () => resolve());
      }),
  );
  open(<ChatPage />, '/chat?model=tuned&with=base');
  await screen.findByRole('combobox', { name: 'Second model' });
  say('Hi');
  const stop = await screen.findByRole('button', { name: 'Stop' });
  for (const name of ['First model', 'Second model'])
    expect((screen.getByRole('combobox', { name }) as HTMLSelectElement).disabled).toBe(true);
  expect(JSON.parse(window.localStorage.getItem(KEY)!).exchanges).toEqual([]);
  fireEvent.click(stop);
  expect(await screen.findAllByText('Stopped')).toHaveLength(2);
  expect(JSON.parse(window.localStorage.getItem(KEY)!).exchanges).toHaveLength(1);
  expect(signals.map((signal) => signal.aborted)).toEqual([true, true]);
  expect(screen.getAllByText('Half an')).toHaveLength(2);
  expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy();
});

it("shows the server's reason in that column, and tries again", async () => {
  ready();
  stream
    .mockImplementationOnce(async (_, onEvent) => onEvent({ error: 'The graphics card ran out.' }))
    .mockImplementation(async (_, onEvent) => {
      onEvent({ text: 'Better now' });
      onEvent(done());
    });
  open(<ChatPage />, '/chat');
  await screen.findByRole('combobox', { name: 'Who answers' });
  say('Hi');
  expect(await screen.findByText('The graphics card ran out.')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(await screen.findByText('Better now')).toBeTruthy();
  expect(screen.queryByText('The graphics card ran out.')).toBeNull();
});

it('keeps the conversation over a reload, and New chat clears it', async () => {
  ready();
  const kept = {
    ...fresh(),
    models: ['base', 'tuned'],
    exchanges: [
      {
        ask: 'Who are you?',
        seed: 4,
        answers: [{ text: 'A small model.', state: 'done', pieces: 4, seconds: 0.5 }, null],
      },
    ],
  };
  window.localStorage.setItem(KEY, JSON.stringify(kept));
  open(<ChatPage />, '/chat');
  expect(await screen.findByText('A small model.')).toBeTruthy();
  expect((screen.getByRole('combobox', { name: 'Who answers' }) as HTMLSelectElement).value).toBe(
    'base',
  );
  fireEvent.click(screen.getByRole('button', { name: 'New chat' }));
  expect(screen.queryByText('A small model.')).toBeNull();
  expect(JSON.parse(window.localStorage.getItem(KEY)!).exchanges).toEqual([]);
});

it('says No answer for an empty one, and Not asked where it was not sent', async () => {
  ready();
  answering({ tuned: [done(0)], base: [done(0)] });
  open(<ChatPage />, '/chat');
  await screen.findByRole('combobox', { name: 'Who answers' });
  say('Hi');
  expect(await screen.findByText('No answer')).toBeTruthy();
  fireEvent.click(screen.getByRole('switch', { name: 'Side by side' }));
  expect(screen.getByText('Not asked')).toBeTruthy();
});

it('holds back Send while a setting is out of range', async () => {
  ready();
  open(<ChatPage />, '/chat');
  fireEvent.change(await screen.findByRole('spinbutton', { name: 'Temperature' }), {
    target: { value: '5' },
  });
  expect(screen.getByText('Use 0 to 2.')).toBeTruthy();
  say('Hi');
  expect(stream).not.toHaveBeenCalled();
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true);
});
