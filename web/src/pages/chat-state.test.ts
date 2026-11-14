import { describe, expect, it } from 'vitest';

import { ApiError, type ChatEvent, type ChatRequest } from '../api/client';
import {
  type Action,
  answer,
  type Chat,
  fresh,
  messages,
  problems,
  reduce,
  restore,
} from './chat-state';

function track(start: Chat = fresh()) {
  const box = { chat: start };
  const dispatch = (action: Action) => {
    box.chat = reduce(box.chat, action);
  };
  return { box, dispatch };
}

// fake chatStream
function playing(events: ChatEvent[]) {
  const asked: ChatRequest[] = [];
  const stream = async (request: ChatRequest, onEvent: (event: ChatEvent) => void) => {
    asked.push(request);
    for (const event of events) onEvent(event);
  };
  return { asked, stream };
}

const DONE: ChatEvent = { done: true, tokens: 5, seconds: 0.4, stopped: 'end' };

function ask(box: { chat: Chat }, dispatch: (action: Action) => void, text: string, seed = 7) {
  dispatch({ type: 'ask', text, seed });
  return box.chat.exchanges.length - 1;
}

describe('a conversation', () => {
  it('streams pieces into the answer, then notes how it ended', async () => {
    const { box, dispatch } = track({ ...fresh(), models: ['a', 'b'] });
    const at = ask(box, dispatch, 'Hello');
    expect(box.chat.exchanges[at].answers).toEqual([{ text: '', state: 'writing' }, null]);
    const { asked, stream } = playing([{ text: 'Hi ' }, { text: 'there' }, DONE]);
    await answer(box.chat, at, 0, dispatch, new AbortController().signal, stream);
    expect(box.chat.exchanges[at].answers[0]).toEqual({
      text: 'Hi there',
      state: 'done',
      pieces: 5,
      seconds: 0.4,
      long: false,
    });
    expect(asked[0]).toEqual({
      model_id: 'a',
      messages: [{ role: 'user', content: 'Hello' }],
      temperature: 0.7,
      top_p: 0.95,
      max_new_tokens: 256,
      seed: 7,
    });
  });

  it('sends the system message first and the earlier turns after it', async () => {
    const { box, dispatch } = track({ ...fresh(), models: ['a', 'b'] });
    dispatch({ type: 'settings', settings: { system: 'Be brief.', seed: 3 } });
    const first = ask(box, dispatch, 'One');
    await answer(
      box.chat,
      first,
      0,
      dispatch,
      new AbortController().signal,
      playing([{ text: 'Uno' }, DONE]).stream,
    );
    const second = ask(box, dispatch, 'Two');
    expect(messages(box.chat, second, 0)).toEqual([
      { role: 'system', content: 'Be brief.' },
      { role: 'user', content: 'One' },
      { role: 'assistant', content: 'Uno' },
      { role: 'user', content: 'Two' },
    ]);
  });

  it('says when an answer reached the length limit', async () => {
    const { box, dispatch } = track();
    const at = ask(box, dispatch, 'Go on');
    await answer(
      box.chat,
      at,
      0,
      dispatch,
      new AbortController().signal,
      playing([{ text: 'and on' }, { ...DONE, stopped: 'length' }]).stream,
    );
    expect(box.chat.exchanges[at].answers[0]?.long).toBe(true);
  });
});

describe('side by side', () => {
  it('sends the same message to both and keeps each history apart', async () => {
    const { box, dispatch } = track({ ...fresh(), models: ['tuned', 'base'], side: true });
    const first = ask(box, dispatch, 'Hi', 11);
    const left = playing([{ text: 'Hello, how can I help?' }, DONE]);
    const right = playing([{ text: 'Hi hi hi the' }, DONE]);
    await Promise.all([
      answer(box.chat, first, 0, dispatch, new AbortController().signal, left.stream),
      answer(box.chat, first, 1, dispatch, new AbortController().signal, right.stream),
    ]);
    expect(box.chat.exchanges[first].answers.map((one) => one?.text)).toEqual([
      'Hello, how can I help?',
      'Hi hi hi the',
    ]);
    expect(left.asked[0].model_id).toBe('tuned');
    expect(right.asked[0].model_id).toBe('base');
    // same seed for both so it's a fair fight
    expect(right.asked[0].seed).toBe(11);

    const second = ask(box, dispatch, 'Thanks');
    expect(messages(box.chat, second, 1)).toEqual([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hi hi hi the' },
      { role: 'user', content: 'Thanks' },
    ]);
  });

  it('leaves out of a history what that model never answered', () => {
    const { box, dispatch } = track({ ...fresh(), models: ['a', 'b'] });
    ask(box, dispatch, 'Before');
    dispatch({ type: 'side', on: true, second: 'b' });
    const at = ask(box, dispatch, 'After');
    expect(box.chat.exchanges[0].answers[1]).toBeNull();
    expect(messages(box.chat, at, 1)).toEqual([{ role: 'user', content: 'After' }]);
  });
});

describe('stop, failure and trying again', () => {
  it('stops every answer being written and keeps what was written', () => {
    const { box, dispatch } = track({ ...fresh(), side: true });
    const at = ask(box, dispatch, 'Hi');
    dispatch({ type: 'piece', at, column: 0, text: 'Half' });
    dispatch({ type: 'stop' });
    expect(box.chat.exchanges[at].answers).toEqual([
      { text: 'Half', state: 'stopped' },
      { text: '', state: 'stopped' },
    ]);
    dispatch({ type: 'piece', at, column: 0, text: ' more' });
    expect(box.chat.exchanges[at].answers[0]?.text).toBe('Half');
  });

  it("shows the server's reason in that column only, and tries again", async () => {
    const { box, dispatch } = track({ ...fresh(), side: true });
    const at = ask(box, dispatch, 'Hi');
    const signal = new AbortController().signal;
    await answer(box.chat, at, 0, dispatch, signal, playing([{ error: 'Out of memory.' }]).stream);
    await answer(box.chat, at, 1, dispatch, signal, playing([{ text: 'Fine' }, DONE]).stream);
    expect(box.chat.exchanges[at].answers[0]).toEqual({
      text: '',
      state: 'error',
      error: 'Out of memory.',
    });
    expect(box.chat.exchanges[at].answers[1]?.state).toBe('done');

    dispatch({ type: 'retry', at, column: 0 });
    await answer(box.chat, at, 0, dispatch, signal, playing([{ text: 'Now' }, DONE]).stream);
    expect(box.chat.exchanges[at].answers[0]?.text).toBe('Now');
    expect(box.chat.exchanges[at].answers[1]?.text).toBe('Fine');
  });

  it('says what went wrong when the answer cannot start or ends early', async () => {
    const { box, dispatch } = track();
    const at = ask(box, dispatch, 'Hi');
    const signal = new AbortController().signal;
    const refused = async () => {
      throw new ApiError(404, 'not_found', 'That model is gone.');
    };
    await answer(box.chat, at, 0, dispatch, signal, refused);
    expect(box.chat.exchanges[at].answers[0]?.error).toBe('That model is gone.');

    dispatch({ type: 'retry', at, column: 0 });
    await answer(box.chat, at, 0, dispatch, signal, playing([{ text: 'Hal' }]).stream);
    expect(box.chat.exchanges[at].answers[0]?.state).toBe('error');
  });

  it('stays quiet when the person stopped it', async () => {
    const { box, dispatch } = track();
    const at = ask(box, dispatch, 'Hi');
    const control = new AbortController();
    const stream = async () => {
      control.abort();
      dispatch({ type: 'stop' });
    };
    await answer(box.chat, at, 0, dispatch, control.signal, stream);
    expect(box.chat.exchanges[at].answers[0]?.state).toBe('stopped');
  });
});

describe('the settings', () => {
  it('names each number that is out of range', () => {
    expect(problems(fresh().settings)).toEqual({});
    expect(
      problems({ ...fresh().settings, temperature: 3, top_p: 0, max_new_tokens: null, seed: 1.5 }),
    ).toEqual({
      temperature: 'Use 0 to 2.',
      top_p: 'Use more than 0, up to 1.',
      max_new_tokens: 'Use 1 to 4,096.',
      seed: 'Use a whole number from 0.',
    });
  });
});

describe('kept in this browser', () => {
  it('comes back as it was, with unfinished answers marked stopped', () => {
    const { box, dispatch } = track({ ...fresh(), models: ['a', 'b'], side: true });
    const at = ask(box, dispatch, 'Hi');
    dispatch({ type: 'piece', at, column: 0, text: 'Hel' });
    const back = restore(JSON.stringify(box.chat));
    expect(back.models).toEqual(['a', 'b']);
    expect(back.side).toBe(true);
    expect(back.exchanges[0].answers[0]).toEqual({ text: 'Hel', state: 'stopped' });
  });

  it('ignores anything it cannot read', () => {
    for (const raw of [
      null,
      '',
      '{',
      '[]',
      '{"models":[1,2]}',
      JSON.stringify({ ...fresh(), exchanges: [{}] }),
    ])
      expect(restore(raw)).toEqual(fresh());
  });

  it('starts over on New chat but keeps the models and settings', () => {
    const { box, dispatch } = track({ ...fresh(), models: ['a', 'b'] });
    dispatch({ type: 'settings', settings: { temperature: 0 } });
    ask(box, dispatch, 'Hi');
    dispatch({ type: 'clear' });
    expect(box.chat.exchanges).toEqual([]);
    expect(box.chat.models).toEqual(['a', 'b']);
    expect(box.chat.settings.temperature).toBe(0);
  });
});
