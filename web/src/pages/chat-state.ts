import {
  ApiError,
  chatStream,
  type ChatEvent,
  type ChatMessage,
  type ChatRequest,
} from '../api/client';

export const KEY = 'forge.chat.v1';
// server cap, system message included
const MOST_TURNS = 200;
const ENDED_EARLY = 'The model stopped before it finished this answer. Try again.';
const UNREADABLE = 'The answer could not be read. Try again.';

export type Answer = {
  text: string;
  state: 'writing' | 'done' | 'stopped' | 'error';
  pieces?: number;
  seconds?: number;
  // hit the length limit
  long?: boolean;
  error?: string;
};

// null answer means that column wasn't asked
type Exchange = { ask: string; seed: number; answers: [Answer | null, Answer | null] };

export type Settings = {
  system: string;
  temperature: number | null;
  top_p: number | null;
  max_new_tokens: number | null;
  // null picks a new one each message
  seed: number | null;
};

export type Chat = {
  models: [string, string];
  side: boolean;
  settings: Settings;
  exchanges: Exchange[];
};

export type Action =
  | { type: 'ask'; text: string; seed: number }
  | { type: 'retry'; at: number; column: number }
  | { type: 'piece'; at: number; column: number; text: string }
  | { type: 'done'; at: number; column: number; pieces: number; seconds: number; long: boolean }
  | { type: 'error'; at: number; column: number; message: string }
  | { type: 'stop' }
  | { type: 'clear' }
  | { type: 'model'; column: number; id: string }
  | { type: 'side'; on: boolean; second?: string }
  | { type: 'settings'; settings: Partial<Settings> };

export const fresh = (): Chat => ({
  models: ['', ''],
  side: false,
  settings: { system: '', temperature: 0.7, top_p: 0.95, max_new_tokens: 256, seed: null },
  exchanges: [],
});

const writing = (): Answer => ({ text: '', state: 'writing' });

// late pieces after a stop get ignored
function update(chat: Chat, at: number, column: number, change: (answer: Answer) => Answer): Chat {
  const answer = chat.exchanges[at]?.answers[column];
  if (!answer || answer.state !== 'writing') return chat;
  return setAnswer(chat, at, column, change(answer));
}

function setAnswer(chat: Chat, at: number, column: number, answer: Answer): Chat {
  const exchanges = chat.exchanges.map((exchange, index) => {
    if (index !== at) return exchange;
    const answers: Exchange['answers'] = [...exchange.answers];
    answers[column] = answer;
    return { ...exchange, answers };
  });
  return { ...chat, exchanges };
}

export function reduce(chat: Chat, action: Action): Chat {
  switch (action.type) {
    case 'ask':
      return {
        ...chat,
        exchanges: [
          ...chat.exchanges,
          {
            ask: action.text,
            seed: action.seed,
            answers: [writing(), chat.side ? writing() : null],
          },
        ],
      };
    case 'retry':
      return setAnswer(chat, action.at, action.column, writing());
    case 'piece':
      return update(chat, action.at, action.column, (a) => ({ ...a, text: a.text + action.text }));
    case 'done':
      return update(chat, action.at, action.column, (a) => ({
        ...a,
        state: 'done',
        pieces: action.pieces,
        seconds: action.seconds,
        long: action.long,
      }));
    case 'error':
      return update(chat, action.at, action.column, (a) => ({
        ...a,
        state: 'error',
        error: action.message,
      }));
    case 'stop':
      return {
        ...chat,
        exchanges: chat.exchanges.map((exchange) => ({
          ...exchange,
          answers: exchange.answers.map((a) =>
            a?.state === 'writing' ? { ...a, state: 'stopped' } : a,
          ) as Exchange['answers'],
        })),
      };
    case 'clear':
      return { ...chat, exchanges: [] };
    case 'model': {
      const models: Chat['models'] = [...chat.models];
      models[action.column] = action.id;
      return { ...chat, models };
    }
    case 'side':
      return {
        ...chat,
        side: action.on,
        models: action.second ? [chat.models[0], action.second] : chat.models,
      };
    case 'settings':
      return { ...chat, settings: { ...chat.settings, ...action.settings } };
  }
}

export const busy = (chat: Chat) =>
  chat.exchanges.some((exchange) => exchange.answers.some((a) => a?.state === 'writing'));

// each column only sees its own answers
export function messages(chat: Chat, at: number, column: number): ChatMessage[] {
  const turns: ChatMessage[] = [];
  chat.exchanges.slice(0, at).forEach((exchange) => {
    const said = exchange.answers[column]?.text;
    if (said)
      turns.push({ role: 'user', content: exchange.ask }, { role: 'assistant', content: said });
  });
  turns.push({ role: 'user', content: chat.exchanges[at].ask });
  // drop old turns, leave room for the system message
  const latest = turns.slice(1 - MOST_TURNS);
  const system = chat.settings.system.trim();
  return system ? [{ role: 'system', content: system }, ...latest] : latest;
}

function request(chat: Chat, at: number, column: number): ChatRequest {
  const { temperature, top_p, max_new_tokens } = chat.settings;
  return {
    model_id: chat.models[column],
    messages: messages(chat, at, column),
    temperature: temperature ?? undefined,
    top_p: top_p ?? undefined,
    max_new_tokens: max_new_tokens ?? undefined,
    seed: chat.exchanges[at].seed,
  };
}

export async function answer(
  chat: Chat,
  at: number,
  column: number,
  dispatch: (action: Action) => void,
  signal: AbortSignal,
  stream: typeof chatStream = chatStream,
): Promise<void> {
  let ended = false;
  const heard = (event: ChatEvent) => {
    if ('text' in event) return dispatch({ type: 'piece', at, column, text: event.text });
    ended = true;
    if ('error' in event) dispatch({ type: 'error', at, column, message: event.error });
    else
      dispatch({
        type: 'done',
        at,
        column,
        pieces: event.tokens,
        seconds: event.seconds,
        long: event.stopped === 'length',
      });
  };
  try {
    await stream(request(chat, at, column), heard, signal);
    if (!ended && !signal.aborted) dispatch({ type: 'error', at, column, message: ENDED_EARLY });
  } catch (problem) {
    if (signal.aborted) return;
    const message = problem instanceof ApiError ? problem.message : UNREADABLE;
    dispatch({ type: 'error', at, column, message });
  }
}

const SEED_MOST = 2 ** 31 - 1;
const LONGEST = 4096;

export const HINTS = {
  temperature: '0 always picks the likeliest word piece. Higher is more varied.',
  max_new_tokens: 'The answer stops here.',
  top_p: 'Lower is safer. 1 allows every word piece.',
  seed: 'The same seed gives the same answer every time.',
};

// same limits the server checks
export function problems(settings: Settings) {
  const { temperature: t, top_p: p, max_new_tokens: n, seed } = settings;
  const out: Partial<Record<keyof Settings, string>> = {};
  if (t === null || t < 0 || t > 2) out.temperature = 'Use 0 to 2.';
  if (p === null || p <= 0 || p > 1) out.top_p = 'Use more than 0, up to 1.';
  if (n === null || !Number.isInteger(n) || n < 1 || n > LONGEST)
    out.max_new_tokens = `Use 1 to ${LONGEST.toLocaleString('en-US')}.`;
  if (seed !== null && (!Number.isInteger(seed) || seed < 0 || seed > SEED_MOST))
    out.seed = 'Use a whole number from 0.';
  return out;
}

export const newSeed = () => Math.floor(Math.random() * SEED_MOST);

// saved chats

const isText = (value: unknown): value is string => typeof value === 'string';
const isNumberOrNull = (value: unknown) => value === null || typeof value === 'number';
const STATES = ['writing', 'done', 'stopped', 'error'];

function readAnswer(value: unknown): Answer | null | undefined {
  if (value === null) return null;
  const a = value as Answer;
  if (typeof a !== 'object' || !isText(a.text) || !STATES.includes(a.state)) return undefined;
  return a.state === 'writing' ? { ...a, state: 'stopped' } : a;
}

function readExchange(value: unknown): Exchange | undefined {
  const e = value as Exchange;
  if (typeof e !== 'object' || e === null || !isText(e.ask) || typeof e.seed !== 'number')
    return undefined;
  if (!Array.isArray(e.answers) || e.answers.length !== 2) return undefined;
  const answers = e.answers.map(readAnswer);
  if (answers.includes(undefined)) return undefined;
  return { ask: e.ask, seed: e.seed, answers: answers as Exchange['answers'] };
}

// anything odd in the saved chat and we just start fresh
export function restore(raw: string | null): Chat {
  try {
    const kept = JSON.parse(raw ?? '') as Chat;
    const { models, side, settings: s, exchanges } = kept;
    const settingsOk =
      typeof s === 'object' &&
      s !== null &&
      isText(s.system) &&
      [s.temperature, s.top_p, s.max_new_tokens, s.seed].every(isNumberOrNull);
    const modelsOk = Array.isArray(models) && models.length === 2 && models.every(isText);
    if (!modelsOk || typeof side !== 'boolean' || !settingsOk || !Array.isArray(exchanges))
      return fresh();
    const read = exchanges.map(readExchange);
    if (read.includes(undefined)) return fresh();
    return {
      models: [models[0], models[1]],
      side,
      settings: {
        system: s.system,
        temperature: s.temperature,
        top_p: s.top_p,
        max_new_tokens: s.max_new_tokens,
        seed: s.seed,
      },
      exchanges: read as Exchange[],
    };
  } catch {
    return fresh();
  }
}
