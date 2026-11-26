import './chat.css';

import { useQuery } from '@tanstack/react-query';
import { useEffect, useLayoutEffect, useReducer, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { api, type Model } from '../api/client';
import { Card, CardHead, Empty, ErrorBox, Notice } from '../components/ui';
import { compactNumber, plural } from '../lib/format';
import {
  type Action,
  answer,
  type Answer,
  busy,
  type Chat,
  HINTS,
  KEY,
  loadHistory,
  newSeed,
  problems,
  reduce,
  restore,
  saveHistory,
  type SavedChat,
  type Settings,
  fresh,
} from './chat-state';

const WANDER = 'This model has not learned to chat. Its answers may wander.';

function Intro() {
  return (
    <div className="page-intro">
      <h1 className="h1">Chat</h1>
      <p className="lede">
        Talk to a model you trained or imported, or put two side by side to see how they answer the
        same message.
      </p>
    </div>
  );
}

export default function ChatPage() {
  const models = useQuery({ queryKey: ['models', 'all'], queryFn: api.listAllModels });
  const ready = (models.data?.items ?? []).filter((model) => model.ready);
  if (ready.length) return <Conversation models={ready} />;
  return (
    <>
      <Intro />
      {models.error ? (
        <ErrorBox message={models.error.message} onRetry={() => void models.refetch()} />
      ) : models.isPending ? (
        <p className="muted" role="status">
          Loading models…
        </p>
      ) : (
        <Empty
          title="No model to talk to yet"
          hint="Train one, or import one from Hugging Face on the Models page. It shows up here once it has a save point."
          action={
            <span className="actions" style={{ justifyContent: 'center' }}>
              <Link className="btn btn-primary" to="/train">
                Train a model
              </Link>
              <Link className="btn" to="/models">
                Models
              </Link>
            </span>
          }
        />
      )}
    </>
  );
}

// saved in this browser for next time
function keep(chat: Chat) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(chat));
  } catch {
    // private window, fine
  }
}

function kinds(model: Model) {
  return [
    model.chat ? 'chat' : '',
    model.parent_name ? `from ${model.parent_name}` : '',
    model.source === 'hub' ? 'imported' : '',
  ].filter(Boolean);
}

// "Letters chat (4.2M weights, chat, from Letters model)"
function optionLabel(model: Model) {
  return `${model.name} (${[`${compactNumber(model.param_count)} weights`, ...kinds(model)].join(', ')})`;
}

// saved chat, but the url picks the models
function opening(params: URLSearchParams): Chat {
  let kept: string | null = null;
  try {
    kept = window.localStorage.getItem(KEY);
  } catch {
    // private window
  }
  const chat = restore(kept);
  const model = params.get('model');
  const other = params.get('with');
  if (!model && !other) return chat;
  return { ...chat, models: [model ?? chat.models[0], other ?? chat.models[1]], side: !!other };
}

// empty means not set
function NumberSetting({
  label,
  value,
  onChange,
  hint,
  error,
  placeholder,
  step = 'any',
}: {
  label: string;
  value: number | null;
  onChange: (next: number | null) => void;
  hint: string;
  error?: string;
  placeholder?: string;
  step?: string;
}) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      <input
        type="number"
        className="input"
        aria-label={label}
        aria-invalid={error ? true : undefined}
        step={step}
        placeholder={placeholder}
        value={value ?? ''}
        onChange={(event) =>
          onChange(event.target.value === '' ? null : Number(event.target.value))
        }
      />
      <span className={`hint${error ? ' chat-bad' : ''}`}>{error ?? hint}</span>
    </label>
  );
}

function Conversation({ models }: { models: Model[] }) {
  const [params, setParams] = useSearchParams();
  const [chat, dispatch] = useReducer(reduce, params, (initialParams) => {
    const history = loadHistory();
    const active = history.conversations.find((saved) => saved.id === history.activeId);

    return active ? active.chat : opening(initialParams);
  });
  const [history, setHistory] = useState(loadHistory);
  useEffect(() => {
    saveHistory(history);
  }, [history]);
  const [draft, setDraft] = useState('');
  const controllers = useRef(new Set<AbortController>());
  const box = useRef<HTMLTextAreaElement>(null);
  const log = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const chatRef = useRef(chat);
  chatRef.current = chat;

  const switchingRef = useRef(false);

  useEffect(() => {
    if (switchingRef.current) {
      switchingRef.current = false;
      return;
    }

    if (!history.activeId) return;

    setHistory((previous) => ({
      ...previous,
      conversations: previous.conversations.map((saved) => {
        if (saved.id !== previous.activeId) return saved;

        return {
          ...saved,
          title: chat.exchanges[0]?.ask.slice(0, 60) || saved.title,
          updatedAt: new Date().toISOString(),
          chat,
        };
      }),
    }));
  }, [chat]);
  const byId = new Map(models.map((model) => [model.id, model]));
  const first = byId.get(chat.models[0]) ?? models.find((model) => model.chat) ?? models[0];
  const parent = first.parent_id ? byId.get(first.parent_id) : undefined;
  const second =
    byId.get(chat.models[1]) ?? parent ?? models.find((model) => model.id !== first.id) ?? first;
  const columns = chat.side ? [first, second] : [first];
  const writing = busy(chat);
  const bad = problems(chat.settings);
  const invalid = Object.keys(bad).length > 0;

  // swap out a model that got deleted
  useEffect(() => {
    if (chat.models[0] !== first.id) dispatch({ type: 'model', column: 0, id: first.id });
    if (chat.models[1] !== second.id) dispatch({ type: 'model', column: 1, id: second.id });
  }, [chat.models, first.id, second.id]);

  // keep the url in sync with who's answering
  useEffect(() => {
    const want: Record<string, string> = { model: first.id };
    if (chat.side) want.with = second.id;
    if (params.get('model') !== want.model || params.get('with') !== (want.with ?? null))
      setParams(want, { replace: true });
  }, [chat.side, first.id, second.id, params, setParams]);

  // save when it's done, not on every piece
  const latest = useRef(chat);
  useEffect(() => {
    latest.current = chat;
    if (!writing) keep(chat);
  }, [chat, writing]);

  // leaving stops the stream but keeps what came in
  useEffect(() => {
    const running = controllers.current;
    const said = latest;
    return () => {
      running.forEach((control) => control.abort());
      keep(said.current);
    };
  }, []);

  // focus the box again once it's done
  const wasWriting = useRef(writing);
  useEffect(() => {
    if (wasWriting.current && !writing) box.current?.focus();
    wasWriting.current = writing;
  }, [writing]);

  // stick to the bottom unless you scrolled up
  useLayoutEffect(() => {
    const view = log.current;
    if (view && follow.current) view.scrollTop = view.scrollHeight;
  }, [chat.exchanges, chat.side]);

  const act = (action: Action) => {
    const next = reduce(chat, action);
    dispatch(action);
    return next;
  };
  const run = (next: Chat, at: number, which: number[]) => {
    const control = new AbortController();
    controllers.current.add(control);
    void Promise.all(
      which.map((column) => answer(next, at, column, dispatch, control.signal)),
    ).then(() => controllers.current.delete(control));
  };
  const stop = () => {
    controllers.current.forEach((control) => control.abort());
    controllers.current.clear();
    dispatch({ type: 'stop' });
  };
  const send = () => {
    const text = draft.trim();
    if (!text || writing || invalid) return;
    const next = act({ type: 'ask', text, seed: chat.settings.seed ?? newSeed() });
    if (!history.activeId) {
      const now = new Date().toISOString();
      const id = crypto.randomUUID();

      setHistory((previous) => ({
        activeId: id,
        conversations: [
          {
            id,
            title: text.slice(0, 60),
            createdAt: now,
            updatedAt: now,
            chat: next,
          },
          ...previous.conversations,
        ],
      }));
    }
    setDraft('');
    follow.current = true;
    run(next, next.exchanges.length - 1, chat.side ? [0, 1] : [0]);
  };
  const retry = (at: number, column: number) => {
    run(act({ type: 'retry', at, column }), at, [column]);
  };
  const set = (settings: Partial<Settings>) => dispatch({ type: 'settings', settings });
  const newChat = () => {
    stop();

    const now = new Date().toISOString();
    const id = crypto.randomUUID();

    const next: Chat = {
      ...chat,
      exchanges: [],
    };

    const saved: SavedChat = {
      id,
      title: 'New conversation',
      createdAt: now,
      updatedAt: now,
      chat: next,
    };

    setHistory((previous) => {
      const conversations = [...previous.conversations];

      if (previous.activeId) {
        const index = conversations.findIndex((item) => item.id === previous.activeId);

        if (index >= 0) {
          conversations[index] = {
            ...conversations[index],
            chat,
            updatedAt: now,
          };
        }
      } else if (chat.exchanges.length) {
        conversations.push({
          id: crypto.randomUUID(),
          title: chat.exchanges[0].ask.slice(0, 60),
          createdAt: now,
          updatedAt: now,
          chat,
        });
      }

      return {
        activeId: id,
        conversations: [saved, ...conversations],
      };
    });

    dispatch({ type: 'load', chat: next });
    setDraft('');
    box.current?.focus();
  };

  const openChat = (saved: SavedChat) => {
    if (writing || history.activeId === saved.id) return;

    const now = new Date().toISOString();

    setHistory((previous) => ({
      ...previous,
      activeId: saved.id,
      conversations: previous.conversations.map((item) =>
        item.id === previous.activeId ? { ...item, chat: chatRef.current, updatedAt: now } : item,
      ),
    }));

    switchingRef.current = true;
    dispatch({ type: 'load', chat: saved.chat });
    setDraft('');
    follow.current = true;
  };
  const deleteChat = (id: string) => {
    const selected = history.conversations.find((item) => item.id === id);

    if (!selected) return;

    if (!window.confirm(`Delete "${selected.title}"? This cannot be undone.`)) {
      return;
    }
    const deletingActive = history.activeId === id;

    setHistory((previous) => ({
      activeId: deletingActive ? null : previous.activeId,
      conversations: previous.conversations.filter((item) => item.id !== id),
    }));

    if (deletingActive) {
      dispatch({ type: 'load', chat: fresh() });
      setDraft('');
    }
  };
  return (
    <>
      <Intro />
      <Card>
        <CardHead title="Chat history" />
        <div className="card-body stack">
          {history.conversations.length === 0 ? (
            <p className="muted">No saved conversations yet.</p>
          ) : (
            history.conversations.map((saved) => (
              <div key={saved.id} className="actions">
                <button
                  type="button"
                  className="btn"
                  disabled={writing}
                  onClick={() => openChat(saved)}
                >
                  {saved.title}
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={writing}
                  aria-label={`Delete ${saved.title}`}
                  onClick={() => deleteChat(saved.id)}
                >
                  Delete
                </button>
              </div>
            ))
          )}
        </div>
      </Card>

      <Card>
        <CardHead
          title={chat.side ? 'Models' : 'Model'}
          end={
            <>
              <label className="help-toggle">
                <input
                  type="checkbox"
                  role="switch"
                  checked={chat.side}
                  disabled={writing}
                  onChange={(event) =>
                    dispatch({
                      type: 'side',
                      on: event.target.checked,
                      second: event.target.checked ? (parent ?? second).id : undefined,
                    })
                  }
                />
                Side by side
              </label>
              <button
                type="button"
                className="btn btn-sm"
                disabled={!chat.exchanges.length}
                onClick={newChat}
              >
                New chat
              </button>
            </>
          }
        />
        <div className="card-body dataset-form">
          {columns.map((model, column) => {
            const label = chat.side ? (column ? 'Second model' : 'First model') : 'Who answers';
            return (
              <label key={column} className="field">
                <span className="label">{label}</span>
                <select
                  className="input"
                  aria-label={label}
                  value={model.id}
                  disabled={writing}
                  onChange={(event) => dispatch({ type: 'model', column, id: event.target.value })}
                >
                  {models.map((one) => (
                    <option key={one.id} value={one.id}>
                      {optionLabel(one)}
                    </option>
                  ))}
                </select>
                {model.chat ? null : <span className="hint">{WANDER}</span>}
              </label>
            );
          })}
        </div>
      </Card>

      <details className="disclosure card" open={invalid || undefined}>
        <summary>Settings</summary>
        <div className="card-body stack">
          <label className="field">
            <span className="label">System message</span>
            <textarea
              className="input"
              aria-label="System message"
              rows={2}
              value={chat.settings.system}
              onChange={(event) => set({ system: event.target.value })}
            />
            <span className="hint">Sent first, before your messages.</span>
          </label>
          <div className="dataset-form">
            <NumberSetting
              label="Temperature"
              value={chat.settings.temperature}
              onChange={(temperature) => set({ temperature })}
              hint={HINTS.temperature}
              error={bad.temperature}
            />
            <NumberSetting
              label="Longest answer"
              step="1"
              value={chat.settings.max_new_tokens}
              onChange={(max_new_tokens) => set({ max_new_tokens })}
              hint={`In word pieces. ${HINTS.max_new_tokens}`}
              error={bad.max_new_tokens}
            />
            <NumberSetting
              label="Top p"
              value={chat.settings.top_p}
              onChange={(top_p) => set({ top_p })}
              hint={HINTS.top_p}
              error={bad.top_p}
            />
            <NumberSetting
              label="Seed"
              step="1"
              placeholder="New each message"
              value={chat.settings.seed}
              onChange={(seed) => set({ seed })}
              hint={HINTS.seed}
              error={bad.seed}
            />
          </div>
        </div>
      </details>

      <Card>
        <div
          ref={log}
          className={`chat-log${chat.side ? ' chat-two' : ''}`}
          role="log"
          aria-label="Conversation"
          aria-busy={writing}
          onScroll={(event) => {
            const view = event.currentTarget;
            follow.current = view.scrollHeight - view.scrollTop - view.clientHeight < 48;
          }}
        >
          {chat.side ? (
            <div className="chat-heads">
              {columns.map((model, column) => (
                <div key={column} className={`chat-head chat-col-${column}`}>
                  <strong>{model.name}</strong>
                  <span className="muted">
                    {[`${compactNumber(model.param_count)} weights`, ...kinds(model)].join(', ')}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
          {chat.exchanges.length ? (
            chat.exchanges.map((exchange, at) => (
              <div key={at} className="chat-exchange">
                <div className="chat-ask">
                  <span className="label">You</span>
                  <p className="chat-text">{exchange.ask}</p>
                </div>
                {columns.map((model, column) => (
                  <Reply
                    key={column}
                    column={column}
                    name={model.name}
                    answer={exchange.answers[column]}
                    retry={() => retry(at, column)}
                  />
                ))}
              </div>
            ))
          ) : (
            <p className="muted chat-start">
              {chat.side
                ? 'Write a message below. Both models answer it, side by side.'
                : 'Write a message below to start a conversation.'}
            </p>
          )}
        </div>

        <form
          className="chat-compose"
          onSubmit={(event) => {
            event.preventDefault();
            send();
          }}
        >
          <textarea
            ref={box}
            className="input chat-box"
            aria-label="Your message"
            rows={2}
            value={draft}
            placeholder="Write a message. Enter sends, Shift+Enter starts a new line."
            autoFocus
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return;
              event.preventDefault();
              send();
            }}
          />
          {writing ? (
            <button type="button" className="btn" onClick={stop}>
              Stop
            </button>
          ) : (
            <button type="submit" className="btn btn-primary" disabled={!draft.trim() || invalid}>
              Send
            </button>
          )}
        </form>
      </Card>
    </>
  );
}

function note(answer: Answer) {
  if (answer.state === 'stopped') return 'Stopped';
  const said = `${plural(answer.pieces ?? 0, 'word piece')} in ${(answer.seconds ?? 0).toFixed(1)} seconds`;
  return answer.long ? `${said}. Stopped at the length limit.` : said;
}

function Reply({
  column,
  name,
  answer,
  retry,
}: {
  column: number;
  name: string;
  answer: Answer | null;
  retry: () => void;
}) {
  return (
    <div className={`chat-reply chat-col-${column}`}>
      <span className="label chat-who">{name}</span>
      {answer === null ? (
        <p className="muted chat-note">Not asked</p>
      ) : (
        <>
          {answer.text || answer.state === 'writing' ? (
            <p className={`chat-text${answer.state === 'writing' ? ' chat-writing' : ''}`}>
              {answer.text}
            </p>
          ) : answer.state === 'done' ? (
            <p className="muted chat-note">No answer</p>
          ) : null}
          {answer.state === 'error' ? (
            <div className="stack">
              <Notice tone="bad">{answer.error}</Notice>
              <div>
                <button type="button" className="btn btn-sm" onClick={retry}>
                  Try again
                </button>
              </div>
            </div>
          ) : answer.state === 'writing' ? (
            <span className="muted chat-note">Writing…</span>
          ) : (
            <p className="muted chat-note">{note(answer)}</p>
          )}
        </>
      )}
    </div>
  );
}
