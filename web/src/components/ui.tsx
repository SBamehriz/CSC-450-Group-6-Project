import { useId, useRef, useState } from 'react';
import type { ReactNode } from 'react';

export function Card({ children }: { children: ReactNode }) {
  return <div className="card">{children}</div>;
}

export function CardHead({ title, end }: { title: string; end?: ReactNode }) {
  return (
    <div className="card-head">
      <h2 className="h2">{title}</h2>
      {end ? <div className="end">{end}</div> : null}
    </div>
  );
}

export type Tone = 'ok' | 'warn' | 'bad' | 'idle';

export function Chip({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <span className={`chip chip-${tone}`}>
      <i aria-hidden="true" />
      {children}
    </span>
  );
}

export function Notice({ tone, children }: { tone: 'bad' | 'warn'; children: ReactNode }) {
  return (
    <p className={`notice notice-${tone}`} role={tone === 'bad' ? 'alert' : undefined}>
      {children}
    </p>
  );
}

export function Empty({
  title,
  hint,
  action,
}: {
  title: string;
  hint: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <p style={{ margin: 0, fontWeight: 500 }}>{title}</p>
      <p className="muted" style={{ margin: '4px 0 0' }}>
        {hint}
      </p>
      {action ? <div className="empty-action">{action}</div> : null}
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="notice notice-bad" role="alert">
      <p style={{ margin: 0 }}>{message}</p>
      <button type="button" className="btn btn-link" style={{ marginTop: 6 }} onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

export function Field({
  label,
  value,
  onChange,
  placeholder,
  maxLength,
  width,
  hint,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  maxLength?: number;
  width?: number | string;
  hint?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <label className="field" style={{ width }}>
      <span className="label">{label}</span>
      <input
        className="input"
        disabled={disabled}
        aria-label={label}
        aria-describedby={hint ? `${id}-hint` : undefined}
        value={value}
        placeholder={placeholder}
        maxLength={maxLength}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint ? (
        <span className="hint" id={`${id}-hint`}>
          {hint}
        </span>
      ) : null}
    </label>
  );
}

export function Dropzone({
  onFiles,
  busy,
  compact = false,
}: {
  onFiles: (files: File[]) => void;
  busy: boolean;
  compact?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const send = (list: FileList | null) => {
    const files = list ? Array.from(list) : [];
    if (!busy && files.length) onFiles(files);
  };

  return (
    <div
      className={`empty dropzone${compact ? ' dropzone-compact' : ''}${dragging && !busy ? ' dropzone-active' : ''}`}
      aria-busy={busy}
      onDragOver={(event) => {
        event.preventDefault();
        if (!busy) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        send(event.dataTransfer.files);
      }}
    >
      <Icon name="upload" />
      <p className="dropzone-title">
        <span role="status">{busy ? 'Parsing…' : 'Drop files here, or '}</span>
        <button
          type="button"
          className="btn btn-link"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
        >
          choose files
        </button>
      </p>
      <p className="muted dropzone-hint">
        .txt .md .html .htm .pdf .docx .png .jpg .json .jsonl .gz · Up to 50 MB per file, 200 files
        at a time
      </p>
      <input
        ref={inputRef}
        type="file"
        multiple
        hidden
        disabled={busy}
        aria-label="Files to upload"
        accept=".txt,.md,.html,.htm,.pdf,.docx,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp,.json,.jsonl,.gz"
        onChange={(event) => {
          send(event.target.files);
          event.target.value = '';
        }}
      />
    </div>
  );
}

const ICONS = {
  home: 'M4 10.5 12 4l8 6.5V20h-5v-6H9v6H4z',
  data: 'M12 4 4 8l8 4 8-4zM4 12l8 4 8-4M4 16l8 4 8-4',
  snapshot: 'M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h6',
  upload: 'M12 16V4M7 9l5-5 5 5M4 15v5h16v-5',
} as const;

export function Icon({ name }: { name: keyof typeof ICONS }) {
  return (
    <svg
      className="icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={ICONS[name]} />
    </svg>
  );
}

export function ConfirmButton({
  label,
  confirmLabel,
  onConfirm,
  disabled,
}: {
  label: string;
  confirmLabel: string;
  onConfirm: () => void;
  disabled?: boolean;
}) {
  const [armed, setArmed] = useState(false);
  return armed ? (
    <span className="actions">
      <button type="button" className="btn btn-danger" disabled={disabled} onClick={onConfirm}>
        {confirmLabel}
      </button>
      <button type="button" className="btn" disabled={disabled} onClick={() => setArmed(false)}>
        Keep it
      </button>
    </span>
  ) : (
    <button type="button" className="btn" disabled={disabled} onClick={() => setArmed(true)}>
      {label}
    </button>
  );
}

export function ExplainToggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="help-toggle">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      Explain what I’m seeing
    </label>
  );
}
