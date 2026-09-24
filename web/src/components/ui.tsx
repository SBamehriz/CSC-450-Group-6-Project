import { useRef, useState } from 'react';
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

export function Empty({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="empty">
      <p style={{ margin: 0, fontWeight: 500 }}>{title}</p>
      <p className="muted" style={{ margin: '4px 0 0' }}>
        {hint}
      </p>
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
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  maxLength?: number;
  width?: number | string;
}) {
  return (
    <label className="field" style={{ width }}>
      <span className="label">{label}</span>
      <input
        className="input"
        value={value}
        placeholder={placeholder}
        maxLength={maxLength}
        onChange={(event) => onChange(event.target.value)}
      />
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
        .txt .md .html .htm .pdf .docx .png .jpg .json .jsonl .gz · Up to 50 MB per file, 200 files at a time
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
