import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { api } from '../api/client';
import { ErrorBox } from './ui';

const WINDOW = 4000;

export default function TextPreview({ documentId }: { documentId: string }) {
  const [offset, setOffset] = useState(0);
  const text = useQuery({
    queryKey: ['documentText', documentId, offset],
    queryFn: () => api.documentText(documentId, offset, WINDOW),
  });
  return (
    <section aria-label="Cleaned text">
      <h3 className="h2">Cleaned text</h3>
      {text.error ? (
        <ErrorBox message={text.error.message} onRetry={() => void text.refetch()} />
      ) : text.isPending ? (
        <p className="muted" role="status">
          Loading cleaned text…
        </p>
      ) : (
        <>
          <pre className="preview">{text.data.text}</pre>
          <div className="document-pager">
            <button
              type="button"
              className="btn"
              disabled={offset === 0 || text.isFetching}
              onClick={() => setOffset(Math.max(0, offset - WINDOW))}
            >
              Previous text
            </button>
            <span className="muted" role="status">
              Characters {text.data.length ? offset + 1 : 0} to {offset + text.data.length} of{' '}
              {text.data.total}
            </span>
            <button
              type="button"
              className="btn"
              disabled={offset + text.data.length >= text.data.total || text.isFetching}
              onClick={() => setOffset(offset + WINDOW)}
            >
              Next text
            </button>
          </div>
        </>
      )}
    </section>
  );
}
