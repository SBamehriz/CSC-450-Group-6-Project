import { useMutation } from '@tanstack/react-query';

import { api } from '../api/client';

export default function SnapshotCheck({ datasetId }: { datasetId: string }) {
  const check = useMutation({ mutationFn: () => api.verifyDataset(datasetId) });
  return (
    <div className="stack">
      <button
        className="btn"
        type="button"
        disabled={check.isPending}
        onClick={() => check.mutate()}
      >
        {check.isPending ? 'Checking files…' : 'Check snapshot'}
      </button>
      {check.error ? (
        <p role="alert" className="notice notice-bad">
          {check.error.message}
        </p>
      ) : null}
      {check.data ? (
        <p role="status" className={check.data.valid ? 'hint' : 'notice notice-bad'}>
          {check.data.message}
        </p>
      ) : null}
    </div>
  );
}
