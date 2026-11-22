import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useId, useRef, useState } from 'react';

import { ApiError, api } from '../api/client';
import { bytesToSize, plural } from '../lib/format';
import { ErrorBox, Notice } from './ui';

export interface DeleteTarget {
  id: string;
  name: string;
}

function formatJobType(jobType: string): string {
  if (jobType === 'dataset_import') return 'Dataset import';
  if (jobType === 'snapshot_create') return 'Snapshot build';
  return jobType;
}

export default function DeleteCollectionDialog({
  target,
  onClose,
  onDeleted,
}: {
  target: DeleteTarget | null;
  onClose: () => void;
  onDeleted?: (target: DeleteTarget) => void;
}) {
  const queryClient = useQueryClient();
  const titleId = useId();
  const [scopeNotice, setScopeNotice] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [cleanupIncomplete, setCleanupIncomplete] = useState(false);
  const submittingRef = useRef(false);

  const summaryQuery = useQuery({
    queryKey: ['bucketDeletionSummary', target?.id],
    queryFn: () => api.bucketDeletionSummary(target!.id),
    enabled: !!target,
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });

  const remove = useMutation({
    mutationFn: (variables: { id: string; scopeToken: string }) =>
      api.deleteBucket(variables.id, { confirm: true, scopeToken: variables.scopeToken }),
  });

  const targetId = target?.id ?? null;
  const resetMutation = remove.reset;

  useEffect(() => {
    submittingRef.current = false;
    setScopeNotice(null);
    setDeleteError(null);
    setCleanupIncomplete(false);
    resetMutation();
  }, [targetId, resetMutation]);

  useEffect(() => {
    if (!target) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !remove.isPending) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [target, remove.isPending, onClose]);

  if (!target) return null;

  const summary = summaryQuery.data;
  const blockedByJobs = (summary?.active_jobs.length ?? 0) > 0;
  const confirmDisabled =
    remove.isPending ||
    summaryQuery.isPending ||
    summaryQuery.isFetching ||
    Boolean(summaryQuery.error) ||
    !summary?.scope_token ||
    blockedByJobs;

  const handleConfirm = async () => {
    if (
      submittingRef.current ||
      !summary?.scope_token ||
      Boolean(summaryQuery.error) ||
      remove.isPending ||
      summaryQuery.isFetching ||
      blockedByJobs
    ) {
      return;
    }
    submittingRef.current = true;
    setScopeNotice(null);
    setDeleteError(null);

    try {
      await remove.mutateAsync({ id: target.id, scopeToken: summary.scope_token });
      queryClient.removeQueries({ queryKey: ['bucket', target.id] });
      queryClient.removeQueries({ queryKey: ['documents', target.id] });
      void queryClient.invalidateQueries({ queryKey: ['buckets'] });
      void queryClient.invalidateQueries({ queryKey: ['overview'] });
      void queryClient.invalidateQueries({ queryKey: ['recentUploads'] });
      void queryClient.invalidateQueries({ queryKey: ['datasets'] });
      void queryClient.invalidateQueries({ queryKey: ['jobs'] });
      onDeleted?.(target);
      onClose();
    } catch (error) {
      if (error instanceof ApiError && error.code === 'scope_changed') {
        setScopeNotice(
          'This collection changed after the summary was loaded. Review the updated totals below before confirming again.',
        );
        await summaryQuery.refetch();
        return;
      }
      if (error instanceof ApiError && error.code === 'bucket_busy') {
        setDeleteError(error.message);
        await summaryQuery.refetch();
        return;
      }
      if (error instanceof ApiError && error.code === 'cleanup_failed') {
        setCleanupIncomplete(true);
        setDeleteError(error.message);
        await summaryQuery.refetch();
        return;
      }
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete collection.');
    } finally {
      submittingRef.current = false;
    }
  };

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget && !remove.isPending) {
          onClose();
        }
      }}
    >
      <div className="modal-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="modal-header">
          <h2 className="h2" id={titleId}>
            Delete “{summary?.bucket_name ?? target.name}”?
          </h2>
        </div>

        <p className="muted" style={{ marginTop: 0 }}>
          This permanently removes the collection, all of its document records, and exclusively
          owned source and cleaned text files from disk.
        </p>

        {summaryQuery.isPending ? (
          <p className="muted" role="status">
            Calculating files and storage space to remove…
          </p>
        ) : summaryQuery.error ? (
          <ErrorBox
            message={
              summaryQuery.error instanceof Error
                ? summaryQuery.error.message
                : 'Could not load deletion summary.'
            }
            onRetry={() => void summaryQuery.refetch()}
          />
        ) : summary ? (
          <>
            <dl className="modal-stats">
              <div>
                <dt>Collection</dt>
                <dd>{summary.bucket_name}</dd>
              </div>
              <div>
                <dt>Documents</dt>
                <dd>{plural(summary.document_count, 'document')}</dd>
              </div>
              <div>
                <dt>Files on disk</dt>
                <dd>{plural(summary.physical_file_count, 'file')}</dd>
              </div>
              <div>
                <dt>Disk space freed</dt>
                <dd>{bytesToSize(summary.total_bytes)}</dd>
              </div>
            </dl>

            <div className="modal-notes">
              <span>
                Includes {plural(summary.raw_file_count, 'source upload')} (
                {bytesToSize(summary.raw_bytes)}) and{' '}
                {plural(summary.text_file_count, 'cleaned text file')} (
                {bytesToSize(summary.text_bytes)}).
              </span>
              {summary.shared_file_count > 0 ? (
                <span>
                  {plural(summary.shared_file_count, 'shared raw file')} still referenced by other
                  collections will be preserved.
                </span>
              ) : null}
              {summary.missing_file_count > 0 ? (
                <span>
                  {plural(summary.missing_file_count, 'file')} already missing from disk will be
                  skipped.
                </span>
              ) : null}
              <span>
                Saved snapshots, other collections, trained models, and checkpoints remain intact.
              </span>
            </div>

            {blockedByJobs ? (
              <Notice tone="bad">
                Cannot delete while active jobs target this collection:{' '}
                {summary.active_jobs
                  .map((job) => `${formatJobType(job.job_type)} (${job.status})`)
                  .join(', ')}
                . Wait for completion or cancel them first, then refresh.
              </Notice>
            ) : null}
          </>
        ) : null}

        {scopeNotice ? <Notice tone="warn">{scopeNotice}</Notice> : null}
        {deleteError ? <Notice tone="bad">{deleteError}</Notice> : null}

        <div className="modal-actions">
          {blockedByJobs ? (
            <button
              type="button"
              className="btn"
              disabled={summaryQuery.isFetching || remove.isPending}
              onClick={() => void summaryQuery.refetch()}
            >
              {summaryQuery.isFetching ? 'Refreshing…' : 'Refresh summary'}
            </button>
          ) : null}
          <button type="button" className="btn" disabled={remove.isPending} onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-danger"
            disabled={confirmDisabled}
            onClick={() => void handleConfirm()}
          >
            {remove.isPending
              ? 'Deleting collection…'
              : cleanupIncomplete
                ? 'Retry delete'
                : 'Delete collection'}
          </button>
        </div>
      </div>
    </div>
  );
}
