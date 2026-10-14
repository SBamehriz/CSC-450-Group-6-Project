import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

import { api } from '../api/client';
import { Card, CardHead, Chip, Dropzone, Notice } from './ui';

export default function UploadPanel({
  bucketId,
  onBusyChange,
  onUploaded,
}: {
  bucketId: string;
  onBusyChange?: (busy: boolean) => void;
  onUploaded?: () => void;
}) {
  const queryClient = useQueryClient();
  const [sourceNote, setSourceNote] = useState('');
  const [validation, setValidation] = useState('');
  const upload = useMutation({
    mutationFn: (files: File[]) => api.uploadDocuments(bucketId, files, sourceNote),
    onSuccess: () => onUploaded?.(),
    onSettled: () => {
      onBusyChange?.(false);
      // A lost response may follow a successful upload.
      for (const queryKey of [
        ['buckets'],
        ['bucket', bucketId],
        ['documents', bucketId],
        ['overview'],
        ['recentUploads'],
      ])
        void queryClient.invalidateQueries({ queryKey });
    },
  });

  const send = (files: File[]) => {
    if (upload.isPending) return;
    upload.reset();
    const tooLarge = files.find((file) => file.size > 50 * 1024 * 1024);
    if (files.length > 200 || tooLarge) {
      setValidation(
        tooLarge ? `${tooLarge.name} is over 50 MB.` : 'Choose up to 200 files at a time.',
      );
      return;
    }
    setValidation('');
    onBusyChange?.(true);
    upload.mutate(files);
  };

  return (
    <Card>
      <CardHead title="Upload files" />
      <div className="card-body stack">
        <label className="field">
          <span className="label">Where these files came from</span>
          <input
            className="input"
            value={sourceNote}
            maxLength={2000}
            disabled={upload.isPending}
            placeholder="Source or download notes"
            onChange={(event) => setSourceNote(event.target.value)}
          />
        </label>
        <Dropzone onFiles={send} busy={upload.isPending} />
        {upload.isPending ? (
          <p className="muted" role="status">
            Uploading and parsing {upload.variables?.length} file(s). OCR can take a little longer.
          </p>
        ) : null}
        {validation ? <Notice tone="bad">{validation}</Notice> : null}
        {upload.error ? (
          <Notice tone="bad">
            {upload.error.message} Check the document list before trying the same files again.
          </Notice>
        ) : null}
        {upload.data ? (
          <div className="stack" role="status">
            <div>
              <Chip tone={upload.data.failed ? 'warn' : 'ok'}>
                {upload.data.parsed} parsed, {upload.data.failed} failed
              </Chip>
            </div>
            <ul className="upload-outcomes">
              {upload.data.outcomes.map((outcome, index) => (
                <li key={index}>
                  <strong>{outcome.filename}</strong>:{' '}
                  {outcome.status === 'parsed' ? 'Parsed' : outcome.error || 'Failed'}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </Card>
  );
}
