// @vitest-environment jsdom
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { api, type Run } from '../api/client';
import { open } from '../test-utils';
import TrainingHistoryPage from './TrainingHistoryPage';

const mockRuns: Run[] = [
  {
    id: 'run-1',
    name: 'run-science-1',
    model_id: 'm-1',
    model_name: 'nano-science',
    model_preset: 'nano',
    dataset_id: 'ds-1',
    dataset_name: 'Science papers',
    status: 'completed',
    device: 'cpu',
    batch_size: 2,
    learning_rate: 0.001,
    max_steps: 1000,
    current_step: 1000,
    loss: 1.82,
    eval_loss: 1.95,
    best_val_loss: 1.842,
    tokens_per_sec: 14500,
    metrics: {},
    hyperparams: {},
    error: null,
    started_at: '2026-10-23T10:00:00Z',
    completed_at: '2026-10-23T10:30:00Z',
    created_at: '2026-10-23T10:00:00Z',
  },
  {
    id: 'run-2',
    name: 'run-pending-2',
    model_id: 'm-2',
    model_name: 'micro-model',
    model_preset: 'micro',
    dataset_id: null,
    dataset_name: null,
    status: 'pending',
    device: 'cpu',
    batch_size: 1,
    learning_rate: 0.001,
    max_steps: 500,
    current_step: 0,
    loss: null,
    eval_loss: null,
    best_val_loss: null, // validation loss unavailable
    tokens_per_sec: null,
    metrics: {},
    hyperparams: {},
    error: null,
    started_at: null,
    completed_at: null,
    created_at: '2026-10-23T09:00:00Z',
  },
];

beforeEach(() => {
  vi.spyOn(api, 'listRuns').mockResolvedValue({ items: mockRuns, total: 2 });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function showPage() {
  return open(
    <Routes>
      <Route path="/history" element={<TrainingHistoryPage />} />
    </Routes>,
    '/history',
  );
}

it('lists runs showing model, dataset, status, and best val loss', async () => {
  showPage();

  await screen.findByText('run-science-1');
  expect(screen.getByText(/nano-science/)).toBeTruthy();
  expect(screen.getByText('Science papers')).toBeTruthy();
  expect(screen.getByText('completed')).toBeTruthy();
  // Validated loss displayed
  expect(screen.getByText('1.8420')).toBeTruthy();

  // Links to the run page
  const link = screen.getByRole('link', { name: /run-science-1/i });
  expect(link.getAttribute('href')).toBe('/runs/run-1');
});

it('shows a clear empty dash when validation loss is unavailable', async () => {
  showPage();

  await screen.findByText('run-pending-2');
  // Second run has no validation loss -> displays em-dash '—'
  const dashes = screen.getAllByLabelText('Validation loss unavailable');
  expect(dashes.length).toBeGreaterThanOrEqual(1);
  expect(dashes[0].textContent).toBe('—');
});

it('displays useful explanation when deletion is blocked by dependent checkpoints', async () => {
  vi.spyOn(api, 'deleteRun').mockRejectedValue(
    new Error(
      "Cannot delete run 'run-science-1' because it has 2 dependent checkpoint result(s). Remove checkpoint results before deleting the run.",
    ),
  );

  showPage();

  await screen.findByText('run-science-1');
  const deleteBtns = screen.getAllByRole('button', { name: 'Delete' });
  fireEvent.click(deleteBtns[0]);

  const confirmBtn = await screen.findByRole('button', { name: 'Delete run-science-1' });
  fireEvent.click(confirmBtn);

  // Expect useful explanation in notice
  expect(
    await screen.findByText(
      /Cannot delete run 'run-science-1' because it has 2 dependent checkpoint result\(s\)/,
    ),
  ).toBeTruthy();
});

it('successfully deletes a clean run upon confirmation', async () => {
  const deleteSpy = vi.spyOn(api, 'deleteRun').mockResolvedValue(undefined);

  showPage();

  await screen.findByText('run-pending-2');
  const deleteBtns = screen.getAllByRole('button', { name: 'Delete' });
  fireEvent.click(deleteBtns[1]);

  const confirmBtn = await screen.findByRole('button', { name: 'Delete run-pending-2' });
  fireEvent.click(confirmBtn);

  await waitFor(() => {
    expect(deleteSpy).toHaveBeenCalledWith('run-2');
  });
});
