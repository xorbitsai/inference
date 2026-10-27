'use client';

import { useEffect, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { Server } from 'lucide-react';
import { ComponentRestartDialog } from '@/components/pages/component-restart-dialog';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';

export interface WorkerRelaunchOperation {
  operation_id: string;
  source_worker: string;
  target_worker?: string;
  status:
    | 'installing'
    | 'migrating'
    | 'starting'
    | 'transferring'
    | 'pending'
    | 'succeeded'
    | 'failed';
  error?: string;
  log?: string;
  requested_version?: string;
  source_version?: string;
  target_version?: string;
}

function relaunchError(error: unknown): string {
  if (isAxiosError(error)) return error.response?.data?.detail || error.message;
  return error instanceof Error ? error.message : String(error);
}

export function useWorkerRelaunches(enabled: boolean) {
  const [operations, setOperations] = useState<WorkerRelaunchOperation[]>([]);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    if (!enabled) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      let pending = false;
      try {
        const data = await request.get<WorkerRelaunchOperation[]>('/v1/worker-relaunches', {
          suppressGlobalError: true,
        });
        if (active) {
          setOperations(data);
          setError('');
          pending = data.some((job) => !['succeeded', 'failed'].includes(job.status));
        }
      } catch (err) {
        if (active) setError(relaunchError(err));
        pending = true;
      } finally {
        if (active && pending) timer = setTimeout(poll, 2000);
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [enabled, revision]);
  const busyAddresses = new Set(
    operations
      .filter((job) => !['succeeded', 'failed'].includes(job.status))
      .flatMap((job) => [job.source_worker, job.target_worker])
  );
  return {
    operations,
    error,
    busyAddresses,
    addOperation: (job: WorkerRelaunchOperation) => {
      setOperations((current) => [
        job,
        ...current.filter((item) => item.operation_id !== job.operation_id),
      ]);
      setRevision((current) => current + 1);
    },
  };
}

export function WorkerRelaunchDialog({
  address,
  onClose,
  onStarted,
}: {
  address: string | null;
  onClose: () => void;
  onStarted: (operation: WorkerRelaunchOperation) => void;
}) {
  const { t } = useI18n();
  const [python, setPython] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const requestId = useRef('');
  useEffect(() => {
    requestId.current =
      crypto.randomUUID?.() ?? `worker-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setPython('');
    setError('');
  }, [address]);
  const submit = async () => {
    if (!address || busy) return;
    setBusy(true);
    setError('');
    try {
      const job = await request.post<WorkerRelaunchOperation>(
        `/v1/workers/${encodeURIComponent(address)}/relaunch`,
        {
          request_id: requestId.current,
          ...(python.trim() ? { python_executable: python.trim() } : {}),
        },
        { suppressGlobalError: true }
      );
      onStarted(job);
      onClose();
    } catch (err) {
      setError(relaunchError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <ComponentRestartDialog
      open={address !== null}
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
      title={t('clusterInfo.relaunch.title')}
      description={t('clusterInfo.relaunch.description')}
      icon={<Server />}
      target={address}
      python={python}
      onPythonChange={setPython}
      environmentHint={t('clusterInfo.relaunch.environmentHint')}
      busy={busy}
      error={error}
      onSubmit={submit}
    />
  );
}

export function WorkerRelaunchHistory({
  operations,
  error,
}: {
  operations: WorkerRelaunchOperation[];
  error: string;
}) {
  const { t } = useI18n();
  if (!operations.length && !error) return null;
  return (
    <div className="mt-4 space-y-2" aria-label={t('clusterInfo.relaunch.history')}>
      <h3 className="font-medium">{t('clusterInfo.relaunch.history')}</h3>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {operations.slice(0, 10).map((job) => (
        <div key={job.operation_id} className="rounded-md border p-3 text-sm">
          <p className="break-all">
            {job.source_worker}
            {job.target_worker ? ` → ${job.target_worker}` : ''}
          </p>
          <p role="status">{t(`clusterInfo.relaunch.${job.status}`)}</p>
          {job.requested_version && (
            <p>
              {job.source_version ? `${job.source_version} → ` : ''}
              {job.target_version || job.requested_version}
            </p>
          )}
          {job.status === 'pending' && <p>{t('clusterInfo.relaunch.pendingHint')}</p>}
          {job.error && <p className="text-destructive">{job.error}</p>}
          {job.log && (
            <details>
              <summary className="cursor-pointer">{t('clusterInfo.relaunch.installLog')}</summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all text-xs">
                {job.log}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}
