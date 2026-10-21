'use client';

import { useEffect, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';

export interface WorkerRelaunchOperation {
  operation_id: string;
  source_worker: string;
  target_worker?: string;
  status: 'starting' | 'transferring' | 'pending' | 'succeeded' | 'failed';
  error?: string;
}

export function relaunchError(error: unknown): string {
  if (isAxiosError(error)) return error.response?.data?.detail || error.message;
  return error instanceof Error ? error.message : String(error);
}

export function useWorkerRelaunches(enabled: boolean) {
  const [operations, setOperations] = useState<WorkerRelaunchOperation[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!enabled) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const data = await request.get<WorkerRelaunchOperation[]>('/v1/worker-relaunches', {
          suppressGlobalError: true,
        });
        if (active) {
          setOperations(data);
          setError('');
        }
      } catch (err) {
        if (active) setError(relaunchError(err));
      } finally {
        if (active) timer = setTimeout(poll, 2000);
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [enabled]);
  const busyAddresses = new Set(
    operations
      .filter((job) => !['succeeded', 'failed'].includes(job.status))
      .flatMap((job) => [job.source_worker, job.target_worker])
  );
  return {
    operations,
    error,
    busyAddresses,
    addOperation: (job: WorkerRelaunchOperation) =>
      setOperations((current) => [
        job,
        ...current.filter((item) => item.operation_id !== job.operation_id),
      ]),
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
    <Dialog
      open={address !== null}
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('clusterInfo.relaunch.title')}</DialogTitle>
          <DialogDescription>{t('clusterInfo.relaunch.description')}</DialogDescription>
        </DialogHeader>
        <p className="break-all text-sm">{address}</p>
        <label className="space-y-2 text-sm">
          <span>{t('clusterInfo.relaunch.python')}</span>
          <Input
            aria-label={t('clusterInfo.relaunch.python')}
            value={python}
            disabled={busy}
            onChange={(event) => setPython(event.target.value)}
            placeholder="/opt/xinference/bin/python"
          />
        </label>
        <p className="text-sm text-muted-foreground">{t('clusterInfo.relaunch.environmentHint')}</p>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={onClose}>
            {t('common.cancel')}
          </Button>
          <Button disabled={busy} onClick={submit}>
            {t('clusterInfo.relaunch.title')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
          {job.status === 'pending' && <p>{t('clusterInfo.relaunch.pendingHint')}</p>}
          {job.error && <p className="text-destructive">{job.error}</p>}
        </div>
      ))}
    </div>
  );
}
