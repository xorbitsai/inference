'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { Check, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';
import type { ClusterVersion } from '@/types/services';

type Component = 'worker' | 'supervisor' | 'api';
interface UpdateInfo {
  current_version: string;
  latest_version: string;
  update_available: boolean;
}
interface Target {
  key: string;
  component: Component;
  address?: string;
  update?: UpdateInfo;
  status: string;
  error?: string;
  log?: string;
}
interface Operation {
  operation_id: string;
  request_id?: string;
  requested_version?: string;
  status: string;
  error?: string;
  log?: string;
  source_worker?: string;
  target_worker?: string;
}
const terminal = (status: string) => status === 'succeeded' || status === 'failed';
const path = (target: Target) =>
  target.component === 'worker'
    ? `workers/${encodeURIComponent(target.address!)}`
    : target.component;
const history = (target: Target) => `/v1/${target.component}-relaunches`;
function message(error: unknown): string {
  const detail = isAxiosError(error) ? error.response?.data?.detail : undefined;
  return typeof detail === 'string'
    ? detail
    : error instanceof Error
      ? error.message
      : String(error);
}

export function SoftwareUpdateDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const { setClusterVersion, globalReady } = useGlobal();
  const { isAdmin } = useMenuAuth();
  const [checking, setChecking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [targets, setTargets] = useState<Target[]>([]);
  const [error, setError] = useState('');
  const [updated, setUpdated] = useState(false);
  const lifecycle = useRef({ mounted: true, revision: 0 });
  const wasOpen = useRef(false);
  const prefix = 'softwareUpdate';
  useEffect(() => {
    const current = lifecycle.current;
    current.mounted = true;
    return () => {
      current.mounted = false;
      current.revision++;
    };
  }, []);
  const patch = useCallback((key: string, fields: Partial<Target>) => {
    if (lifecycle.current.mounted)
      setTargets((current) =>
        current.map((target) => (target.key === key ? { ...target, ...fields } : target))
      );
  }, []);
  const check = useCallback(async () => {
    if (busy || checking || !globalReady || !isAdmin) return;
    const revision = ++lifecycle.current.revision;
    setChecking(true);
    setTargets([]);
    setError('');
    setUpdated(false);
    try {
      const workers = await request.get<{ 'work-ip': string }[]>('/v1/workers', {
        suppressGlobalError: true,
      });
      // The API goes last: its version changes after the other handoffs finish.
      const components: Target[] = [
        ...workers.map((worker) => ({
          key: worker['work-ip'],
          component: 'worker' as const,
          address: worker['work-ip'],
          status: 'checking',
        })),
        { key: 'supervisor', component: 'supervisor', status: 'checking' },
        { key: 'api', component: 'api', status: 'checking' },
      ];
      if (revision !== lifecycle.current.revision) return;
      setTargets(components);
      await Promise.all(
        components.map(async (target) => {
          try {
            const update = await request.get<UpdateInfo>(`/v1/${path(target)}/updates`, {
              suppressGlobalError: true,
              timeout: 90000,
            });
            if (revision === lifecycle.current.revision)
              patch(target.key, {
                update,
                status: update.update_available ? 'available' : 'current',
              });
          } catch (err) {
            if (revision === lifecycle.current.revision)
              patch(target.key, { status: 'failed', error: message(err) });
          }
        })
      );
    } catch (err) {
      if (revision === lifecycle.current.revision) setError(message(err));
    } finally {
      if (revision === lifecycle.current.revision && lifecycle.current.mounted) setChecking(false);
    }
  }, [busy, checking, globalReady, isAdmin, patch]);
  useEffect(() => {
    if (open && !wasOpen.current) void check();
    wasOpen.current = open;
  }, [open, check]);

  const upgrade = async () => {
    if (
      busy ||
      checking ||
      !isAdmin ||
      !globalReady ||
      targets.some((target) => target.status === 'failed')
    )
      return;
    setBusy(true);
    setError('');
    let currentKey: string | undefined;
    let confirmedFailure = false;
    try {
      for (const target of targets.filter(
        (target) => target.update?.update_available && target.status !== 'succeeded'
      )) {
        if (!lifecycle.current.mounted) return;
        currentKey = target.key;
        patch(target.key, { status: 'starting', error: undefined });
        const requestId =
          crypto.randomUUID?.() ?? `upgrade-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        let operation: Operation;
        try {
          operation = await request.post<Operation>(
            `/v1/${path(target)}/relaunch`,
            { version: target.update!.latest_version, request_id: requestId },
            { suppressGlobalError: true }
          );
        } catch (err) {
          // Recover an accepted job after a lost response, including API cutover.
          const jobs = await request.get<Operation[]>(history(target), {
            suppressGlobalError: true,
          });
          const accepted = jobs.find(
            (job) =>
              job.request_id === requestId ||
              (!terminal(job.status) &&
                job.requested_version === target.update!.latest_version &&
                (target.component !== 'worker' ||
                  job.source_worker === target.address ||
                  job.target_worker === target.address))
          );
          if (!accepted) throw err;
          operation = accepted;
        }
        for (;;) {
          patch(target.key, {
            status: operation.status,
            error: operation.error,
            log: operation.log,
          });
          if (operation.status === 'failed') {
            confirmedFailure = true;
            throw new Error(operation.error || t(`${prefix}.failed`));
          }
          if (operation.status === 'succeeded') break;
          await new Promise((resolve) => setTimeout(resolve, 1500));
          if (!lifecycle.current.mounted) return;
          try {
            operation = await request.get<Operation>(
              `${history(target)}/${encodeURIComponent(operation.operation_id)}`,
              { suppressGlobalError: true }
            );
          } catch (err) {
            const status = isAxiosError(err) ? err.response?.status : undefined;
            if (status !== undefined && status < 500 && status !== 408 && status !== 429) throw err;
            patch(target.key, { error: t(`${prefix}.reconnecting`) });
          }
        }
      }
      currentKey = undefined;
      const version = await request.get<ClusterVersion>('/v1/cluster/version', {
        suppressGlobalError: true,
      });
      if (lifecycle.current.mounted) {
        setClusterVersion(version);
        setUpdated(true);
      }
    } catch (err) {
      if (currentKey)
        patch(currentKey, { status: confirmedFailure ? 'failed' : 'pending', error: message(err) });
      if (lifecycle.current.mounted) setError(message(err));
    } finally {
      if (lifecycle.current.mounted) setBusy(false);
    }
  };

  if (!isAdmin || !globalReady) return null;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-[calc(100%-2rem)] sm:max-w-xl">
        <DialogHeader className="pr-6 text-left">
          <DialogTitle>{t(`${prefix}.title`)}</DialogTitle>
          <DialogDescription>{t(`${prefix}.description`)}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          {checking && (
            <p className="flex items-center gap-2 text-sm" role="status">
              <Loader2 className="size-4 animate-spin" />
              {t(`${prefix}.checking`)}
            </p>
          )}
          {targets.length > 0 && (
            <div className="divide-y overflow-hidden rounded-xl border">
              {targets.map((target) => (
                <div key={target.key} className="space-y-2 px-4 py-3 text-sm">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium">
                      {target.component === 'api'
                        ? 'API'
                        : target.component === 'supervisor'
                          ? 'Supervisor'
                          : 'Worker'}
                    </span>
                    <span
                      role="status"
                      className={
                        target.status === 'failed' ? 'text-destructive' : 'text-muted-foreground'
                      }
                    >
                      {t(`${prefix}.${target.status}`)}
                    </span>
                  </div>
                  {target.address && (
                    <p className="break-all font-mono text-xs text-muted-foreground">
                      {target.address}
                    </p>
                  )}
                  {target.update && (
                    <p className="text-muted-foreground">
                      {target.update.current_version}
                      {target.update.update_available ? ` → ${target.update.latest_version}` : ''}
                    </p>
                  )}
                  {target.error && <p className="break-words text-destructive">{target.error}</p>}
                  {target.log && (
                    <details>
                      <summary className="cursor-pointer text-muted-foreground">
                        {t('clusterInfo.relaunch.installLog')}
                      </summary>
                      <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-all text-xs">
                        {target.log}
                      </pre>
                    </details>
                  )}
                </div>
              ))}
            </div>
          )}
          {updated && (
            <p className="flex items-center gap-2 text-sm" role="status">
              <Check className="size-4 text-primary" />
              {t(`${prefix}.complete`)}
            </p>
          )}
          {error && (
            <p role="alert" className="break-words text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
        <DialogFooter className="border-t pt-4">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t(`${prefix}.close`)}
          </Button>
          {updated ? (
            <Button onClick={() => window.location.reload()}>
              <RefreshCw className="size-4" />
              {t(`${prefix}.refresh`)}
            </Button>
          ) : (
            <>
              <Button variant="outline" disabled={checking || busy} onClick={() => void check()}>
                {t(`${prefix}.check`)}
              </Button>
              <Button
                disabled={
                  checking ||
                  busy ||
                  targets.some((target) => target.status === 'failed') ||
                  !targets.some(
                    (target) => target.update?.update_available && target.status !== 'succeeded'
                  )
                }
                onClick={() => void upgrade()}
              >
                {busy && <Loader2 className="size-4 animate-spin" />}
                {t(`${prefix}.${busy ? 'upgrading' : 'upgrade'}`)}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
