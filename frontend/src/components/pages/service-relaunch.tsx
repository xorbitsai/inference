'use client';

import { useEffect, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';

type Service = 'api' | 'supervisor';

export interface ServiceRelaunchOperation {
  operation_id: string;
  request_id?: string;
  status: 'starting' | 'preparing' | 'switching' | 'draining' | 'succeeded' | 'failed';
  created_at: number;
  source_version?: string;
  target_version?: string;
  python_executable?: string;
  error?: string;
}

const SERVICES = {
  api: { start: '/v1/api/relaunch', history: '/v1/api-relaunches' },
  supervisor: { start: '/v1/supervisor/relaunch', history: '/v1/supervisor-relaunches' },
};

function finished(job: ServiceRelaunchOperation): boolean {
  return job.status === 'succeeded' || job.status === 'failed';
}

function errorMessage(error: unknown): string {
  if (isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    return typeof detail === 'string' ? detail : error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

function useServiceRelaunches(service: Service) {
  const [operations, setOperations] = useState<ServiceRelaunchOperation[]>([]);
  const [error, setError] = useState('');
  const [available, setAvailable] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      let again = false;
      try {
        const data = await request.get<ServiceRelaunchOperation[]>(SERVICES[service].history, {
          suppressGlobalError: true,
        });
        if (active) {
          setOperations(data);
          setError('');
          setRetrying(false);
          setAvailable(true);
          again = data.some((job) => !finished(job));
        }
      } catch (err) {
        if (active) {
          const status = isAxiosError(err) ? err.response?.status : undefined;
          again =
            status === undefined ||
            status === 408 ||
            status === 429 ||
            (status >= 500 && status !== 501);
          setError(errorMessage(err));
          setRetrying(again);
          setAvailable(false);
        }
      } finally {
        if (active && again) timer = setTimeout(poll, 2000);
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [service, revision]);
  return {
    operations,
    error,
    retrying,
    canStart: available && !operations.some((job) => !finished(job)),
    refresh: () => setRevision((value) => value + 1),
    addOperation: (job: ServiceRelaunchOperation) => {
      setOperations((current) => [
        job,
        ...current.filter((item) => item.operation_id !== job.operation_id),
      ]);
      setRevision((value) => value + 1);
    },
  };
}

function ServiceRelaunchControl({ service }: { service: Service }) {
  const { t } = useI18n();
  const state = useServiceRelaunches(service);
  const [open, setOpen] = useState(false);
  const [python, setPython] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState('');
  const launcherRef = useRef<HTMLButtonElement>(null);
  const requestId = useRef('');
  const prefix = 'clusterInfo.serviceRelaunch';
  useEffect(() => {
    if (!open) return;
    requestId.current =
      crypto.randomUUID?.() ?? `${service}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setPython('');
    setSubmitError('');
  }, [open, service]);
  useEffect(() => {
    if (open && state.operations.some((job) => job.request_id === requestId.current))
      setOpen(false);
  }, [open, state.operations]);

  const submit = async () => {
    if (submitting || !state.canStart) return;
    setSubmitting(true);
    setSubmitError('');
    try {
      const job = await request.post<ServiceRelaunchOperation>(
        SERVICES[service].start,
        {
          request_id: requestId.current,
          ...(python.trim() ? { python_executable: python.trim() } : {}),
        },
        { suppressGlobalError: true }
      );
      state.addOperation(job);
      setOpen(false);
    } catch (err) {
      setSubmitError(errorMessage(err));
      // A lost response can still mean the stable host accepted the request.
      state.refresh();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <h3>{t(`${prefix}.${service}.name`)}</h3>
        </CardTitle>
        <CardDescription>{t(`${prefix}.${service}.description`)}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Button
          ref={launcherRef}
          variant="outline"
          disabled={!state.canStart}
          onClick={() => setOpen(true)}
        >
          {t(`${prefix}.${service}.title`)}
        </Button>
        {state.error && (
          <div className="space-y-1 text-sm" role="status">
            <p>{t(`${prefix}.${state.retrying ? 'reconnecting' : 'unavailable'}`)}</p>
            <p className="break-words text-muted-foreground">{state.error}</p>
            {!state.retrying && (
              <Button variant="outline" size="sm" onClick={state.refresh}>
                {t(`${prefix}.retry`)}
              </Button>
            )}
          </div>
        )}
        {!!state.operations.length && (
          <div className="space-y-2" aria-label={t(`${prefix}.${service}.history`)}>
            <h4 className="text-sm font-medium">{t(`${prefix}.${service}.history`)}</h4>
            {[...state.operations]
              .sort((a, b) => b.created_at - a.created_at)
              .slice(0, 10)
              .map((job) => (
                <div key={job.operation_id} className="space-y-1 rounded-md border p-3 text-sm">
                  <Badge
                    variant={job.status === 'failed' ? 'destructive' : 'secondary'}
                    role="status"
                  >
                    {t(`${prefix}.${job.status}`)}
                  </Badge>
                  {job.status === 'draining' && <p>{t(`${prefix}.${service}.drainingHint`)}</p>}
                  <p className="text-muted-foreground">
                    {new Date(job.created_at * 1000).toLocaleString()}
                  </p>
                  {job.source_version && (
                    <p>
                      {job.source_version}
                      {job.target_version ? ` → ${job.target_version}` : ''}
                    </p>
                  )}
                  {job.python_executable && (
                    <p className="break-all text-muted-foreground">{job.python_executable}</p>
                  )}
                  {job.error && <p className="break-words text-destructive">{job.error}</p>}
                </div>
              ))}
          </div>
        )}
      </CardContent>
      <Dialog
        open={open}
        onOpenChange={(value) => {
          if (!submitting) setOpen(value);
        }}
      >
        <DialogContent
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            launcherRef.current?.focus();
          }}
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
            className="space-y-4"
          >
            <DialogHeader>
              <DialogTitle>{t(`${prefix}.${service}.title`)}</DialogTitle>
              <DialogDescription>{t(`${prefix}.${service}.description`)}</DialogDescription>
            </DialogHeader>
            <label className="block space-y-2 text-sm">
              <span>{t('clusterInfo.relaunch.python')}</span>
              <Input
                value={python}
                disabled={submitting}
                onChange={(event) => setPython(event.target.value)}
                placeholder="/opt/xinference/bin/python"
              />
            </label>
            <p className="text-sm text-muted-foreground">{t(`${prefix}.environmentHint`)}</p>
            {submitError && (
              <p role="alert" className="break-words text-sm text-destructive">
                {submitError}
              </p>
            )}
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                disabled={submitting}
                onClick={() => setOpen(false)}
              >
                {t('common.cancel')}
              </Button>
              <Button type="submit" disabled={submitting || !state.canStart}>
                {t(`${prefix}.${service}.title`)}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

export function ServiceRelaunchPanel() {
  const { globalReady } = useGlobal();
  const { isAdmin } = useMenuAuth();
  if (!globalReady || !isAdmin) return null;
  return (
    <div className="mt-4 grid gap-4 lg:grid-cols-2">
      <ServiceRelaunchControl service="api" />
      <ServiceRelaunchControl service="supervisor" />
    </div>
  );
}
