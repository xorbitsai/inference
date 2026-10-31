'use client';

import { useEffect, useId, useState } from 'react';
import { isAxiosError } from 'axios';
import {
  AlertCircle,
  ArrowUpRight,
  Check,
  ChevronDown,
  Loader2,
  SlidersHorizontal,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
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
import { Switch } from '@/components/ui/switch';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';

interface ReloadConfig {
  enabled: boolean;
  model_config: Record<string, number | boolean | Record<string, number>>;
  parameters: Record<string, string>;
  hot_parameters?: string[];
  engine?: string;
  engine_version?: string;
  python_executable?: string;
  can_upgrade_environment?: boolean;
  mixed_parameters?: string[];
  replicas?: { replica_uid: string; engine?: string; engine_version?: string }[];
}
interface ReloadStatus {
  operation_id?: string;
  status: string;
  stage?: string;
  error?: string;
  restored?: boolean;
  partial?: boolean;
  completed_replicas?: number;
  replica_count?: number;
  replicas?: { replica_uid: string; status: string; stage?: string; error?: string }[];
}

function errorMessage(error: unknown) {
  if (isAxiosError(error)) return error.response?.data?.detail || error.message;
  return error instanceof Error ? error.message : String(error);
}

const PARAMETER_LABELS: Record<string, string> = {
  max_num_seqs: 'concurrency',
  max_running_requests: 'concurrency',
  max_num_batched_tokens: 'batchTokens',
  max_model_len: 'contextLength',
  context_length: 'contextLength',
  gpu_memory_utilization: 'gpuMemory',
  mem_fraction_static: 'gpuMemory',
  max_prefill_tokens: 'prefillTokens',
  chunked_prefill_size: 'prefillChunkSize',
  enforce_eager: 'eagerExecution',
  disable_cuda_graph: 'disableCudaGraph',
  cuda_graph_max_bs_decode: 'decodeGraphBatch',
  cuda_graph_max_bs_prefill: 'prefillGraphBatch',
  schedule_conservativeness: 'schedulingConservativeness',
};

export function ReloadDialog({
  open,
  onOpenChange,
  modelUid,
  onReady,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelUid: string;
  onReady: () => void;
}) {
  const { t } = useI18n();
  const fieldId = useId();
  const [config, setConfig] = useState<ReloadConfig | null>(null);
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const [job, setJob] = useState<ReloadStatus | null>(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [python, setPython] = useState('');
  const url = `/v1/models/${encodeURIComponent(modelUid)}/reload`;
  const busy = submitting || job?.status === 'reloading';

  useEffect(() => {
    if (!open) return;
    let active = true;
    setConfig(null);
    setJob(null);
    setError('');
    setPython('');
    Promise.all([
      request.get<ReloadConfig>(`${url}/config`, { suppressGlobalError: true }),
      request.get<ReloadStatus>(url, { suppressGlobalError: true }),
    ])
      .then(([metadata, status]) => {
        if (!active) return;
        setConfig(metadata);
        setValues(
          Object.fromEntries(
            Object.entries(metadata.model_config).map(([key, value]) => [
              key,
              typeof value === 'boolean'
                ? value
                : String(typeof value === 'object' ? (value.image ?? '') : value),
            ])
          )
        );
        setJob(
          status.status === 'reloading' ||
            (status.status === 'error' && (status.replicas?.length || 0) > 1)
            ? status
            : null
        );
      })
      .catch((err) => {
        if (active) setError(errorMessage(err));
      });
    return () => {
      active = false;
    };
  }, [open, url]);

  useEffect(() => {
    if (!open || job?.status !== 'reloading') return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const status = await request.get<ReloadStatus>(url, { suppressGlobalError: true });
        if (!active) return;
        setJob(status);
        setError('');
        if (status.status === 'reloading') {
          timer = setTimeout(poll, 1000);
        } else if (status.status === 'ready') {
          onReady();
          onOpenChange(false);
        }
      } catch (err) {
        if (active) {
          setError(errorMessage(err));
          timer = setTimeout(poll, 3000);
        }
      }
    };
    timer = setTimeout(poll, 1000);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [open, job?.status, url, onReady, onOpenChange]);

  const engineVersions = [
    ...new Set(
      config?.replicas?.map((replica) =>
        [replica.engine, replica.engine_version].filter(Boolean).join(' ')
      ) || (config?.engine_version ? [`${config.engine} ${config.engine_version}`] : [])
    ),
  ].filter(Boolean);

  const parameters = Object.entries(config?.parameters || {}).filter(
    ([key]) => config?.enabled || config?.hot_parameters?.includes(key)
  );
  const patch = Object.fromEntries(
    Object.entries(values)
      .filter(([key, value]) => value !== '' && parameters.some(([name]) => name === key))
      .filter(([key, value]) => {
        const current = config?.model_config[key];
        return typeof current === 'object'
          ? Number(value) !== current.image
          : (typeof value === 'boolean' ? value : Number(value)) !== current;
      })
      .map(([key, value]) => [
        key,
        config?.parameters[key] === 'image_limits'
          ? { image: Number(value) }
          : typeof value === 'boolean'
            ? value
            : Number(value),
      ])
  );

  const submit = async () => {
    setSubmitting(true);
    setError('');
    try {
      setJob(
        await request.post<ReloadStatus>(
          url,
          { model_config: patch, ...(python.trim() ? { python_executable: python.trim() } : {}) },
          { suppressGlobalError: true }
        )
      );
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  const liveParameters = parameters.filter(([key]) => config?.hot_parameters?.includes(key));
  const engineParameters = parameters.filter(([key]) => !config?.hot_parameters?.includes(key));
  const changeCount = Object.keys(patch).length + Number(Boolean(python.trim()));

  const renderParameter = ([key, kind]: [string, string]) => {
    const label =
      kind === 'image_limits'
        ? t('runningModels.reload.imageLimit')
        : key === 'reasoning_content'
          ? t('launchModel.parsingReasoningContent')
          : PARAMETER_LABELS[key]
            ? t(`runningModels.reload.fields.${PARAMETER_LABELS[key]}`)
            : key;
    const id = `${fieldId}-${key}`;
    return (
      <div
        key={key}
        className={
          kind === 'bool'
            ? 'flex items-center justify-between gap-4 rounded-xl border bg-background px-4 py-3.5'
            : 'space-y-3 rounded-xl border bg-background p-4'
        }
      >
        <label htmlFor={id} className="block min-w-0 cursor-pointer space-y-1">
          <span className="block text-sm font-medium">{label}</span>
          <span className="block break-all font-mono text-[11px] text-muted-foreground">{key}</span>
        </label>
        {kind === 'bool' && config?.mixed_parameters?.includes(key) ? (
          <select
            id={id}
            aria-label={key}
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            disabled={busy}
            value={typeof values[key] === 'boolean' ? String(values[key]) : ''}
            onChange={(event) =>
              setValues((current) => ({ ...current, [key]: event.target.value === 'true' }))
            }
          >
            <option value="" disabled>
              {t('runningModels.reload.mixed')}
            </option>
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        ) : kind === 'bool' ? (
          <Switch
            id={id}
            aria-label={key}
            checked={values[key] === true}
            disabled={busy}
            onChange={(value) => setValues((prev) => ({ ...prev, [key]: value }))}
          />
        ) : (
          <Input
            id={id}
            aria-label={key}
            className="h-10 bg-muted/20 tabular-nums"
            type="number"
            min={kind === 'image_limits' ? 1 : undefined}
            step={kind === 'fraction' || kind === 'positive_number' ? 'any' : 1}
            value={String(values[key] ?? '')}
            placeholder={t(
              config?.mixed_parameters?.includes(key)
                ? 'runningModels.reload.mixed'
                : 'runningModels.reload.default'
            )}
            disabled={busy}
            onChange={(event) => setValues((prev) => ({ ...prev, [key]: event.target.value }))}
          />
        )}
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-[calc(100%-2rem)] gap-5 sm:max-w-2xl">
        <DialogHeader className="gap-3 pr-7 text-left">
          <div className="flex items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <SlidersHorizontal className="size-5" />
            </div>
            <div className="min-w-0 space-y-1">
              <DialogTitle className="text-lg">{t('runningModels.reload.title')}</DialogTitle>
              <p className="truncate font-mono text-xs text-muted-foreground" title={modelUid}>
                {modelUid}
              </p>
            </div>
          </div>
          <DialogDescription className="text-sm leading-6">
            {t('runningModels.reload.description')}
          </DialogDescription>
          {config && (
            <div className="flex flex-wrap items-center gap-2">
              {engineVersions.map((engine) => (
                <Badge key={engine} variant="secondary" className="font-normal">
                  {engine}
                </Badge>
              ))}
              {config.enabled && (
                <Badge variant="outline" className="gap-1.5 border-primary/20 text-primary">
                  <Check />
                  {t('runningModels.reload.weightsRetained')}
                </Badge>
              )}
            </div>
          )}
        </DialogHeader>
        <div className="space-y-5">
          {!config && !error && (
            <div
              role="status"
              className="flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground"
            >
              <Loader2 className="size-4 animate-spin" />
              {t('common.loading')}
            </div>
          )}
          {config?.enabled === false && (
            <p className="rounded-xl border bg-muted/40 px-4 py-3 text-sm leading-6 text-muted-foreground">
              {t('runningModels.reload.disabled')}
            </p>
          )}
          {engineParameters.length > 0 && (
            <section className="space-y-3" aria-label={t('runningModels.reload.engineSettings')}>
              <h3 className="text-sm font-semibold">{t('runningModels.reload.engineSettings')}</h3>
              <div className="grid gap-3 sm:grid-cols-2">
                {engineParameters.map(renderParameter)}
              </div>
            </section>
          )}
          {liveParameters.length > 0 && (
            <section className="space-y-3" aria-label={t('runningModels.reload.responseSettings')}>
              <h3 className="text-sm font-semibold">
                {t('runningModels.reload.responseSettings')}
              </h3>
              {liveParameters.map(renderParameter)}
              {config?.hot_parameters?.includes('reasoning_content') &&
                !python.trim() &&
                Object.keys(patch).every((key) => config.hot_parameters?.includes(key)) && (
                  <p className="text-xs leading-5 text-muted-foreground">
                    {t('runningModels.reload.reasoningHint')}
                  </p>
                )}
            </section>
          )}
          {config?.enabled && config.can_upgrade_environment !== false && (
            <details className="group rounded-xl border">
              <summary className="flex cursor-pointer list-none items-center gap-3 rounded-xl p-4 outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
                <ArrowUpRight className="size-4 shrink-0 text-muted-foreground" />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">
                    {t('runningModels.reload.environmentTitle')}
                  </span>
                  <span
                    className="mt-1 block truncate text-xs text-muted-foreground"
                    title={python.trim() || undefined}
                  >
                    {python.trim() || t('runningModels.reload.environmentSummary')}
                  </span>
                </span>
                <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
              </summary>
              <div className="space-y-3 border-t px-4 py-4">
                <label htmlFor={`${fieldId}-python`} className="block text-sm font-medium">
                  {t('runningModels.reload.python')}
                </label>
                <Input
                  id={`${fieldId}-python`}
                  aria-label={t('runningModels.reload.python')}
                  className="font-mono text-xs"
                  value={python}
                  disabled={busy}
                  onChange={(event) => setPython(event.target.value)}
                  placeholder={config.python_executable || '/opt/xinference/bin/python'}
                />
                <p className="text-xs leading-5 text-muted-foreground">
                  {t('runningModels.reload.environmentHint')}
                </p>
              </div>
            </details>
          )}
          {(job?.replicas?.length || 0) > 1 && (
            <section
              className="space-y-3 rounded-xl border p-4"
              aria-label={t('runningModels.reload.replicaProgress')}
            >
              <p className="text-sm font-medium" aria-live="polite">
                {t('runningModels.reload.replicaProgress', {
                  completed: job?.completed_replicas || 0,
                  total: job?.replica_count || job?.replicas?.length || 0,
                })}
              </p>
              {job?.replicas?.map((replica) => (
                <div
                  key={replica.replica_uid}
                  className="flex items-center justify-between gap-3 text-xs"
                >
                  <span className="min-w-0 truncate font-mono" title={replica.replica_uid}>
                    {replica.replica_uid}
                  </span>
                  <span
                    className={
                      replica.status === 'error' ? 'text-destructive' : 'text-muted-foreground'
                    }
                  >
                    {t(`runningModels.reload.${replica.stage || replica.status}`)}
                  </span>
                </div>
              ))}
            </section>
          )}
          {job?.status === 'reloading' && (
            <div
              role="status"
              className="flex items-center gap-2 rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-primary"
            >
              <Loader2 className="size-4 shrink-0 animate-spin" />
              {t(`runningModels.reload.${job.stage || 'queued'}`)}
            </div>
          )}
          {(error || job?.error) && (
            <div
              role="alert"
              className="flex items-start gap-3 rounded-xl border border-destructive/20 bg-destructive/5 p-4 text-sm text-destructive"
            >
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              <p className="min-w-0 break-words leading-6 [overflow-wrap:anywhere]">
                {error || job?.error}
                {job?.partial &&
                  ` ${t('runningModels.reload.partial', {
                    completed: job.completed_replicas || 0,
                    total: job.replica_count || 0,
                  })}`}
                {job?.restored && ` ${t('runningModels.reload.restored')}`}
              </p>
            </div>
          )}
        </div>
        <DialogFooter className="items-center border-t pt-4 sm:justify-between">
          <p className="text-xs text-muted-foreground" aria-live="polite">
            {t(changeCount ? 'runningModels.reload.changes' : 'runningModels.reload.noChanges', {
              count: changeCount,
            })}
          </p>
          <div className="flex items-center justify-end gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              {t('common.cancel')}
            </Button>
            <Button
              disabled={
                !parameters.length || busy || (!Object.keys(patch).length && !python.trim())
              }
              onClick={submit}
            >
              {busy && <Loader2 className="size-4 animate-spin" />}
              {t('runningModels.reload.apply')}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
