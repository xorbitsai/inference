'use client';

import { useEffect, useState } from 'react';
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
}
interface ReloadStatus {
  operation_id?: string;
  status: string;
  stage?: string;
  error?: string;
  restored?: boolean;
}

function errorMessage(error: unknown) {
  if (isAxiosError(error)) return error.response?.data?.detail || error.message;
  return error instanceof Error ? error.message : String(error);
}

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
        setJob(status.status === 'reloading' ? status : null);
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

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t('runningModels.reload.title')}</DialogTitle>
          <DialogDescription>{t('runningModels.reload.description')}</DialogDescription>
        </DialogHeader>
        {config?.engine_version && (
          <p className="text-sm text-muted-foreground">
            {config.engine} {config.engine_version}
          </p>
        )}
        {config?.enabled === false && (
          <p className="text-sm">{t('runningModels.reload.disabled')}</p>
        )}
        {config?.hot_parameters?.includes('reasoning_content') &&
          !python.trim() &&
          Object.keys(patch).every((key) => config.hot_parameters?.includes(key)) && (
            <p className="text-sm text-muted-foreground">
              {t('runningModels.reload.reasoningHint')}
            </p>
          )}
        {parameters.map(([key, kind]) => (
          <label key={key} className="flex items-center justify-between gap-4 text-sm">
            <span>
              {kind === 'image_limits'
                ? t('runningModels.reload.imageLimit')
                : key === 'reasoning_content'
                  ? t('launchModel.parsingReasoningContent')
                  : key}
            </span>
            {kind === 'bool' ? (
              <Switch
                aria-label={key}
                checked={values[key] === true}
                disabled={busy}
                onChange={(value) => setValues((prev) => ({ ...prev, [key]: value }))}
              />
            ) : (
              <Input
                aria-label={key}
                className="w-40"
                type="number"
                min={kind === 'image_limits' ? 1 : undefined}
                step={kind === 'fraction' || kind === 'positive_number' ? 'any' : 1}
                value={String(values[key] ?? '')}
                placeholder={t('runningModels.reload.default')}
                disabled={busy}
                onChange={(event) => setValues((prev) => ({ ...prev, [key]: event.target.value }))}
              />
            )}
          </label>
        ))}
        {config?.enabled && (
          <label className="space-y-2 text-sm">
            <span>{t('runningModels.reload.python')}</span>
            <Input
              aria-label={t('runningModels.reload.python')}
              value={python}
              disabled={busy}
              onChange={(event) => setPython(event.target.value)}
              placeholder={config.python_executable || '/opt/xinference/bin/python'}
            />
            <p className="text-muted-foreground">{t('runningModels.reload.environmentHint')}</p>
          </label>
        )}
        {job?.status === 'reloading' && (
          <p role="status" className="text-sm">
            {t(`runningModels.reload.${job.stage || 'queued'}`)}
          </p>
        )}
        {(error || job?.error) && (
          <p role="alert" className="text-sm text-destructive">
            {error || job?.error}
            {job?.restored && ` ${t('runningModels.reload.restored')}`}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button
            disabled={!parameters.length || busy || (!Object.keys(patch).length && !python.trim())}
            onClick={submit}
          >
            {t('runningModels.reload.apply')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
