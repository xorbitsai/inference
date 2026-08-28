'use client';

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Check, CloudDownload, Loader2, RotateCcw, Save, SlidersHorizontal } from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import PageContainer from '@/components/ui/page-container';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { cn } from '@/lib/utils';

type DownloadSource = 'auto' | 'huggingface' | 'modelscope' | 'openmind_hub' | 'csghub';
type NumericValue = number | '';
type NumericField = 'download_max_attempts' | 'hub_detect_timeout' | 'model_download_workers';

interface SystemSettingsData {
  download_source: DownloadSource;
  hf_endpoint: string;
  hf_token: string;
  pip_index_url: string;
  download_max_attempts: number;
  hub_detect_timeout: number;
  model_download_workers: number;
}

interface SystemSettingsForm extends Omit<
  SystemSettingsData,
  'download_max_attempts' | 'hub_detect_timeout' | 'model_download_workers'
> {
  download_max_attempts: NumericValue;
  hub_detect_timeout: NumericValue;
  model_download_workers: NumericValue;
}

interface FieldBlockProps {
  id: string;
  label: string;
  environmentVariable: string;
  children: ReactNode;
}

const EMPTY_FORM: SystemSettingsForm = {
  download_source: 'auto',
  hf_endpoint: '',
  hf_token: '',
  pip_index_url: '',
  download_max_attempts: '',
  hub_detect_timeout: '',
  model_download_workers: '',
};

function EnvironmentVariable({ name }: { name: string }) {
  return (
    <code className="w-fit rounded-md border bg-muted/50 px-2 py-1 text-[11px] font-medium text-muted-foreground">
      {name}
    </code>
  );
}

function FieldBlock({ id, label, environmentVariable, children }: FieldBlockProps) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Label htmlFor={id}>{label}</Label>
        <EnvironmentVariable name={environmentVariable} />
      </div>
      {children}
    </div>
  );
}

function normalizeSettings(data: SystemSettingsData): SystemSettingsForm {
  return {
    download_source: data.download_source,
    hf_endpoint: data.hf_endpoint ?? '',
    hf_token: data.hf_token ?? '',
    pip_index_url: data.pip_index_url ?? '',
    download_max_attempts: data.download_max_attempts,
    hub_detect_timeout: data.hub_detect_timeout,
    model_download_workers: data.model_download_workers,
  };
}

export default function SystemSettings() {
  const { t } = useI18n();
  const [form, setForm] = useState<SystemSettingsForm>(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);

  const loadSettings = useCallback(async () => {
    setLoading(true);
    try {
      const data = await request.get<SystemSettingsData>('/v1/cluster/system_settings');
      setForm(normalizeSettings(data));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const updateField = <K extends keyof SystemSettingsForm>(
    field: K,
    value: SystemSettingsForm[K]
  ) => {
    setForm((current) => ({ ...current, [field]: value }));
  };

  const updateNumericField = (field: NumericField, value: string) => {
    updateField(field, value === '' ? '' : Number(value));
  };

  const isValid =
    form.download_max_attempts !== '' &&
    Number.isFinite(form.download_max_attempts) &&
    Number.isInteger(form.download_max_attempts) &&
    form.download_max_attempts >= 1 &&
    form.hub_detect_timeout !== '' &&
    Number.isFinite(form.hub_detect_timeout) &&
    form.hub_detect_timeout > 0 &&
    form.model_download_workers !== '' &&
    Number.isFinite(form.model_download_workers) &&
    Number.isInteger(form.model_download_workers) &&
    form.model_download_workers >= 1;

  const saveSettings = async () => {
    if (!isValid) return;
    setSaving(true);
    try {
      const data = await request.put<SystemSettingsData>('/v1/cluster/system_settings', {
        ...form,
        download_max_attempts: Number(form.download_max_attempts),
        hub_detect_timeout: Number(form.hub_detect_timeout),
        model_download_workers: Number(form.model_download_workers),
      });
      setForm(normalizeSettings(data));
      toast.success(t('systemSettings.saveSuccess'));
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = async () => {
    setResetting(true);
    try {
      const data = await request.post<SystemSettingsData>('/v1/cluster/system_settings/reset');
      setForm(normalizeSettings(data));
      setResetOpen(false);
      toast.success(t('systemSettings.restoreSuccess'));
    } finally {
      setResetting(false);
    }
  };

  const sources: Array<{
    value: DownloadSource;
    abbreviation: string;
    label: string;
    tone: string;
  }> = [
    {
      value: 'auto',
      abbreviation: 'AUTO',
      label: t('systemSettings.autoSource'),
      tone: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
    },
    {
      value: 'huggingface',
      abbreviation: 'HF',
      label: t('systemSettings.huggingFace'),
      tone: 'bg-amber-500/10 text-amber-700 dark:text-amber-400',
    },
    {
      value: 'modelscope',
      abbreviation: 'MS',
      label: t('systemSettings.modelScope'),
      tone: 'bg-sky-500/10 text-sky-700 dark:text-sky-400',
    },
    {
      value: 'openmind_hub',
      abbreviation: 'OM',
      label: t('systemSettings.openMindHub'),
      tone: 'bg-violet-500/10 text-violet-700 dark:text-violet-400',
    },
    {
      value: 'csghub',
      abbreviation: 'CSG',
      label: t('systemSettings.csgHub'),
      tone: 'bg-rose-500/10 text-rose-700 dark:text-rose-400',
    },
  ];

  const downloadPolicies: Array<{
    id: string;
    field: NumericField;
    label: string;
    environmentVariable: string;
    min: number;
    step: number;
    unit: string;
  }> = [
    {
      id: 'download-max-attempts',
      field: 'download_max_attempts',
      label: t('systemSettings.maxAttempts'),
      environmentVariable: 'XINFERENCE_DOWNLOAD_MAX_ATTEMPTS',
      min: 1,
      step: 1,
      unit: t('systemSettings.attemptsUnit'),
    },
    {
      id: 'hub-detect-timeout',
      field: 'hub_detect_timeout',
      label: t('systemSettings.detectTimeout'),
      environmentVariable: 'XINFERENCE_HUB_DETECT_TIMEOUT',
      min: 0.1,
      step: 0.1,
      unit: t('systemSettings.secondsUnit'),
    },
    {
      id: 'model-download-workers',
      field: 'model_download_workers',
      label: t('systemSettings.downloadWorkers'),
      environmentVariable: 'XINFERENCE_MODEL_DOWNLOAD_WORKERS',
      min: 1,
      step: 1,
      unit: t('systemSettings.threadsUnit'),
    },
  ];

  return (
    <>
      <PageContainer
        title={t('menu.systemSettings')}
        subTitle={t('systemSettings.pageDescription')}
        extraContent={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => setResetOpen(true)}
              disabled={loading || saving || resetting}
            >
              <RotateCcw className="h-4 w-4" />
              {t('systemSettings.restoreSettings')}
            </Button>
            <Button onClick={saveSettings} disabled={loading || saving || resetting || !isValid}>
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              {t('systemSettings.saveChanges')}
            </Button>
          </div>
        }
      >
        {loading ? (
          <div className="flex min-h-72 items-center justify-center">
            <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="w-full space-y-6">
            <Card className="gap-0 overflow-hidden rounded-xl py-0 shadow-none">
              <CardHeader className="border-b bg-muted/20 py-5">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <CloudDownload className="h-5 w-5" />
                  </div>
                  <div className="space-y-1">
                    <CardTitle className="text-base">{t('systemSettings.modelDownload')}</CardTitle>
                    <CardDescription>
                      {t('systemSettings.modelDownloadDescription')}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="space-y-8 py-6">
                <div className="space-y-3">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <Label>{t('systemSettings.downloadSource')}</Label>
                    <EnvironmentVariable name="XINFERENCE_MODEL_SRC" />
                  </div>

                  <div
                    role="radiogroup"
                    aria-label={t('systemSettings.downloadSource')}
                    className="grid gap-3 md:grid-cols-3 xl:grid-cols-5"
                  >
                    {sources.map((source) => {
                      const selected = form.download_source === source.value;

                      return (
                        <button
                          key={source.value}
                          type="button"
                          role="radio"
                          aria-checked={selected}
                          onClick={() => updateField('download_source', source.value)}
                          className={cn(
                            'relative flex min-h-20 items-center gap-4 rounded-lg border p-4 text-left outline-none transition-all',
                            'hover:border-primary/40 hover:bg-muted/30 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
                            selected && 'border-primary bg-primary/[0.04] shadow-sm'
                          )}
                        >
                          <span
                            className={cn(
                              'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-xs font-bold',
                              source.tone
                            )}
                          >
                            {source.abbreviation}
                          </span>
                          <span className="min-w-0 pr-6">
                            <span className="block text-sm font-semibold">{source.label}</span>
                          </span>
                          <span
                            className={cn(
                              'absolute right-4 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded-full border',
                              selected
                                ? 'border-primary bg-primary text-primary-foreground'
                                : 'border-muted-foreground/30 bg-background'
                            )}
                          >
                            {selected && <Check className="h-3.5 w-3.5" />}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="grid gap-6 border-t pt-7 lg:grid-cols-2">
                  <FieldBlock
                    id="hf-mirror"
                    label={t('systemSettings.hfMirror')}
                    environmentVariable="HF_ENDPOINT"
                  >
                    <Input
                      id="hf-mirror"
                      type="url"
                      value={form.hf_endpoint}
                      onChange={(event) => updateField('hf_endpoint', event.target.value)}
                      placeholder={t('systemSettings.hfMirrorPlaceholder')}
                    />
                  </FieldBlock>

                  <FieldBlock
                    id="hf-token"
                    label={t('systemSettings.hfToken')}
                    environmentVariable="HUGGING_FACE_HUB_TOKEN"
                  >
                    <Input
                      id="hf-token"
                      type="password"
                      value={form.hf_token}
                      onFocus={(event) => {
                        if (form.hf_token.includes('*')) event.currentTarget.select();
                      }}
                      onChange={(event) => updateField('hf_token', event.target.value)}
                      placeholder={t('systemSettings.hfTokenPlaceholder')}
                      autoComplete="off"
                    />
                  </FieldBlock>

                  <div className="lg:col-span-2">
                    <FieldBlock
                      id="pip-mirror"
                      label={t('systemSettings.pipMirror')}
                      environmentVariable="PIP_INDEX_URL"
                    >
                      <Input
                        id="pip-mirror"
                        type="url"
                        value={form.pip_index_url}
                        onChange={(event) => updateField('pip_index_url', event.target.value)}
                        placeholder={t('systemSettings.pipMirrorPlaceholder')}
                      />
                    </FieldBlock>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="gap-0 overflow-hidden rounded-xl py-0 shadow-none">
              <CardHeader className="border-b bg-muted/20 py-5">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 text-violet-700 dark:text-violet-400">
                    <SlidersHorizontal className="h-5 w-5" />
                  </div>
                  <div className="space-y-1">
                    <CardTitle className="text-base">
                      {t('systemSettings.downloadPolicy')}
                    </CardTitle>
                    <CardDescription>
                      {t('systemSettings.downloadPolicyDescription')}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="divide-y py-1">
                {downloadPolicies.map((policy) => (
                  <div
                    key={policy.id}
                    className="grid gap-4 py-5 md:grid-cols-[minmax(0,1fr)_13rem] md:items-center"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <Label htmlFor={policy.id}>{policy.label}</Label>
                      <EnvironmentVariable name={policy.environmentVariable} />
                    </div>
                    <div className="relative">
                      <Input
                        id={policy.id}
                        type="number"
                        value={form[policy.field]}
                        onChange={(event) => updateNumericField(policy.field, event.target.value)}
                        min={policy.min}
                        step={policy.step}
                        className="h-10 pr-16"
                      />
                      <Badge
                        variant="secondary"
                        className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 border-0 font-normal text-muted-foreground"
                      >
                        {policy.unit}
                      </Badge>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        )}
      </PageContainer>

      <ConfirmDialog
        isOpen={resetOpen}
        onOpenChange={setResetOpen}
        onConfirm={resetSettings}
        title={t('systemSettings.restoreTitle')}
        description={t('systemSettings.restoreDescription')}
        confirmText={t('systemSettings.restoreConfirm')}
        confirmClassName="bg-destructive text-destructive-foreground hover:bg-destructive/90"
        isLoading={resetting}
      />
    </>
  );
}
