'use client';

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import {
  Activity,
  Check,
  CloudDownload,
  Gauge,
  Info,
  Loader2,
  Network,
  RotateCcw,
  Save,
  SlidersHorizontal,
} from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import PageContainer from '@/components/ui/page-container';
import { Select } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';
import { cn } from '@/lib/utils';

type DownloadSource = 'auto' | 'huggingface' | 'modelscope' | 'openmind_hub' | 'csghub';
type OtelProtocol = 'http/protobuf' | 'grpc';
type SettingsTab = 'download' | 'observability';
type NumericValue = number | '';
type DownloadNumericField =
  | 'download_max_attempts'
  | 'hub_detect_timeout'
  | 'model_download_workers';
type ObservabilityNumericField =
  | 'otel_sampling_rate'
  | 'otel_batch_export_schedule_delay'
  | 'otel_max_queue_size'
  | 'otel_max_export_batch_size'
  | 'otel_metric_export_interval'
  | 'otel_batch_export_timeout'
  | 'otel_metric_export_timeout';

interface DownloadSettingsData {
  download_source: DownloadSource;
  hf_endpoint: string;
  hf_token: string;
  pip_index_url: string;
  download_max_attempts: number;
  hub_detect_timeout: number;
  model_download_workers: number;
}

interface OpenTelemetrySettingsData {
  enable_otel: boolean;
  otlp_base_endpoint: string;
  otlp_trace_endpoint: string;
  otlp_metric_endpoint: string;
  otlp_api_key: string;
  otel_exporter_otlp_protocol: OtelProtocol;
  otel_sampling_rate: number;
  otel_batch_export_schedule_delay: number;
  otel_max_queue_size: number;
  otel_max_export_batch_size: number;
  otel_metric_export_interval: number;
  otel_batch_export_timeout: number;
  otel_metric_export_timeout: number;
}

interface ObservabilitySettingsData {
  opentelemetry: OpenTelemetrySettingsData;
  runtime_enable_otel?: boolean;
}

type DownloadSettingsForm = Omit<DownloadSettingsData, DownloadNumericField> &
  Record<DownloadNumericField, NumericValue>;
type ObservabilitySettingsForm = Omit<OpenTelemetrySettingsData, ObservabilityNumericField> &
  Record<ObservabilityNumericField, NumericValue>;

interface FieldBlockProps {
  id: string;
  label: string;
  environmentVariable: string;
  children: ReactNode;
}

const EMPTY_DOWNLOAD_FORM: DownloadSettingsForm = {
  download_source: 'auto',
  hf_endpoint: '',
  hf_token: '',
  pip_index_url: '',
  download_max_attempts: '',
  hub_detect_timeout: '',
  model_download_workers: '',
};

const EMPTY_OBSERVABILITY_FORM: ObservabilitySettingsForm = {
  enable_otel: false,
  otlp_base_endpoint: 'http://localhost:4318',
  otlp_trace_endpoint: '',
  otlp_metric_endpoint: '',
  otlp_api_key: '',
  otel_exporter_otlp_protocol: 'http/protobuf',
  otel_sampling_rate: '',
  otel_batch_export_schedule_delay: '',
  otel_max_queue_size: '',
  otel_max_export_batch_size: '',
  otel_metric_export_interval: '',
  otel_batch_export_timeout: '',
  otel_metric_export_timeout: '',
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

function normalizeDownloadSettings(data: DownloadSettingsData): DownloadSettingsForm {
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

function normalizeObservabilitySettings(
  data: ObservabilitySettingsData
): ObservabilitySettingsForm {
  const opentelemetry = data.opentelemetry;
  return {
    enable_otel: opentelemetry.enable_otel,
    otlp_base_endpoint: opentelemetry.otlp_base_endpoint ?? 'http://localhost:4318',
    otlp_trace_endpoint: opentelemetry.otlp_trace_endpoint ?? '',
    otlp_metric_endpoint: opentelemetry.otlp_metric_endpoint ?? '',
    otlp_api_key: opentelemetry.otlp_api_key ?? '',
    otel_exporter_otlp_protocol: opentelemetry.otel_exporter_otlp_protocol,
    otel_sampling_rate: opentelemetry.otel_sampling_rate,
    otel_batch_export_schedule_delay: opentelemetry.otel_batch_export_schedule_delay,
    otel_max_queue_size: opentelemetry.otel_max_queue_size,
    otel_max_export_batch_size: opentelemetry.otel_max_export_batch_size,
    otel_metric_export_interval: opentelemetry.otel_metric_export_interval,
    otel_batch_export_timeout: opentelemetry.otel_batch_export_timeout,
    otel_metric_export_timeout: opentelemetry.otel_metric_export_timeout,
  };
}

function getRuntimeEnableOtel(data: ObservabilitySettingsData): boolean {
  return data.runtime_enable_otel ?? data.opentelemetry.enable_otel;
}

export default function SystemSettings() {
  const { t } = useI18n();
  const { clusterUIConfig } = useGlobal();
  const { canWriteSettings } = useMenuAuth();
  const [activeTab, setActiveTab] = useState<SettingsTab>('download');
  const [downloadForm, setDownloadForm] = useState<DownloadSettingsForm>(EMPTY_DOWNLOAD_FORM);
  const [observabilityForm, setObservabilityForm] =
    useState<ObservabilitySettingsForm>(EMPTY_OBSERVABILITY_FORM);
  const [runtimeEnableOtel, setRuntimeEnableOtel] = useState<boolean | null>(null);
  const [downloadLoading, setDownloadLoading] = useState(true);
  const [observabilityLoading, setObservabilityLoading] = useState(true);
  const [savingTab, setSavingTab] = useState<SettingsTab | null>(null);
  const [resettingTab, setResettingTab] = useState<SettingsTab | null>(null);
  const [resetTarget, setResetTarget] = useState<SettingsTab | null>(null);
  const canEdit = !clusterUIConfig?.auth_advanced || canWriteSettings;

  const loadDownloadSettings = useCallback(async () => {
    setDownloadLoading(true);
    try {
      const data = await request.get<DownloadSettingsData>('/v1/cluster/system_settings');
      setDownloadForm(normalizeDownloadSettings(data));
    } finally {
      setDownloadLoading(false);
    }
  }, []);

  const loadObservabilitySettings = useCallback(async () => {
    setObservabilityLoading(true);
    try {
      const data = await request.get<ObservabilitySettingsData>(
        '/v1/cluster/observability_settings'
      );
      setObservabilityForm(normalizeObservabilitySettings(data));
      setRuntimeEnableOtel(getRuntimeEnableOtel(data));
    } finally {
      setObservabilityLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDownloadSettings();
    void loadObservabilitySettings();
  }, [loadDownloadSettings, loadObservabilitySettings]);

  const updateDownloadField = <K extends keyof DownloadSettingsForm>(
    field: K,
    value: DownloadSettingsForm[K]
  ) => {
    setDownloadForm((current) => ({ ...current, [field]: value }));
  };

  const updateDownloadNumericField = (field: DownloadNumericField, value: string) => {
    updateDownloadField(field, value === '' ? '' : Number(value));
  };

  const updateObservabilityField = <K extends keyof ObservabilitySettingsForm>(
    field: K,
    value: ObservabilitySettingsForm[K]
  ) => {
    setObservabilityForm((current) => ({ ...current, [field]: value }));
  };

  const updateObservabilityNumericField = (field: ObservabilityNumericField, value: string) => {
    updateObservabilityField(field, value === '' ? '' : Number(value));
  };

  const isDownloadValid =
    downloadForm.download_max_attempts !== '' &&
    Number.isFinite(downloadForm.download_max_attempts) &&
    Number.isInteger(downloadForm.download_max_attempts) &&
    downloadForm.download_max_attempts >= 1 &&
    downloadForm.hub_detect_timeout !== '' &&
    Number.isFinite(downloadForm.hub_detect_timeout) &&
    downloadForm.hub_detect_timeout > 0 &&
    downloadForm.model_download_workers !== '' &&
    Number.isFinite(downloadForm.model_download_workers) &&
    Number.isInteger(downloadForm.model_download_workers) &&
    downloadForm.model_download_workers >= 1;

  const positiveIntegerOtelFields: ObservabilityNumericField[] = [
    'otel_batch_export_schedule_delay',
    'otel_max_queue_size',
    'otel_max_export_batch_size',
    'otel_metric_export_interval',
    'otel_batch_export_timeout',
    'otel_metric_export_timeout',
  ];
  const areOtelIntegersValid = positiveIntegerOtelFields.every((field) => {
    const value = observabilityForm[field];
    return value !== '' && Number.isFinite(value) && Number.isInteger(value) && value >= 1;
  });
  const isOtelValid =
    observabilityForm.otlp_base_endpoint.trim() !== '' &&
    observabilityForm.otel_sampling_rate !== '' &&
    Number.isFinite(observabilityForm.otel_sampling_rate) &&
    observabilityForm.otel_sampling_rate >= 0 &&
    observabilityForm.otel_sampling_rate <= 1 &&
    areOtelIntegersValid &&
    observabilityForm.otel_max_export_batch_size !== '' &&
    observabilityForm.otel_max_queue_size !== '' &&
    observabilityForm.otel_max_export_batch_size <= observabilityForm.otel_max_queue_size;
  const loading = downloadLoading || observabilityLoading;
  const activeIsValid = activeTab === 'download' ? isDownloadValid : isOtelValid;
  const enableOtelChanged =
    runtimeEnableOtel !== null && observabilityForm.enable_otel !== runtimeEnableOtel;

  const saveSettings = async () => {
    if (!canEdit || !activeIsValid) return;
    const target = activeTab;
    setSavingTab(target);
    try {
      if (target === 'download') {
        const data = await request.put<DownloadSettingsData>('/v1/cluster/system_settings', {
          ...downloadForm,
          download_max_attempts: Number(downloadForm.download_max_attempts),
          hub_detect_timeout: Number(downloadForm.hub_detect_timeout),
          model_download_workers: Number(downloadForm.model_download_workers),
        });
        setDownloadForm(normalizeDownloadSettings(data));
        toast.success(t('systemSettings.downloadSaveSuccess'));
      } else {
        const data = await request.put<ObservabilitySettingsData>(
          '/v1/cluster/observability_settings',
          {
            opentelemetry: {
              ...observabilityForm,
              otel_sampling_rate: Number(observabilityForm.otel_sampling_rate),
              otel_batch_export_schedule_delay: Number(
                observabilityForm.otel_batch_export_schedule_delay
              ),
              otel_max_queue_size: Number(observabilityForm.otel_max_queue_size),
              otel_max_export_batch_size: Number(observabilityForm.otel_max_export_batch_size),
              otel_metric_export_interval: Number(observabilityForm.otel_metric_export_interval),
              otel_batch_export_timeout: Number(observabilityForm.otel_batch_export_timeout),
              otel_metric_export_timeout: Number(observabilityForm.otel_metric_export_timeout),
            },
          }
        );
        setObservabilityForm(normalizeObservabilitySettings(data));
        setRuntimeEnableOtel(getRuntimeEnableOtel(data));
        toast.success(t('systemSettings.observabilitySaveSuccess'));
      }
    } finally {
      setSavingTab(null);
    }
  };

  const resetSettings = async () => {
    if (!canEdit || !resetTarget) return;
    const target = resetTarget;
    setResettingTab(target);
    try {
      if (target === 'download') {
        const data = await request.post<DownloadSettingsData>('/v1/cluster/system_settings/reset');
        setDownloadForm(normalizeDownloadSettings(data));
        toast.success(t('systemSettings.downloadRestoreSuccess'));
      } else {
        const data = await request.post<ObservabilitySettingsData>(
          '/v1/cluster/observability_settings/reset'
        );
        setObservabilityForm(normalizeObservabilitySettings(data));
        setRuntimeEnableOtel(getRuntimeEnableOtel(data));
        toast.success(t('systemSettings.observabilityRestoreSuccess'));
      }
      setResetTarget(null);
    } finally {
      setResettingTab(null);
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
    field: DownloadNumericField;
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

  const otelProtocols: Array<{
    value: OtelProtocol;
    label: string;
    description: string;
  }> = [
    {
      value: 'http/protobuf',
      label: 'HTTP / Protobuf',
      description: t('systemSettings.httpProtobufDescription'),
    },
    {
      value: 'grpc',
      label: 'gRPC',
      description: t('systemSettings.grpcDescription'),
    },
  ];

  const otelPolicies: Array<{
    id: string;
    field: ObservabilityNumericField;
    label: string;
    environmentVariable: string;
    min: number;
    max?: number;
    step: number;
    unit: string;
  }> = [
    {
      id: 'otel-sampling-rate',
      field: 'otel_sampling_rate',
      label: t('systemSettings.samplingRate'),
      environmentVariable: 'XINFERENCE_OTEL_SAMPLING_RATE',
      min: 0,
      max: 1,
      step: 0.01,
      unit: t('systemSettings.ratioUnit'),
    },
    {
      id: 'otel-schedule-delay',
      field: 'otel_batch_export_schedule_delay',
      label: t('systemSettings.exportScheduleDelay'),
      environmentVariable: 'XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY',
      min: 1,
      step: 1,
      unit: t('systemSettings.millisecondsUnit'),
    },
    {
      id: 'otel-max-queue-size',
      field: 'otel_max_queue_size',
      label: t('systemSettings.maxQueueSize'),
      environmentVariable: 'XINFERENCE_OTEL_MAX_QUEUE_SIZE',
      min: 1,
      step: 1,
      unit: t('systemSettings.spansUnit'),
    },
    {
      id: 'otel-max-export-batch-size',
      field: 'otel_max_export_batch_size',
      label: t('systemSettings.maxExportBatchSize'),
      environmentVariable: 'XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE',
      min: 1,
      step: 1,
      unit: t('systemSettings.spansUnit'),
    },
    {
      id: 'otel-metric-export-interval',
      field: 'otel_metric_export_interval',
      label: t('systemSettings.metricExportInterval'),
      environmentVariable: 'XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL',
      min: 1,
      step: 1,
      unit: t('systemSettings.millisecondsUnit'),
    },
    {
      id: 'otel-batch-export-timeout',
      field: 'otel_batch_export_timeout',
      label: t('systemSettings.traceExportTimeout'),
      environmentVariable: 'XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT',
      min: 1,
      step: 1,
      unit: t('systemSettings.millisecondsUnit'),
    },
    {
      id: 'otel-metric-export-timeout',
      field: 'otel_metric_export_timeout',
      label: t('systemSettings.metricExportTimeout'),
      environmentVariable: 'XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT',
      min: 1,
      step: 1,
      unit: t('systemSettings.millisecondsUnit'),
    },
  ];

  return (
    <>
      <PageContainer
        title={t('menu.systemSettings')}
        subTitle={t('systemSettings.pageDescription')}
        extraContent={
          canEdit ? (
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                onClick={() => setResetTarget(activeTab)}
                disabled={loading || savingTab !== null || resettingTab !== null}
              >
                <RotateCcw className="h-4 w-4" />
                {t('systemSettings.restoreSettings')}
              </Button>
              <Button
                onClick={saveSettings}
                disabled={loading || savingTab !== null || resettingTab !== null || !activeIsValid}
              >
                {savingTab === activeTab ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                {t('systemSettings.saveChanges')}
              </Button>
            </div>
          ) : (
            <Badge variant="secondary">{t('systemSettings.readOnly')}</Badge>
          )
        }
      >
        {loading ? (
          <div className="flex min-h-72 items-center justify-center">
            <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <Tabs
            value={activeTab}
            onValueChange={(value) => setActiveTab(value as SettingsTab)}
            className="w-full gap-6"
          >
            <TabsList
              aria-label={t('systemSettings.settingsCategory')}
              className="grid w-full max-w-sm grid-cols-2"
            >
              <TabsTrigger value="download">
                <CloudDownload className="h-4 w-4" />
                {t('systemSettings.downloadTab')}
              </TabsTrigger>
              <TabsTrigger value="observability">
                <Activity className="h-4 w-4" />
                {t('systemSettings.observabilityTab')}
              </TabsTrigger>
            </TabsList>

            <TabsContent value="download" className="space-y-6">
              <Card className="gap-0 overflow-hidden rounded-xl py-0 shadow-none">
                <CardHeader className="border-b bg-muted/20 py-5">
                  <div className="flex items-start gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <CloudDownload className="h-5 w-5" />
                    </div>
                    <div className="space-y-1">
                      <CardTitle className="text-base">
                        {t('systemSettings.modelDownload')}
                      </CardTitle>
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
                        const selected = downloadForm.download_source === source.value;

                        return (
                          <button
                            key={source.value}
                            type="button"
                            role="radio"
                            aria-checked={selected}
                            disabled={!canEdit}
                            onClick={() => updateDownloadField('download_source', source.value)}
                            className={cn(
                              'relative flex min-h-20 items-center gap-4 rounded-lg border p-4 text-left outline-none transition-all',
                              'hover:border-primary/40 hover:bg-muted/30 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
                              'disabled:cursor-not-allowed disabled:opacity-60',
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
                        value={downloadForm.hf_endpoint}
                        disabled={!canEdit}
                        onChange={(event) => updateDownloadField('hf_endpoint', event.target.value)}
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
                        value={downloadForm.hf_token}
                        disabled={!canEdit}
                        onFocus={(event) => {
                          if (downloadForm.hf_token.includes('*')) event.currentTarget.select();
                        }}
                        onChange={(event) => updateDownloadField('hf_token', event.target.value)}
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
                          value={downloadForm.pip_index_url}
                          disabled={!canEdit}
                          onChange={(event) =>
                            updateDownloadField('pip_index_url', event.target.value)
                          }
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
                          value={downloadForm[policy.field]}
                          disabled={!canEdit}
                          onChange={(event) =>
                            updateDownloadNumericField(policy.field, event.target.value)
                          }
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
            </TabsContent>

            <TabsContent value="observability" className="space-y-6">
              <Card className="gap-0 overflow-hidden rounded-xl py-0 shadow-none">
                <CardHeader className="border-b bg-muted/20 px-5 py-4 sm:px-6">
                  <div className="flex flex-col gap-4 xl:flex-row xl:items-center">
                    <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-5 gap-y-3">
                      <div className="flex min-w-0 items-center gap-3">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-sky-500/10 text-sky-700 dark:text-sky-400">
                          <Network className="h-5 w-5" />
                        </div>
                        <div className="min-w-0 space-y-1">
                          <CardTitle className="text-base">
                            {t('systemSettings.openTelemetry')}
                          </CardTitle>
                          <CardDescription>
                            {t('systemSettings.openTelemetryDescription')}
                          </CardDescription>
                        </div>
                      </div>

                      {enableOtelChanged && (
                        <div
                          role="status"
                          className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/[0.08] px-3 py-2 text-sm font-medium shadow-sm"
                        >
                          <Info className="h-4 w-4 text-amber-700 dark:text-amber-300" />
                          <span>{t('systemSettings.restartRequired')}</span>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between gap-4 border-t pt-4 xl:border-t-0 xl:pt-0">
                      <div className="flex flex-col items-start gap-2">
                        <Label htmlFor="enable-otel" className="block font-semibold">
                          {t('systemSettings.enableOtel')}
                        </Label>
                        <EnvironmentVariable name="XINFERENCE_ENABLE_OTEL" />
                      </div>
                      <Switch
                        id="enable-otel"
                        checked={observabilityForm.enable_otel}
                        disabled={!canEdit}
                        onChange={(checked) => updateObservabilityField('enable_otel', checked)}
                        aria-label={t('systemSettings.enableOtel')}
                        className="shadow-sm data-[state=unchecked]:bg-muted-foreground/25"
                      />
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="py-6">
                  <div className="grid gap-6 lg:grid-cols-2">
                    <FieldBlock
                      id="otlp-base-endpoint"
                      label={t('systemSettings.otlpBaseEndpoint')}
                      environmentVariable="XINFERENCE_OTLP_BASE_ENDPOINT"
                    >
                      <Input
                        id="otlp-base-endpoint"
                        type="url"
                        value={observabilityForm.otlp_base_endpoint}
                        disabled={!canEdit}
                        onChange={(event) =>
                          updateObservabilityField('otlp_base_endpoint', event.target.value)
                        }
                        placeholder="http://localhost:4318"
                      />
                    </FieldBlock>

                    <FieldBlock
                      id="otel-protocol"
                      label={t('systemSettings.otlpProtocol')}
                      environmentVariable="XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL"
                    >
                      <Select<OtelProtocol>
                        value={observabilityForm.otel_exporter_otlp_protocol}
                        options={otelProtocols}
                        disabled={!canEdit}
                        allowClear={false}
                        onChange={(value) => {
                          if (value) {
                            updateObservabilityField('otel_exporter_otlp_protocol', value);
                          }
                        }}
                      />
                    </FieldBlock>

                    <FieldBlock
                      id="otlp-trace-endpoint"
                      label={t('systemSettings.traceEndpoint')}
                      environmentVariable="XINFERENCE_OTLP_TRACE_ENDPOINT"
                    >
                      <Input
                        id="otlp-trace-endpoint"
                        type="url"
                        value={observabilityForm.otlp_trace_endpoint}
                        disabled={!canEdit}
                        onChange={(event) =>
                          updateObservabilityField('otlp_trace_endpoint', event.target.value)
                        }
                        placeholder={t('systemSettings.traceEndpointPlaceholder')}
                      />
                      <p className="text-xs text-muted-foreground">
                        {t('systemSettings.endpointOverrideDescription')}
                      </p>
                    </FieldBlock>

                    <FieldBlock
                      id="otlp-metric-endpoint"
                      label={t('systemSettings.metricEndpoint')}
                      environmentVariable="XINFERENCE_OTLP_METRIC_ENDPOINT"
                    >
                      <Input
                        id="otlp-metric-endpoint"
                        type="url"
                        value={observabilityForm.otlp_metric_endpoint}
                        disabled={!canEdit}
                        onChange={(event) =>
                          updateObservabilityField('otlp_metric_endpoint', event.target.value)
                        }
                        placeholder={t('systemSettings.metricEndpointPlaceholder')}
                      />
                      <p className="text-xs text-muted-foreground">
                        {t('systemSettings.endpointOverrideDescription')}
                      </p>
                    </FieldBlock>

                    <div className="lg:col-span-2">
                      <FieldBlock
                        id="otlp-api-key"
                        label={t('systemSettings.otlpApiKey')}
                        environmentVariable="XINFERENCE_OTLP_API_KEY"
                      >
                        <Input
                          id="otlp-api-key"
                          type="password"
                          value={observabilityForm.otlp_api_key}
                          disabled={!canEdit}
                          onFocus={(event) => {
                            if (observabilityForm.otlp_api_key.includes('*')) {
                              event.currentTarget.select();
                            }
                          }}
                          onChange={(event) =>
                            updateObservabilityField('otlp_api_key', event.target.value)
                          }
                          placeholder={t('systemSettings.otlpApiKeyPlaceholder')}
                          autoComplete="off"
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
                      <Gauge className="h-5 w-5" />
                    </div>
                    <div className="space-y-1">
                      <CardTitle className="text-base">
                        {t('systemSettings.telemetryPolicy')}
                      </CardTitle>
                      <CardDescription>
                        {t('systemSettings.telemetryPolicyDescription')}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="divide-y py-1">
                  {otelPolicies.map((policy) => (
                    <div
                      key={policy.id}
                      className="grid gap-4 py-5 md:grid-cols-[minmax(0,1fr)_13rem] md:items-center"
                    >
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <Label htmlFor={policy.id}>{policy.label}</Label>
                          <EnvironmentVariable name={policy.environmentVariable} />
                        </div>
                        {policy.field === 'otel_max_export_batch_size' && (
                          <p className="text-xs text-muted-foreground">
                            {t('systemSettings.batchSizeDescription')}
                          </p>
                        )}
                      </div>
                      <div className="relative">
                        <Input
                          id={policy.id}
                          type="number"
                          value={observabilityForm[policy.field]}
                          disabled={!canEdit}
                          onChange={(event) =>
                            updateObservabilityNumericField(policy.field, event.target.value)
                          }
                          min={policy.min}
                          max={
                            policy.field === 'otel_max_export_batch_size' &&
                            observabilityForm.otel_max_queue_size !== ''
                              ? observabilityForm.otel_max_queue_size
                              : policy.max
                          }
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
            </TabsContent>
          </Tabs>
        )}
      </PageContainer>

      <ConfirmDialog
        isOpen={resetTarget !== null}
        onOpenChange={(open) => {
          if (!open) setResetTarget(null);
        }}
        onConfirm={resetSettings}
        title={t(
          resetTarget === 'observability'
            ? 'systemSettings.observabilityRestoreTitle'
            : 'systemSettings.downloadRestoreTitle'
        )}
        description={t(
          resetTarget === 'observability'
            ? 'systemSettings.observabilityRestoreDescription'
            : 'systemSettings.downloadRestoreDescription'
        )}
        confirmText={t('systemSettings.restoreConfirm')}
        confirmClassName="bg-destructive text-destructive-foreground hover:bg-destructive/90"
        isLoading={resettingTab !== null}
      />
    </>
  );
}
