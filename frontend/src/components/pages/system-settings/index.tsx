'use client';

import { useState, type ReactNode } from 'react';
import { Check, CloudDownload, Info, Save, SlidersHorizontal } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import PageContainer from '@/components/ui/page-container';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

type DownloadSource = 'huggingface' | 'modelscope';

interface FieldBlockProps {
  id: string;
  label: string;
  environmentVariable: string;
  children: ReactNode;
}

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

export default function SystemSettings() {
  const { t } = useI18n();
  const [downloadSource, setDownloadSource] = useState<DownloadSource>('huggingface');

  const sources: Array<{
    value: DownloadSource;
    abbreviation: string;
    label: string;
    tone: string;
  }> = [
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
  ];

  const downloadPolicies = [
    {
      id: 'download-max-attempts',
      label: t('systemSettings.maxAttempts'),
      environmentVariable: 'XINFERENCE_DOWNLOAD_MAX_ATTEMPTS',
      defaultValue: 3,
      min: 1,
      step: 1,
      unit: t('systemSettings.attemptsUnit'),
    },
    {
      id: 'hub-detect-timeout',
      label: t('systemSettings.detectTimeout'),
      environmentVariable: 'XINFERENCE_HUB_DETECT_TIMEOUT',
      defaultValue: 3,
      min: 0.1,
      step: 0.1,
      unit: t('systemSettings.secondsUnit'),
    },
    {
      id: 'model-download-workers',
      label: t('systemSettings.downloadWorkers'),
      environmentVariable: 'XINFERENCE_MODEL_DOWNLOAD_WORKERS',
      defaultValue: 2,
      min: 1,
      step: 1,
      unit: t('systemSettings.threadsUnit'),
    },
  ];

  return (
    <PageContainer
      title={t('menu.systemSettings')}
      subTitle={t('systemSettings.pageDescription')}
      extraContent={
        <Button disabled title={t('systemSettings.saveUnavailable')}>
          <Save className="h-4 w-4" />
          {t('systemSettings.saveChanges')}
        </Button>
      }
    >
      <div className="w-full space-y-6">
        <div className="flex items-start gap-3 rounded-lg border border-primary/20 bg-primary/[0.04] px-4 py-3.5">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <div className="space-y-0.5">
            <p className="text-sm font-medium">{t('systemSettings.previewTitle')}</p>
            <p className="text-xs leading-5 text-muted-foreground">
              {t('systemSettings.previewDescription')}
            </p>
          </div>
        </div>

        <Card className="gap-0 overflow-hidden rounded-xl py-0 shadow-none">
          <CardHeader className="border-b bg-muted/20 py-5">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <CloudDownload className="h-5 w-5" />
              </div>
              <div className="space-y-1">
                <CardTitle className="text-base">{t('systemSettings.modelDownload')}</CardTitle>
                <CardDescription>{t('systemSettings.modelDownloadDescription')}</CardDescription>
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
                className="grid gap-3 md:grid-cols-2"
              >
                {sources.map((source) => {
                  const selected = downloadSource === source.value;

                  return (
                    <button
                      key={source.value}
                      type="button"
                      role="radio"
                      aria-checked={selected}
                      onClick={() => setDownloadSource(source.value)}
                      className={cn(
                        'relative flex min-h-20 items-center gap-4 rounded-lg border p-4 text-left outline-none transition-all',
                        'hover:border-primary/40 hover:bg-muted/30 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
                        selected && 'border-primary bg-primary/[0.04] shadow-sm'
                      )}
                    >
                      <span
                        className={cn(
                          'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-sm font-bold',
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
                <CardTitle className="text-base">{t('systemSettings.downloadPolicy')}</CardTitle>
                <CardDescription>{t('systemSettings.downloadPolicyDescription')}</CardDescription>
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
                    defaultValue={policy.defaultValue}
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
    </PageContainer>
  );
}
