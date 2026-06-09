'use client';

import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { FormInstance, FormValues } from '@/types/form';
import { toOptionValue, transformFetchToForm, transformFormToFetch } from '../utils';

export interface LaunchConfigHistoryItem {
  data: FormValues;
  model_name: string;
  model_uid: string;
  updated_at: number;
}

interface ConfigCacheProps {
  form: FormInstance;
  modelName?: string;
  refreshKey?: number;
}

const LAUNCH_CONFIG_HISTORY_KEY = 'historyArr';

export const readLaunchConfigHistory = (): LaunchConfigHistoryItem[] => {
  if (typeof window === 'undefined') return [];

  try {
    const history = JSON.parse(window.localStorage.getItem(LAUNCH_CONFIG_HISTORY_KEY) || '[]');

    if (!Array.isArray(history)) return [];

    return history.filter(
      (item): item is LaunchConfigHistoryItem =>
        Boolean(
          item &&
            typeof item === 'object' &&
            typeof item.model_name === 'string' &&
            typeof item.updated_at === 'number' &&
            item.data &&
            typeof item.data === 'object' &&
            !Array.isArray(item.data)
        )
    );
  } catch {
    return [];
  }
};

const writeLaunchConfigHistory = (history: LaunchConfigHistoryItem[]) => {
  if (typeof window === 'undefined') return;

  window.localStorage.setItem(LAUNCH_CONFIG_HISTORY_KEY, JSON.stringify(history));
};

export const getModelConfigHistory = (history: LaunchConfigHistoryItem[], modelName?: string) => {
  if (!modelName) return [];

  return history
    .filter((item) => item.model_name === modelName)
    .sort((a, b) => b.updated_at - a.updated_at);
};

export const getLatestModelConfigHistory = (modelName?: string) => {
  return getModelConfigHistory(readLaunchConfigHistory(), modelName)[0];
};

export const saveLaunchConfigHistory = (values: FormValues) => {
  const modelName = toOptionValue(values.model_name);
  const modelUid = toOptionValue(values?.model_uid);

  if (!modelName) return;

  const history = readLaunchConfigHistory();
  const nextItem = {
    data: values,
    model_name: modelName,
    model_uid: modelUid,
    updated_at: Date.now(),
  };
  const matchedIndex = history.findIndex((item) => {
    if (item.model_name !== modelName) return false;

    return modelUid ? item.model_uid === modelUid : !item.model_uid;
  });
  const nextHistory =
    matchedIndex >= 0
      ? history.map((item, index) => (index === matchedIndex ? nextItem : item))
      : [nextItem, ...history];

  writeLaunchConfigHistory(nextHistory);
};

const normalizeForCompare = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(normalizeForCompare);
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
        .map(([key, itemValue]) => [key, normalizeForCompare(itemValue)])
    );
  }

  return value;
};

const isSameConfig = (left: FormValues, right: FormValues) => {
  return JSON.stringify(normalizeForCompare(left)) === JSON.stringify(normalizeForCompare(right));
};

export default function ConfigCache({ form, modelName, refreshKey }: ConfigCacheProps) {
  const { t } = useI18n();
  const [configCacheOpen, setConfigCacheOpen] = useState(false);
  const [clearConfigCacheOpen, setClearConfigCacheOpen] = useState(false);
  const [pendingDeleteConfig, setPendingDeleteConfig] = useState<LaunchConfigHistoryItem>();
  const [configHistory, setConfigHistory] = useState<LaunchConfigHistoryItem[]>([]);
  const [formUpdateKey, setFormUpdateKey] = useState(0);
  const modelConfigHistory = useMemo(
    () => getModelConfigHistory(configHistory, modelName),
    [configHistory, modelName]
  );
  const currentFetchValues = useMemo(
    () =>
      transformFormToFetch(form.getFieldsValue()),
    [form, formUpdateKey]
  );
  const hasModelConfigHistory = modelConfigHistory.length > 0;

  const refreshConfigHistory = () => {
    setConfigHistory(readLaunchConfigHistory());
  };

  const handleOpenConfigCache = () => {
    refreshConfigHistory();
    setConfigCacheOpen(true);
  };

  const handleUseConfigCache = (item: LaunchConfigHistoryItem) => {
    form.resetFields();
    form.setFieldsValue(transformFetchToForm(item.data));
    setConfigCacheOpen(false);
  };

  const deleteConfigCache = (item: LaunchConfigHistoryItem) => {
    const nextHistory = readLaunchConfigHistory().filter(
      (historyItem) =>
        historyItem.model_name !== item.model_name || historyItem.updated_at !== item.updated_at
    );

    writeLaunchConfigHistory(nextHistory);
    setConfigHistory(nextHistory);
  };

  const handleConfirmDeleteConfigCache = () => {
    if (!pendingDeleteConfig) return;

    deleteConfigCache(pendingDeleteConfig);
    setPendingDeleteConfig(undefined);
  };

  const handleNewConfigCache = () => {
    form.resetFields();
    setConfigCacheOpen(false);
  };

  const handleClearModelConfigCache = () => {
    const nextHistory = readLaunchConfigHistory().filter((item) => item.model_name !== modelName);

    writeLaunchConfigHistory(nextHistory);
    setConfigHistory(nextHistory);
    setClearConfigCacheOpen(false);
    setConfigCacheOpen(false);
    form.resetFields();
  };

  useEffect(() => {
    setConfigHistory(readLaunchConfigHistory());
  }, [modelName, refreshKey]);

  useEffect(() => {
    if (modelName) return;

    setConfigCacheOpen(false);
    setClearConfigCacheOpen(false);
    setPendingDeleteConfig(undefined);
  }, [modelName]);

  useEffect(() => {
    return form.subscribe(() => {
      setFormUpdateKey((key) => key + 1);
    });
  }, [form]);

  return (
    <>
      <div className="flex shrink-0 items-center">
        <div className="inline-flex h-8 overflow-hidden rounded-md border bg-background text-sm font-medium shadow-xs">
          <button
            type="button"
            className="flex h-full items-center gap-2 px-3 transition-colors hover:bg-accent hover:text-accent-foreground"
            onClick={handleOpenConfigCache}
          >
            {t('launchModel.configCache')}
            {hasModelConfigHistory && (
              <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-xs text-primary">
                {modelConfigHistory.length}
              </span>
            )}
          </button>
          {hasModelConfigHistory && (
            <button
              type="button"
              className="flex h-full w-8 items-center justify-center border-l text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              onClick={() => {
                setClearConfigCacheOpen(true);
              }}
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>
      <Dialog open={configCacheOpen} onOpenChange={setConfigCacheOpen}>
        <DialogContent className="!max-w-2xl gap-0 p-0" showCloseButton={false}>
          <DialogHeader className="border-b px-6 py-5">
            <DialogTitle>{t('launchModel.configCache')}</DialogTitle>
          </DialogHeader>
          <div className="max-h-[52vh] space-y-3 overflow-y-auto px-6 py-5">
            {modelConfigHistory.length ? (
              modelConfigHistory.map((item) => {
                const isActiveConfig = isSameConfig(currentFetchValues, item.data);

                return (
                  <div
                    key={`${item.model_name}-${item.updated_at}`}
                    className={cn(
                      'flex items-center justify-between gap-4 rounded-md border p-5',
                      isActiveConfig ? 'border-primary' : 'border-border'
                    )}
                  >
                    <div className="min-w-0 space-y-1">
                      <div className="font-semibold">
                        {item.model_uid || t('launchModel.defaultConfig')}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {t('launchModel.modelUid')}: {item.model_uid || '-'}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {t('launchModel.lastUpdated')}:{' '}
                        {new Date(item.updated_at).toLocaleString()}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-4">
                      <Button onClick={() => handleUseConfigCache(item)}>
                        {t('launchModel.loadCache')}
                      </Button>
                      <Button
                        variant="ghost"
                        className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                        onClick={() => setPendingDeleteConfig(item)}
                      >
                        {t('launchModel.deleteCache')}
                      </Button>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="rounded-md border border-dashed py-10 text-center text-sm text-muted-foreground">
                {t('launchModel.noConfigCache')}
              </div>
            )}
          </div>
          <DialogFooter className="border-t px-6 py-4 sm:justify-between">
            <Button variant="ghost" className="text-primary" onClick={handleNewConfigCache}>
              {t('launchModel.newCache')}
            </Button>
            <Button
              variant="outline"
              onClick={() => setConfigCacheOpen(false)}
            >
              {t('common.cancel')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <ConfirmDialog
        isOpen={clearConfigCacheOpen}
        onOpenChange={setClearConfigCacheOpen}
        description={t('launchModel.confirmDeleteConfigCache')}
        confirmText={t('common.confirm')}
        onConfirm={handleClearModelConfigCache}
        confirmClassName="bg-destructive  hover:bg-destructive/90"
      />
      <ConfirmDialog
        isOpen={Boolean(pendingDeleteConfig)}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDeleteConfig(undefined);
          }
        }}
        description={t('launchModel.confirmDeleteConfigCache')}
        confirmText={t('common.confirm')}
        onConfirm={handleConfirmDeleteConfigCache}
        confirmClassName="bg-destructive  hover:bg-destructive/90"
      />
    </>
  );
}
