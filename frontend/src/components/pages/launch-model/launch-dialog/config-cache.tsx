'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { toast } from 'sonner';
import { useGlobal } from '@/contexts/global-context';
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
import { Input } from '@/components/ui/input';
import type { FormInstance, FormValues } from '@/types/form';
import type { RequestModelType } from '../types';
import { transformFetchToForm, transformFormToFetch } from '../utils';
import {
  deleteLaunchHistory,
  getModelConfigHistory,
  readLaunchConfigHistory,
  refreshLaunchConfigHistory,
  removeCachedLaunchHistoryItem,
  writeLaunchConfigHistory,
} from './launch-history';
import type { LaunchConfigHistoryItem } from './launch-history';
import {
  buildLaunchTemplateData,
  getLaunchHistoryItemKey,
  getOtherModelLaunchHistory,
} from './launch-history-utils.mjs';

interface ConfigCacheProps {
  form: FormInstance;
  modelName?: string;
  modelType: RequestModelType;
  refreshKey?: number;
  onHistoryRefreshed?: (history: LaunchConfigHistoryItem[]) => void;
  onUserChange?: () => void;
}

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

export default function ConfigCache({
  form,
  modelName,
  modelType,
  refreshKey,
  onHistoryRefreshed,
  onUserChange,
}: ConfigCacheProps) {
  const { t } = useI18n();
  const { clusterAuth } = useGlobal();
  const authenticated = clusterAuth?.auth === true;
  const [configCacheOpen, setConfigCacheOpen] = useState(false);
  const [clearConfigCacheOpen, setClearConfigCacheOpen] = useState(false);
  const [pendingDeleteConfig, setPendingDeleteConfig] = useState<LaunchConfigHistoryItem>();
  const [pendingTemplateConfig, setPendingTemplateConfig] = useState<LaunchConfigHistoryItem>();
  const [configHistory, setConfigHistory] = useState<LaunchConfigHistoryItem[]>([]);
  const [historySearch, setHistorySearch] = useState('');
  const [historyLoading, setHistoryLoading] = useState(false);
  const [formUpdateKey, setFormUpdateKey] = useState(0);
  const modelConfigHistory = useMemo(
    () => getModelConfigHistory(configHistory, modelName),
    [configHistory, modelName]
  );
  const otherModelConfigHistory = useMemo(
    () => getOtherModelLaunchHistory(configHistory, modelName, modelType, historySearch),
    [configHistory, historySearch, modelName, modelType]
  ) as LaunchConfigHistoryItem[];
  const allOtherModelConfigHistory = useMemo(
    () => getOtherModelLaunchHistory(configHistory, modelName, modelType),
    [configHistory, modelName, modelType]
  ) as LaunchConfigHistoryItem[];
  const currentFetchValues = useMemo(() => {
    void formUpdateKey;
    return transformFormToFetch(form.getFieldsValue());
  }, [form, formUpdateKey]);
  const hasModelConfigHistory = modelConfigHistory.length > 0;
  const hasAnyConfigHistory = hasModelConfigHistory || allOtherModelConfigHistory.length > 0;

  const refreshConfigHistory = useCallback(
    async (showWarnings = true) => {
      if (!modelName || clusterAuth === null) return;

      setConfigHistory(readLaunchConfigHistory(authenticated));
      setHistoryLoading(true);
      try {
        const result = await refreshLaunchConfigHistory(modelName, authenticated);
        setConfigHistory(result.history);
        onHistoryRefreshed?.(result.history);
        if (showWarnings && result.usedLocalFallback) {
          toast.warning(t('launchModel.configHistoryLocalFallback'));
        }
        if (showWarnings && result.syncFailed) {
          toast.warning(t('launchModel.configHistorySyncFailed'));
        }
      } finally {
        setHistoryLoading(false);
      }
    },
    [authenticated, clusterAuth, modelName, onHistoryRefreshed, t]
  );

  const handleOpenConfigCache = () => {
    setHistorySearch('');
    setConfigCacheOpen(true);
    void refreshConfigHistory();
  };

  const handleUseConfigCache = (item: LaunchConfigHistoryItem) => {
    onUserChange?.();
    form.resetFields();
    form.setFieldsValue(transformFetchToForm(item.data));
    setConfigCacheOpen(false);
  };

  const handleConfirmUseTemplate = () => {
    const item = pendingTemplateConfig;
    setPendingTemplateConfig(undefined);
    if (!item || !modelName) return;

    const templateData = buildLaunchTemplateData(item.data, modelName, modelType);
    if (!templateData) return;

    onUserChange?.();
    form.resetFields();
    form.setFieldsValue(transformFetchToForm(templateData));
    setConfigCacheOpen(false);
  };

  const handleConfirmDeleteConfigCache = async () => {
    const item = pendingDeleteConfig;
    if (!item || item.autostart_enabled || !item.is_owner) return;

    setPendingDeleteConfig(undefined);
    if (item.source === 'local' && item.pending_sync) {
      setConfigHistory(removeCachedLaunchHistoryItem(item, authenticated));
      return;
    }

    try {
      await deleteLaunchHistory(item);
      setConfigHistory(removeCachedLaunchHistoryItem(item, authenticated));
      await refreshConfigHistory(false);
    } catch {
      toast.error(t('launchModel.deleteConfigHistoryFailed'));
    }
  };

  const handleNewConfigCache = () => {
    onUserChange?.();
    form.resetFields();
    setConfigCacheOpen(false);
  };

  const handleClearModelConfigCache = async () => {
    if (!modelName) return;
    setClearConfigCacheOpen(false);

    const cached = readLaunchConfigHistory(authenticated);
    const nextCached = cached.filter(
      (item) => !(item.model_name === modelName && item.is_owner && !item.autostart_enabled)
    );
    writeLaunchConfigHistory(nextCached, authenticated);
    setConfigHistory(nextCached);

    const serverRecords = modelConfigHistory.filter(
      (item) => item.source === 'server' && item.is_owner && !item.autostart_enabled
    );
    const results = await Promise.allSettled(serverRecords.map(deleteLaunchHistory));
    await refreshConfigHistory(false);

    if (results.some((result) => result.status === 'rejected')) {
      toast.warning(t('launchModel.partialConfigHistoryDeleteFailed'));
    }
    if (modelConfigHistory.some((item) => item.is_owner && item.autostart_enabled)) {
      toast.warning(t('launchModel.autostartConfigDeleteProtected'));
    }
  };

  useEffect(() => {
    if (!modelName || clusterAuth === null) return;

    setConfigHistory(readLaunchConfigHistory(authenticated));
    void refreshConfigHistory(false);
  }, [authenticated, clusterAuth, modelName, refreshConfigHistory, refreshKey]);

  useEffect(() => {
    if (modelName) return;

    setConfigCacheOpen(false);
    setClearConfigCacheOpen(false);
    setPendingDeleteConfig(undefined);
    setPendingTemplateConfig(undefined);
    setHistorySearch('');
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
              aria-label={t('launchModel.clearMyConfigCache')}
              title={t('launchModel.clearMyConfigCache')}
              className="flex h-full w-8 items-center justify-center border-l text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              onClick={() => setClearConfigCacheOpen(true)}
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>
      <Dialog open={configCacheOpen} onOpenChange={setConfigCacheOpen}>
        <DialogContent className="!max-w-3xl gap-0 p-0" showCloseButton={false}>
          <DialogHeader className="border-b px-6 py-5">
            <DialogTitle>{t('launchModel.configCache')}</DialogTitle>
          </DialogHeader>
          <div className="max-h-[60vh] space-y-6 overflow-y-auto px-6 py-5">
            {historyLoading && (
              <div className="text-sm text-muted-foreground">
                {t('launchModel.loadingConfigHistory')}
              </div>
            )}
            {!historyLoading && !hasAnyConfigHistory && (
              <div className="rounded-md border border-dashed py-10 text-center text-sm text-muted-foreground">
                {t('launchModel.noConfigCache')}
              </div>
            )}

            {hasAnyConfigHistory && (
              <>
                <section className="space-y-3">
                  <h3 className="text-sm font-semibold">
                    {t('launchModel.currentModelConfigHistory')}
                  </h3>
                  {modelConfigHistory.length ? (
                    modelConfigHistory.map((item) => {
                      const isActiveConfig = isSameConfig(currentFetchValues, item.data);
                      const canDelete = item.is_owner && !item.autostart_enabled;

                      return (
                        <div
                          key={getLaunchHistoryItemKey(item)}
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
                              {t('launchModel.configHistoryCreator')}: {item.created_by || '-'}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {t('launchModel.lastUpdated')}:{' '}
                              {new Date(item.updated_at).toLocaleString()}
                            </div>
                            {item.pending_sync && (
                              <div className="text-sm text-amber-600">
                                {t('launchModel.configHistoryPendingSync')}
                              </div>
                            )}
                            {item.autostart_enabled && (
                              <div className="text-sm text-amber-600">
                                {t('launchModel.autostartConfigDeleteProtected')}
                              </div>
                            )}
                            {!item.is_owner && (
                              <div className="text-sm text-muted-foreground">
                                {t('launchModel.onlyDeleteOwnConfig')}
                              </div>
                            )}
                          </div>
                          <div className="flex shrink-0 items-center gap-4">
                            <Button onClick={() => handleUseConfigCache(item)}>
                              {t('launchModel.loadCache')}
                            </Button>
                            {item.is_owner && (
                              <Button
                                variant="ghost"
                                disabled={!canDelete}
                                title={
                                  item.autostart_enabled
                                    ? t('launchModel.autostartConfigDeleteProtected')
                                    : undefined
                                }
                                className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                                onClick={() => setPendingDeleteConfig(item)}
                              >
                                {t('launchModel.deleteCache')}
                              </Button>
                            )}
                          </div>
                        </div>
                      );
                    })
                  ) : hasAnyConfigHistory ? (
                    <div className="rounded-md border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
                      {t('launchModel.noExactModelConfigHistory')}
                    </div>
                  ) : null}
                </section>

                <section className="space-y-3 border-t pt-5">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <h3 className="text-sm font-semibold">
                      {t('launchModel.myOtherModelConfigHistory')}
                    </h3>
                    <Input
                      value={historySearch}
                      onChange={(event) => setHistorySearch(event.target.value)}
                      placeholder={t('launchModel.searchOtherModelConfigHistory')}
                      className="sm:max-w-xs"
                    />
                  </div>
                  {otherModelConfigHistory.length ? (
                    otherModelConfigHistory.map((item) => (
                      <div
                        key={getLaunchHistoryItemKey(item)}
                        className="flex items-center justify-between gap-4 rounded-md border border-border p-5"
                      >
                        <div className="min-w-0 space-y-1">
                          <div className="truncate font-semibold">{item.model_name}</div>
                          <div className="text-sm text-muted-foreground">
                            {t('launchModel.sourceModel')}: {item.model_name}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {t('launchModel.modelUid')}: {item.model_uid || '-'}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {t('launchModel.lastUpdated')}:{' '}
                            {new Date(item.updated_at).toLocaleString()}
                          </div>
                          {item.pending_sync && (
                            <div className="text-sm text-amber-600">
                              {t('launchModel.configHistoryPendingSync')}
                            </div>
                          )}
                        </div>
                        <Button
                          className="shrink-0"
                          variant="outline"
                          onClick={() => setPendingTemplateConfig(item)}
                        >
                          {t('launchModel.useAsConfigTemplate')}
                        </Button>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-md border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
                      {allOtherModelConfigHistory.length
                        ? t('launchModel.noMatchingOtherModelConfigHistory')
                        : t('launchModel.noOtherModelConfigHistory')}
                    </div>
                  )}
                </section>
              </>
            )}
          </div>
          <DialogFooter className="border-t px-6 py-4 sm:justify-between">
            <Button variant="ghost" className="text-primary" onClick={handleNewConfigCache}>
              {t('launchModel.newCache')}
            </Button>
            <Button variant="outline" onClick={() => setConfigCacheOpen(false)}>
              {t('common.cancel')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <ConfirmDialog
        isOpen={clearConfigCacheOpen}
        onOpenChange={setClearConfigCacheOpen}
        description={t('launchModel.confirmClearMyConfigCache')}
        confirmText={t('common.confirm')}
        onConfirm={() => void handleClearModelConfigCache()}
        confirmClassName="bg-destructive hover:bg-destructive/90"
      />
      <ConfirmDialog
        isOpen={Boolean(pendingDeleteConfig)}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteConfig(undefined);
        }}
        description={t('launchModel.confirmDeleteConfigCache')}
        confirmText={t('common.confirm')}
        onConfirm={() => void handleConfirmDeleteConfigCache()}
        confirmClassName="bg-destructive hover:bg-destructive/90"
      />
      <ConfirmDialog
        isOpen={Boolean(pendingTemplateConfig)}
        onOpenChange={(open) => {
          if (!open) setPendingTemplateConfig(undefined);
        }}
        description={t('launchModel.confirmUseOtherModelConfigTemplate', {
          sourceModel: pendingTemplateConfig?.model_name || '-',
          currentModel: modelName || '-',
        })}
        confirmText={t('launchModel.useAsConfigTemplate')}
        onConfirm={handleConfirmUseTemplate}
      />
    </>
  );
}
