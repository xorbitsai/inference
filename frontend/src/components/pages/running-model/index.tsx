'use client';

import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react';
import {
  Loader2,
  MousePointerClick,
  Power,
  SearchX,
  ServerOff,
  Trash2,
  MessageCircleMore,
  Cpu,
  Server,
  Code,
  RefreshCw,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import PageContainer from '@/components/ui/page-container';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { InfoTooltip } from '@/components/ui/tooltip';
import { useI18n } from '@/contexts/i18n-context';
import type { RunningModelItem, ReplicaItem } from '@/types/services';
import request from '@/lib/request';
import { cn } from '@/lib/utils';
import {
  getTryApiAbility,
  TryApiDrawer,
} from '@/components/pages/running-model-detail/components/try-api-drawer';
import { transformRunningModelDetail } from '@/components/pages/running-model-detail/utils';

interface EmptyStateProps {
  icon: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  loading?: boolean;
}

const EmptyState = ({ icon, title, description, loading = false }: EmptyStateProps) => (
  <div className="flex min-h-72 flex-1 flex-col items-center justify-center px-6 py-10 text-center">
    {loading ? (
      <Loader2 className="size-7 animate-spin text-muted-foreground mb-2" />
    ) : (
      <div className="mb-4 flex size-14 items-center justify-center rounded-full bg-background text-muted-foreground shadow-sm ring-1 ring-border/70">
        {icon}
      </div>
    )}

    <p className="text-sm font-medium text-foreground">{title}</p>
    {description && <p className="mt-1 text-xs leading-5 text-muted-foreground">{description}</p>}
  </div>
);
interface ContentItemInfoProps {
  title: React.ReactNode;
  value: React.ReactNode;
}

interface AutostartSummary {
  models?: Array<{
    enabled?: boolean;
    model_uid?: string;
  }>;
}

const ContentItemInfo = ({ title, value }: ContentItemInfoProps) => {
  return (
    <div className="p-3 rounded-lg bg-muted/50">
      <div className="text-xs text-muted-foreground">{title}</div>
      <div className="font-medium mt-1.5">{value}</div>
    </div>
  );
};

const formatGpuMemory = (bytes: number): string => {
  const gib = bytes / 1024 ** 3;
  if (gib >= 1) {
    return `${gib.toFixed(2)} GiB`;
  }
  return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
};

const RunningModel = () => {
  const { t } = useI18n();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [models, setModels] = useState<RunningModelItem[]>([]);
  const [activeModel, setActiveModel] = useState<RunningModelItem | undefined>(undefined);
  const [replicaLogs, setReplicaLogs] = useState<Record<string, ReplicaItem[]>>({});
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteConfirmLoading, setDeleteConfirmLoading] = useState(false);
  const [deleteReplicaId, setDeleteReplicaId] = useState<string | undefined>(undefined);
  const [deleteReplicaLoading, setDeleteReplicaLoading] = useState(false);
  const [tryApiOpen, setTryApiOpen] = useState(false);
  const tryApiAbility = useMemo(
    () => getTryApiAbility(activeModel?.model_ability || []),
    [activeModel?.model_ability]
  );
  const [autostartModelIds, setAutostartModelIds] = useState<string[]>([]);
  const [autostartBusyIds, setAutostartBusyIds] = useState<string[]>([]);
  const visibleModels = useMemo(() => {
    const keyword = query.trim().toLowerCase();

    const filteredModels = models.filter((model) => {
      const matchesKeyword =
        !keyword ||
        model.model_name.toLowerCase().includes(keyword) ||
        model.id.toLowerCase().includes(keyword);
      return matchesKeyword;
    });

    return filteredModels;
  }, [models, query]);

  const fetchModels = useCallback(() => {
    setLoading(true);
    request
      .get('/v1/models')
      .then((res) => {
        const list = ((res?.data || []) as RunningModelItem[]).map(
          (item) => transformRunningModelDetail(item) as RunningModelItem
        );
        setModels(list);
        setActiveModel((prev) => {
          if (!prev) {
            return list?.[0];
          }
          const current = list.find((item) => item.id === prev.id);

          return current ?? list?.[0];
        });
      })
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, []);

  const fetchAutostartModels = useCallback(() => {
    request
      .get<AutostartSummary>('/v1/autostart/models/summary')
      .then((res) => {
        setAutostartModelIds(
          (res.models || [])
            .filter((item) => item.enabled !== false && item.model_uid)
            .map((item) => String(item.model_uid))
        );
      })
      .catch(() => {
        setAutostartModelIds([]);
      });
  }, []);

  const fetchReplicas = useCallback((modelUid: string) => {
    request.get(`/v1/models/${modelUid}/replicas`).then((res) => {
      setReplicaLogs((prev) => ({
        ...prev,
        [modelUid]: Array.isArray(res) ? res : [],
      }));
    });
  }, []);

  const handleRefresh = useCallback(() => {
    fetchModels();
    fetchAutostartModels();
    if (activeModel?.id) {
      fetchReplicas(activeModel.id);
    }
  }, [activeModel?.id, fetchAutostartModels, fetchModels, fetchReplicas]);

  const handleDeleteModel = () => {
    if (!activeModel) return;
    setDeleteConfirmLoading(true);
    request
      .delete(`/v1/models/${activeModel?.id}`)
      .then(() => {
        setDeleteConfirmOpen(false);
        setActiveModel(undefined);
        setReplicaLogs((prev) => {
          const next = { ...prev };
          delete next[activeModel.id];
          return next;
        });
        fetchModels();
      })
      .finally(() => {
        setDeleteConfirmLoading(false);
      });
  };
  const handleDeleteReplica = () => {
    if (!activeModel || !deleteReplicaId) return;
    setDeleteReplicaLoading(true);
    request
      .delete(`/v1/models/${activeModel.id}/replicas/${deleteReplicaId}`)
      .then((res) => {
        setDeleteReplicaId(undefined);
        if (res?.remaining_replicas > 0) {
          fetchReplicas(activeModel.id);
        }
        fetchModels();
      })
      .finally(() => {
        setDeleteReplicaLoading(false);
      });
  };

  const handleRemoveAutostart = (modelUid: string) => {
    setAutostartBusyIds((prev) => (prev.includes(modelUid) ? prev : [...prev, modelUid]));
    request
      .delete(`/v1/autostart/models/${encodeURIComponent(modelUid)}`)
      .then(() => {
        fetchAutostartModels();
        toast.success(t('runningModels.removeAutostartSuccess', { modelUid }));
      })
      .catch(() => {
        toast.error(t('runningModels.removeAutostartFailed'));
      })
      .finally(() => {
        setAutostartBusyIds((prev) => prev.filter((id) => id !== modelUid));
      });
  };

  const renderModelCardList = () => {
    if (loading) {
      return (
        <EmptyState
          loading
          icon={null}
          title={t('runningModels.loadingModels')}
          description={t('runningModels.fetchingModelList')}
        />
      );
    }
    if (!models.length) {
      return (
        <EmptyState
          icon={<ServerOff className="size-7" />}
          title={t('runningModels.noRunningModels')}
          description={t('runningModels.launchModelFirst')}
        />
      );
    }
    if (!visibleModels.length) {
      return (
        <EmptyState
          icon={<SearchX className="size-7" />}
          title={t('runningModels.noModelsFound')}
          description={t('runningModels.tryChangingSearchKeywords')}
        />
      );
    }
    return visibleModels.map((model) => (
      <div
        key={model.id}
        className={cn(
          'relative p-4 border border-border/50 rounded-xl flex items-start justify-between gap-1 cursor-pointer transition-colors hover:bg-accent',
          activeModel?.id === model.id && 'bg-primary/10 text-primary border-primary/50'
        )}
        onClick={() => setActiveModel(model)}
      >
        <div className="flex-1 min-w-0">
          <h4 className="mb-1 font-semibold truncate">{model.id}</h4>
          <div className="text-muted-foreground text-xs truncate">{model.model_name}</div>
        </div>
        <button className="shrink-0 rounded-full border border-primary/20 bg-primary/5 text-primary px-3 py-1 text-xs font-medium text-muted-foreground">
          {model.model_type}
        </button>
      </div>
    ));
  };
  const renderModelContent = () => {
    if (!activeModel) {
      return (
        <EmptyState
          icon={<MousePointerClick className="size-7" />}
          title={t('runningModels.selectModel')}
          description={t('runningModels.selectModelDescription')}
        />
      );
    }
    const isAutostart = autostartModelIds.includes(activeModel.id);
    const isAutostartBusy = autostartBusyIds.includes(activeModel.id);

    return (
      <>
        <div className="p-6 flex items-start justify-between border-b">
          <div className="flex-1 min-w-0">
            <h4 className="mb-1 font-semibold truncate">{activeModel?.id}</h4>
            <div className="text-muted-foreground text-xs truncate">{activeModel?.model_name}</div>
          </div>
          <div className="shrink-0 flex item-center gap-2">
            <InfoTooltip content={t('runningModels.tryApi')}>
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => setTryApiOpen(true)}
                className="h-8 text-muted-foreground"
              >
                <Code />
              </Button>
            </InfoTooltip>
            {isAutostart && (
              <Button
                type="button"
                variant="outline"
                size="icon"
                loading={isAutostartBusy}
                onClick={() => handleRemoveAutostart(activeModel.id)}
                className="h-8 text-primary"
                title={t('runningModels.autostartEnabled')}
              >
                {!isAutostartBusy && <Power />}
              </Button>
            )}
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => activeModel?.id && router.push(`/running-model/${activeModel.id}`)}
              className="h-8 text-muted-foreground"
            >
              <MessageCircleMore />
            </Button>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => setDeleteConfirmOpen(true)}
              className="h-8 text-muted-foreground hover:bg-destructive/10 hover:text-destructive hover:border-destructive/50"
            >
              <Trash2 />
            </Button>
          </div>
        </div>
        <div className="p-6 space-y-3 overflow-y-auto">
          <h3 className="font-medium">{t('runningModels.baseInfo')}</h3>
          <div className="grid grid-cols-2 gap-4">
            <ContentItemInfo title={t('runningModels.modelType')} value={activeModel.model_type} />
            <ContentItemInfo
              title={t('runningModels.modelEngine')}
              value={activeModel.model_engine || '-'}
            />
            <ContentItemInfo
              title={t('runningModels.modelFormat')}
              value={activeModel.model_format || '-'}
            />
            {'model_size_in_billions' in activeModel && (
              <ContentItemInfo
                title={t('runningModels.modelSize')}
                value={activeModel.model_size_in_billions ?? '-'}
              />
            )}

            <ContentItemInfo
              title={t('runningModels.quantization')}
              value={activeModel.quantization || '-'}
            />
          </div>
          <h3 className="font-medium">{t('runningModels.resources')}</h3>
          <div className="grid grid-cols-2 gap-4">
            <ContentItemInfo title={t('runningModels.workerAddress')} value={activeModel.address} />
            <ContentItemInfo
              title={t('runningModels.gpuIndexes')}
              value={
                Array.isArray(activeModel?.accelerators) ? (
                  <div className="flex gap-1 text-xs">
                    {activeModel.accelerators.map((item) => (
                      <span
                        key={item}
                        className="flex items-center gap-1 rounded-full border border-primary/20 bg-primary/5 py-0.5 px-1 text-primary"
                      >
                        <Cpu className="size-3" />
                        GPU {item}
                      </span>
                    ))}
                  </div>
                ) : (
                  '-'
                )
              }
            />
            <ContentItemInfo
              title={t('runningModels.gpuMemory')}
              value={
                activeModel?.gpu_memory && Object.keys(activeModel.gpu_memory).length > 0 ? (
                  <div className="flex flex-col gap-0.5 text-xs">
                    {Object.keys(activeModel.gpu_memory)
                      .sort()
                      .map((workerAddress, _, workers) => {
                        const perGpu = activeModel.gpu_memory![workerAddress];
                        const multiWorker = workers.length > 1;
                        return Object.keys(perGpu)
                          .map((k) => [Number(k), perGpu[k]] as [number, number])
                          .sort((a, b) => a[0] - b[0])
                          .map(([gpuIdx, bytes]) => (
                            <span key={`${workerAddress}-${gpuIdx}`}>
                              {multiWorker ? `${workerAddress} · ` : ''}
                              GPU {gpuIdx}: {formatGpuMemory(bytes)}
                            </span>
                          ));
                      })}
                  </div>
                ) : (
                  '-'
                )
              }
            />
          </div>
          <h3 className="font-medium">{t('runningModels.replicaDetail')}</h3>
          {(replicaLogs?.[activeModel.id] || []).map((replica) => (
            <div key={replica.replica_model_uid} className="p-3 rounded-lg border">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Server size={16} className="text-muted-foreground" />
                  <span className="font-mono">{replica.replica_model_uid}</span>
                  <div
                    className={cn(
                      'rounded-md border px-2 py-1 text-xs font-medium leading-none',
                      replica.status === 'READY'
                        ? 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-300'
                        : replica.status === 'ERROR'
                          ? 'border-red-200 bg-red-50 text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300'
                          : 'border-border bg-muted/40 text-muted-foreground'
                    )}
                  >
                    {replica.status}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 h-8 w-8 rounded-full text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                  onClick={() => setDeleteReplicaId(String(replica.replica_id))}
                >
                  <Trash2 />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {t('runningModels.workerAddress')}: {replica.worker_address}
              </p>
            </div>
          ))}
        </div>
      </>
    );
  };
  useEffect(() => {
    if (activeModel && !replicaLogs[activeModel.id]) {
      fetchReplicas(activeModel.id);
    }
  }, [activeModel, fetchReplicas, replicaLogs]);
  useEffect(() => {
    fetchModels();
    fetchAutostartModels();
  }, [fetchAutostartModels, fetchModels]);

  return (
    <PageContainer
      title={t('menu.runningModels')}
      extraContent={
        <Button
          variant="outline"
          size="icon"
          aria-label={t('runningModels.refresh')}
          loading={loading}
          onClick={handleRefresh}
        >
          {!loading && <RefreshCw className="size-4" />}
        </Button>
      }
    >
      <div className="flex gap-6 h-[calc(100vh-130px)]">
        <div className="w-80 rounded-xl bg-card text-card-foreground shadow-sm shrink-0 border border-border flex flex-col overflow-hidden overflow-y-auto">
          <div className="px-4 pt-6 pb-2 shrink-0">
            <Input
              placeholder={t('runningModels.searchPlaceholder')}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          <div className="flex flex-1 flex-col gap-2 p-4 overflow-y-auto">
            {renderModelCardList()}
          </div>
        </div>
        <div className="rounded-2xl bg-card text-card-foreground shadow-sm flex-1 border border-border flex flex-col min-w-0 overflow-hidden">
          {renderModelContent()}
        </div>
      </div>
      <ConfirmDialog
        isOpen={deleteConfirmOpen}
        onOpenChange={setDeleteConfirmOpen}
        description={t('runningModels.terminateConfirmBody', {
          name: activeModel?.model_name,
          replica: activeModel?.replica,
        })}
        confirmText={t('runningModels.terminateConfirmOk')}
        confirmClassName="bg-destructive  hover:bg-destructive/90"
        onConfirm={handleDeleteModel}
        isLoading={deleteConfirmLoading}
      />
      <ConfirmDialog
        isOpen={!!deleteReplicaId}
        onOpenChange={(open) => {
          if (!open) setDeleteReplicaId(undefined);
        }}
        description={t('runningModels.removeReplicaConfirm', {
          modelUid: activeModel?.id,
          replicaId: deleteReplicaId,
        })}
        confirmClassName="bg-destructive  hover:bg-destructive/90"
        onConfirm={handleDeleteReplica}
        isLoading={deleteReplicaLoading}
      />
      <TryApiDrawer
        open={tryApiOpen}
        onOpenChange={setTryApiOpen}
        modelUid={activeModel?.id}
        ability={tryApiAbility}
      />
    </PageContainer>
  );
};
export default RunningModel;
