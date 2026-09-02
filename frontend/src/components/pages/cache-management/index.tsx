'use client';

import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Ban,
  Box,
  ChevronRight,
  CircleAlert,
  Copy,
  Database,
  Download,
  Pause,
  Play,
  RefreshCw,
  Settings2,
  Trash2,
} from 'lucide-react';
import { toast } from 'sonner';

import DownloadProgressDetails from '@/components/pages/launch-model/launch-dialog/download-progress-details';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Collapsible, CollapsibleContent } from '@/components/ui/collapsible';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import PageContainer from '@/components/ui/page-container';
import { Progress } from '@/components/ui/progress';
import { SearchInput } from '@/components/ui/search-input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { InfoTooltip } from '@/components/ui/tooltip';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';
import { cn, copyToClipboard } from '@/lib/utils';
import type { ModelCachedItem, ModelDownloadItem, ModelEnvItem } from '@/types/services';

type TabValue = 'models' | 'environments';
const ACTIVE_CACHE_DOWNLOAD_STAGES = new Set(['pending', 'resuming', 'downloading', 'pausing']);
type PendingAction =
  | { kind: 'download'; item: ModelDownloadItem }
  | { kind: 'downloadDelete'; item: ModelDownloadItem }
  | { kind: 'cache'; item: ModelCachedItem }
  | { kind: 'environment'; item: ModelEnvItem };

interface ListResponse<T> {
  list?: T[];
}

interface DeleteResponse {
  result?: boolean;
}

function asList<T>(response: ListResponse<T>): T[] {
  return Array.isArray(response?.list) ? response.list : [];
}

function progressPercent(progress: number): number {
  const value = Number(progress);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value <= 1 ? value * 100 : value));
}

function modelCacheLabel(item: ModelDownloadItem | ModelCachedItem): string {
  const size = String(item.model_size_in_billions ?? '').trim();
  const quantization = String(item.quantization ?? '').trim();
  const details = [
    size && size !== '-' ? (/b$/i.test(size) ? size : `${size}B`) : '',
    item.model_format,
    quantization.toLowerCase() === 'none' ? '' : quantization,
  ]
    .map((value) => String(value ?? '').trim())
    .filter((value) => value && value !== '-');

  return details.length ? `${item.model_name}（${details.join(' · ')}）` : item.model_name;
}

function includesQuery(query: string, ...values: unknown[]): boolean {
  return values.some((value) =>
    String(value ?? '')
      .toLowerCase()
      .includes(query)
  );
}

function SummaryCard({ icon, label, value }: { icon: ReactNode; label: string; value: number }) {
  return (
    <Card className="gap-3 py-4 shadow-none">
      <CardContent className="flex items-center justify-between px-5">
        <div>
          <div className="text-sm text-muted-foreground">{label}</div>
          <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
        </div>
        <div className="rounded-lg bg-primary/10 p-2.5 text-primary">{icon}</div>
      </CardContent>
    </Card>
  );
}

function PathCell({ path, copyLabel }: { path?: string; copyLabel: string }) {
  if (!path) return <>-</>;

  return (
    <div className="flex w-full items-center gap-2">
      <InfoTooltip content={path} contentClassName="max-w-[calc(100vw-2rem)] break-all">
        <span className="min-w-0 flex-1 truncate font-mono text-xs">{path}</span>
      </InfoTooltip>
      <button
        type="button"
        aria-label={copyLabel}
        className="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
        onClick={() => copyToClipboard(path)}
      >
        <Copy className="size-4" />
      </button>
    </div>
  );
}

export default function CacheManagement() {
  const { t } = useI18n();
  const { clusterAuth, clusterUIConfig } = useGlobal();
  const auth = useMenuAuth();
  const unrestricted = clusterAuth?.auth === false || !clusterUIConfig?.auth_advanced;
  const canViewDownloads = unrestricted || auth.hasModelsRead;
  const canCancelDownloads = unrestricted || auth.canWriteModels;
  const canViewCache = unrestricted || auth.hasCacheList;
  const canDeleteCache = unrestricted || auth.canDeleteCache;
  const canViewEnvironments = unrestricted || auth.hasVirtualEnvList;
  const canDeleteEnvironments = unrestricted || auth.canDeleteVirtualEnv;
  const [downloads, setDownloads] = useState<ModelDownloadItem[]>([]);
  const [cachedModels, setCachedModels] = useState<ModelCachedItem[]>([]);
  const [environments, setEnvironments] = useState<ModelEnvItem[]>([]);
  const [activeTab, setActiveTab] = useState<TabValue>('models');
  const [query, setQuery] = useState('');
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [downloadActionUid, setDownloadActionUid] = useState<string>();
  const [expandedDownloadUids, setExpandedDownloadUids] = useState<Set<string>>(() => new Set());
  const [pendingAction, setPendingAction] = useState<PendingAction>();
  const [lastUpdated, setLastUpdated] = useState<number>();
  const downloadsInFlight = useRef(false);
  const previousDownloadUids = useRef<Set<string>>(new Set());

  const availableTabs = useMemo<TabValue[]>(() => {
    const tabs: TabValue[] = [];
    if (canViewDownloads || canViewCache) tabs.push('models');
    if (canViewEnvironments) tabs.push('environments');
    return tabs;
  }, [canViewCache, canViewDownloads, canViewEnvironments]);

  useEffect(() => {
    if (!availableTabs.includes(activeTab) && availableTabs[0]) {
      setActiveTab(availableTabs[0]);
    }
  }, [activeTab, availableTabs]);

  const toggleDownloadDetails = useCallback((modelUid: string) => {
    setExpandedDownloadUids((current) => {
      const next = new Set(current);

      if (next.has(modelUid)) next.delete(modelUid);
      else next.add(modelUid);

      return next;
    });
  }, []);

  const loadDownloads = useCallback(
    async (isCancelled?: () => boolean) => {
      if (!canViewDownloads || downloadsInFlight.current) return;
      downloadsInFlight.current = true;
      try {
        const response = await request.get<ListResponse<ModelDownloadItem>>('/v1/downloads');
        if (isCancelled?.()) return;
        setDownloads(asList(response));
        setLastUpdated(Date.now());
      } finally {
        downloadsInFlight.current = false;
      }
    },
    [canViewDownloads]
  );

  const loadCachedModels = useCallback(async () => {
    if (!canViewCache) return;
    const response = await request.get<ListResponse<ModelCachedItem>>('/v1/cache/models');
    setCachedModels(asList(response));
    setLastUpdated(Date.now());
  }, [canViewCache]);

  const loadEnvironments = useCallback(async () => {
    if (!canViewEnvironments) return;
    const response = await request.get<ListResponse<ModelEnvItem>>('/v1/virtualenvs');
    setEnvironments(asList(response));
    setLastUpdated(Date.now());
  }, [canViewEnvironments]);

  const loadAll = useCallback(async () => {
    setInitialLoading(true);
    try {
      await Promise.all([loadDownloads(), loadCachedModels(), loadEnvironments()]);
    } finally {
      setInitialLoading(false);
    }
  }, [loadCachedModels, loadDownloads, loadEnvironments]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    if (!canViewDownloads) return;
    let isCancelled = false;
    let timeoutId: number | undefined;

    const poll = async () => {
      if (isCancelled) return;
      try {
        await loadDownloads(() => isCancelled);
      } finally {
        if (!isCancelled) timeoutId = window.setTimeout(poll, 2000);
      }
    };

    timeoutId = window.setTimeout(poll, 2000);
    return () => {
      isCancelled = true;
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
    };
  }, [canViewDownloads, loadDownloads]);

  useEffect(() => {
    const currentDownloadUids = new Set(downloads.map((item) => item.model_uid));
    const downloadFinished = Array.from(previousDownloadUids.current).some(
      (modelUid) => !currentDownloadUids.has(modelUid)
    );
    previousDownloadUids.current = currentDownloadUids;
    if (downloadFinished && canViewCache) {
      void loadCachedModels();
    }
  }, [canViewCache, downloads, loadCachedModels]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await Promise.all([loadDownloads(), loadCachedModels(), loadEnvironments()]);
    } finally {
      setRefreshing(false);
    }
  };

  const handleDownloadTaskAction = async (item: ModelDownloadItem, action: 'pause' | 'resume') => {
    if (!item.cache_uid) return;
    setDownloadActionUid(item.cache_uid);
    try {
      await request.post(`/v1/downloads/${encodeURIComponent(item.cache_uid)}/${action}`);
      toast.success(
        t(
          action === 'pause'
            ? 'cacheManagement.pauseDownloadSuccess'
            : 'cacheManagement.resumeDownloadSuccess'
        )
      );
      await loadDownloads();
    } finally {
      setDownloadActionUid(undefined);
    }
  };

  const handleConfirmAction = async () => {
    if (!pendingAction) return;
    setActionLoading(true);
    try {
      if (pendingAction.kind === 'download') {
        const endpoint =
          pendingAction.item.kind === 'cache' && pendingAction.item.cache_uid
            ? `/v1/cache/models/${encodeURIComponent(pendingAction.item.cache_uid)}/cancel`
            : `/v1/models/${encodeURIComponent(pendingAction.item.model_uid)}/cancel`;
        await request.post(endpoint);
        toast.success(t('cacheManagement.cancelSuccess'));
        await loadDownloads();
      } else if (pendingAction.kind === 'downloadDelete') {
        if (!pendingAction.item.cache_uid) return;
        await request.delete(`/v1/downloads/${encodeURIComponent(pendingAction.item.cache_uid)}`);
        toast.success(t('cacheManagement.deleteDownloadSuccess'));
        await loadDownloads();
      } else if (pendingAction.kind === 'cache') {
        const params = new URLSearchParams({
          model_version: pendingAction.item.model_version,
          worker_ip: pendingAction.item.actor_ip_address,
        });
        const response = await request.delete<DeleteResponse>(
          `/v1/cache/models?${params.toString()}`
        );
        if (response?.result === false) {
          toast.error(t('cacheManagement.deleteCacheFailed'));
          return;
        }
        toast.success(t('common.deleteSuccess'));
        await loadCachedModels();
      } else {
        const params = new URLSearchParams({
          model_name: pendingAction.item.model_name,
          model_engine: pendingAction.item.model_engine,
          python_version: pendingAction.item.python_version,
          worker_ip: pendingAction.item.actor_ip_address,
        });
        await request.delete(`/v1/virtualenvs?${params.toString()}`);
        toast.success(t('common.deleteSuccess'));
        await loadEnvironments();
      }
      setPendingAction(undefined);
    } finally {
      setActionLoading(false);
    }
  };

  const normalizedQuery = query.trim().toLowerCase();
  const filteredDownloads = downloads.filter((item) =>
    includesQuery(
      normalizedQuery,
      item.model_name,
      item.model_uid,
      item.model_version,
      item.model_engine,
      item.model_format,
      item.model_size_in_billions,
      item.quantization,
      ...item.replicas.map((replica) => replica.worker_address)
    )
  );
  const filteredCachedModels = cachedModels.filter((item) =>
    includesQuery(
      normalizedQuery,
      item.model_name,
      item.model_version,
      item.model_format,
      item.model_size_in_billions,
      item.quantization,
      item.path,
      item.real_path,
      item.actor_ip_address
    )
  );
  const filteredEnvironments = environments.filter((item) =>
    includesQuery(
      normalizedQuery,
      item.model_name,
      item.model_engine,
      item.python_version,
      item.path,
      item.real_path,
      item.actor_ip_address
    )
  );

  const confirmDescription = pendingAction
    ? pendingAction.kind === 'download'
      ? t(
          pendingAction.item.kind === 'cache'
            ? 'cacheManagement.cancelCacheDownloadConfirm'
            : 'cacheManagement.cancelDownloadConfirm',
          { model: pendingAction.item.model_uid }
        )
      : pendingAction.kind === 'downloadDelete'
        ? t('cacheManagement.deleteDownloadConfirm', {
            model: modelCacheLabel(pendingAction.item),
          })
        : pendingAction.kind === 'cache'
          ? t('cacheManagement.deleteCacheConfirm', {
              model: modelCacheLabel(pendingAction.item),
            })
          : t('cacheManagement.deleteEnvironmentConfirm', {
              model: pendingAction.item.model_name,
              worker: pendingAction.item.actor_ip_address,
            })
    : '';

  return (
    <PageContainer
      title={t('cacheManagement.title')}
      subTitle={t('cacheManagement.description')}
      loading={initialLoading}
      extraContent={
        <Button variant="outline" loading={refreshing} onClick={() => void handleRefresh()}>
          <RefreshCw className="size-4" />
          {t('common.refresh')}
        </Button>
      }
    >
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          {canViewDownloads && (
            <SummaryCard
              icon={<Download className="size-5" />}
              label={t('cacheManagement.incompleteDownloads')}
              value={downloads.length}
            />
          )}
          {canViewCache && (
            <SummaryCard
              icon={<Database className="size-5" />}
              label={t('cacheManagement.cachedModels')}
              value={cachedModels.length}
            />
          )}
          {canViewEnvironments && (
            <SummaryCard
              icon={<Settings2 className="size-5" />}
              label={t('cacheManagement.virtualEnvironments')}
              value={environments.length}
            />
          )}
        </div>

        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as TabValue)}>
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <TabsList>
              {(canViewDownloads || canViewCache) && (
                <TabsTrigger value="models">
                  <Box />
                  {t('cacheManagement.modelCache')}
                  <Badge variant="secondary">
                    {(canViewDownloads ? downloads.length : 0) +
                      (canViewCache ? cachedModels.length : 0)}
                  </Badge>
                </TabsTrigger>
              )}
              {canViewEnvironments && (
                <TabsTrigger value="environments">
                  <Settings2 />
                  {t('cacheManagement.virtualEnvironments')}
                  <Badge variant="secondary">{environments.length}</Badge>
                </TabsTrigger>
              )}
            </TabsList>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              {lastUpdated && (
                <span className="whitespace-nowrap text-xs text-muted-foreground">
                  {t('common.lastUpdateTime')}: {new Date(lastUpdated).toLocaleTimeString()}
                </span>
              )}
              <SearchInput
                value={query}
                onChange={setQuery}
                containerClassName="w-full sm:w-72"
                placeholder={t('cacheManagement.searchPlaceholder')}
              />
            </div>
          </div>

          {(canViewDownloads || canViewCache) && (
            <TabsContent value="models" className="mt-4 rounded-lg border">
              <Table className="table-fixed">
                <colgroup>
                  <col className="w-[12%]" />
                  <col className="w-[15%]" />
                  <col className="w-[6%]" />
                  <col className="w-[5%]" />
                  <col className="w-[4%]" />
                  <col className="w-[15%]" />
                  <col className="w-[12%]" />
                  <col className="w-[12%]" />
                  <col className="w-[8%]" />
                  <col className="w-[11%]" />
                </colgroup>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t('cacheManagement.modelName')}</TableHead>
                    <TableHead>{t('cacheManagement.modelVersion')}</TableHead>
                    <TableHead>{t('cacheManagement.format')}</TableHead>
                    <TableHead>{t('cacheManagement.size')}</TableHead>
                    <TableHead>{t('cacheManagement.quantization')}</TableHead>
                    <TableHead>{t('cacheManagement.statusAndProgress')}</TableHead>
                    <TableHead>{t('cacheManagement.path')}</TableHead>
                    <TableHead>{t('cacheManagement.realPath')}</TableHead>
                    <TableHead>{t('cacheManagement.worker')}</TableHead>
                    <TableHead className="text-center">{t('common.operation')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredDownloads.map((download) => {
                    const percent = progressPercent(download.progress);
                    const isCacheDownload =
                      download.kind === 'cache' && Boolean(download.cache_uid);
                    const isActiveCacheDownload =
                      isCacheDownload && ACTIVE_CACHE_DOWNLOAD_STAGES.has(download.stage);
                    const stageLabel =
                      {
                        pending: t('cacheManagement.downloadPending'),
                        resuming: t('cacheManagement.downloadResuming'),
                        downloading: t('cacheManagement.downloads'),
                        pausing: t('cacheManagement.downloadPausing'),
                        paused: t('cacheManagement.downloadPaused'),
                        interrupted: t('cacheManagement.downloadInterrupted'),
                        failed: t('cacheManagement.downloadFailed'),
                      }[download.stage] ?? download.stage;
                    const workers = Array.from(
                      new Set(
                        download.replicas
                          .map((replica) => replica.worker_address)
                          .filter((worker): worker is string => Boolean(worker))
                      )
                    );
                    const showDetails =
                      download.download_files.length > 0 ||
                      download.stage === 'interrupted' ||
                      (download.stage === 'failed' && Boolean(download.error));
                    const isDetailsExpanded =
                      showDetails && expandedDownloadUids.has(download.model_uid);

                    return (
                      <Fragment key={`download:${download.model_uid}`}>
                        <TableRow
                          className={
                            showDetails ? 'cursor-pointer bg-primary/[0.02]' : 'bg-primary/[0.02]'
                          }
                          aria-expanded={showDetails ? isDetailsExpanded : undefined}
                          onClick={
                            showDetails
                              ? (event) => {
                                  if (
                                    (event.target as HTMLElement).closest(
                                      'button, a, input, select, textarea'
                                    )
                                  ) {
                                    return;
                                  }
                                  toggleDownloadDetails(download.model_uid);
                                }
                              : undefined
                          }
                        >
                          <TableCell className="break-words">
                            <div className="flex min-w-0 items-center gap-1">
                              <div className="min-w-0 break-words font-medium">
                                {download.model_name}
                              </div>
                              {showDetails && (
                                <InfoTooltip
                                  content={t(isDetailsExpanded ? 'common.packUp' : 'common.unfold')}
                                >
                                  <button
                                    type="button"
                                    aria-label={t(
                                      isDetailsExpanded ? 'common.packUp' : 'common.unfold'
                                    )}
                                    aria-expanded={isDetailsExpanded}
                                    className="shrink-0 rounded-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      toggleDownloadDetails(download.model_uid);
                                    }}
                                  >
                                    <ChevronRight
                                      className={cn(
                                        'size-4 transition-transform duration-300 ease-out motion-reduce:transition-none',
                                        isDetailsExpanded && 'rotate-90'
                                      )}
                                    />
                                  </button>
                                </InfoTooltip>
                              )}
                            </div>
                          </TableCell>
                          <TableCell className="max-w-64 truncate font-mono text-xs">
                            {download.model_version || '-'}
                          </TableCell>
                          <TableCell>{download.model_format || '-'}</TableCell>
                          <TableCell className="font-semibold tabular-nums">
                            {download.model_size_in_billions ?? '-'}
                          </TableCell>
                          <TableCell>{download.quantization || '-'}</TableCell>
                          <TableCell>
                            <div className="w-full space-y-2">
                              <Badge
                                variant={
                                  ['interrupted', 'failed'].includes(download.stage)
                                    ? 'destructive'
                                    : 'secondary'
                                }
                              >
                                {stageLabel}
                              </Badge>
                              <div className="flex items-center gap-2">
                                <Progress value={percent} className="h-1.5 flex-1" />
                                <span className="w-9 text-right text-xs tabular-nums text-muted-foreground">
                                  {Math.round(percent)}%
                                </span>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>-</TableCell>
                          <TableCell>-</TableCell>
                          <TableCell>
                            <div className="space-y-1 break-all font-mono text-xs">
                              {workers.length
                                ? workers.map((worker) => <div key={worker}>{worker}</div>)
                                : '-'}
                            </div>
                          </TableCell>
                          <TableCell className="text-center">
                            <div className="flex flex-wrap items-center justify-center gap-1">
                              {canCancelDownloads && isActiveCacheDownload && (
                                <InfoTooltip content={t('cacheManagement.pauseDownload')}>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    aria-label={t('cacheManagement.pauseDownload')}
                                    loading={downloadActionUid === download.cache_uid}
                                    onClick={() => void handleDownloadTaskAction(download, 'pause')}
                                  >
                                    {downloadActionUid !== download.cache_uid && <Pause />}
                                  </Button>
                                </InfoTooltip>
                              )}
                              {canCancelDownloads && isCacheDownload && download.resumable && (
                                <InfoTooltip content={t('cacheManagement.resumeDownload')}>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    aria-label={t('cacheManagement.resumeDownload')}
                                    loading={downloadActionUid === download.cache_uid}
                                    onClick={() =>
                                      void handleDownloadTaskAction(download, 'resume')
                                    }
                                  >
                                    {downloadActionUid !== download.cache_uid && <Play />}
                                  </Button>
                                </InfoTooltip>
                              )}
                              {canCancelDownloads && (
                                <InfoTooltip content={t('cacheManagement.cancelDownload')}>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    aria-label={t('cacheManagement.cancelDownload')}
                                    onClick={() =>
                                      setPendingAction({ kind: 'download', item: download })
                                    }
                                  >
                                    <Ban />
                                  </Button>
                                </InfoTooltip>
                              )}
                              {canDeleteCache && isCacheDownload && (
                                <InfoTooltip content={t('cacheManagement.deleteDownload')}>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    aria-label={t('cacheManagement.deleteDownload')}
                                    className="hover:bg-destructive/10 hover:text-destructive"
                                    onClick={() =>
                                      setPendingAction({ kind: 'downloadDelete', item: download })
                                    }
                                  >
                                    <Trash2 />
                                  </Button>
                                </InfoTooltip>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                        {showDetails && (
                          <TableRow
                            className={cn('hover:bg-transparent', !isDetailsExpanded && 'border-0')}
                            aria-hidden={!isDetailsExpanded}
                          >
                            <TableCell colSpan={10} className="p-0">
                              <Collapsible open={isDetailsExpanded}>
                                <CollapsibleContent forceMount>
                                  <div className="space-y-3 bg-muted/10 px-4 py-3">
                                    {download.stage === 'interrupted' && (
                                      <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-700 dark:text-amber-300">
                                        <CircleAlert className="mt-0.5 size-4 shrink-0" />
                                        {t('cacheManagement.downloadInterruptedHint')}
                                      </div>
                                    )}
                                    {download.stage === 'failed' && download.error && (
                                      <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                                        <CircleAlert className="mt-0.5 size-4 shrink-0" />
                                        <span className="break-all">{download.error}</span>
                                      </div>
                                    )}
                                    {download.download_files.length > 0 && (
                                      <DownloadProgressDetails
                                        files={download.download_files}
                                        embedded
                                      />
                                    )}
                                  </div>
                                </CollapsibleContent>
                              </Collapsible>
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                    );
                  })}

                  {filteredCachedModels.map((item) => (
                    <TableRow
                      key={`cache:${item.model_version}:${item.actor_ip_address}:${item.path}`}
                    >
                      <TableCell className="break-words font-medium">{item.model_name}</TableCell>
                      <TableCell className="break-all font-mono text-xs">
                        {item.model_version}
                      </TableCell>
                      <TableCell>{item.model_format || '-'}</TableCell>
                      <TableCell className="font-semibold tabular-nums">
                        {item.model_size_in_billions ?? '-'}
                      </TableCell>
                      <TableCell>{item.quantization || '-'}</TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className="border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                        >
                          {t('cacheManagement.cachedState')}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <PathCell path={item.path} copyLabel={t('cacheManagement.copyPath')} />
                      </TableCell>
                      <TableCell>
                        <PathCell path={item.real_path} copyLabel={t('cacheManagement.copyPath')} />
                      </TableCell>
                      <TableCell className="break-all font-mono text-xs">
                        {item.actor_ip_address}
                      </TableCell>
                      <TableCell className="text-center">
                        {canDeleteCache ? (
                          <InfoTooltip content={t('cacheManagement.deleteCache')}>
                            <Button
                              variant="ghost"
                              size="icon"
                              aria-label={t('cacheManagement.deleteCache')}
                              className="hover:bg-destructive/10 hover:text-destructive"
                              onClick={() => setPendingAction({ kind: 'cache', item })}
                            >
                              <Trash2 />
                            </Button>
                          </InfoTooltip>
                        ) : (
                          '-'
                        )}
                      </TableCell>
                    </TableRow>
                  ))}

                  {filteredDownloads.length === 0 && filteredCachedModels.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={10} className="h-64 text-center text-muted-foreground">
                        {t('cacheManagement.noModelCacheItems')}
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TabsContent>
          )}

          {canViewEnvironments && (
            <TabsContent value="environments" className="mt-4 rounded-lg border">
              <Table className="table-fixed">
                <colgroup>
                  <col className="w-[14%]" />
                  <col className="w-[12%]" />
                  <col className="w-[10%]" />
                  <col className="w-[21%]" />
                  <col className="w-[21%]" />
                  <col className="w-[11%]" />
                  <col className="w-[11%]" />
                </colgroup>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t('cacheManagement.modelName')}</TableHead>
                    <TableHead>{t('cacheManagement.engine')}</TableHead>
                    <TableHead>{t('cacheManagement.pythonVersion')}</TableHead>
                    <TableHead>{t('cacheManagement.path')}</TableHead>
                    <TableHead>{t('cacheManagement.realPath')}</TableHead>
                    <TableHead>{t('cacheManagement.worker')}</TableHead>
                    <TableHead className="w-20">{t('common.operation')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredEnvironments.length ? (
                    filteredEnvironments.map((item) => (
                      <TableRow
                        key={`${item.model_name}:${item.model_engine}:${item.python_version}:${item.actor_ip_address}`}
                      >
                        <TableCell className="break-words font-medium">{item.model_name}</TableCell>
                        <TableCell className="break-words">{item.model_engine}</TableCell>
                        <TableCell>{item.python_version}</TableCell>
                        <TableCell>
                          <PathCell path={item.path} copyLabel={t('cacheManagement.copyPath')} />
                        </TableCell>
                        <TableCell>
                          <PathCell
                            path={item.real_path}
                            copyLabel={t('cacheManagement.copyPath')}
                          />
                        </TableCell>
                        <TableCell className="break-all font-mono text-xs">
                          {item.actor_ip_address}
                        </TableCell>
                        <TableCell>
                          {canDeleteEnvironments ? (
                            <InfoTooltip content={t('cacheManagement.deleteEnvironment')}>
                              <Button
                                variant="ghost"
                                size="icon"
                                aria-label={t('cacheManagement.deleteEnvironment')}
                                className="hover:bg-destructive/10 hover:text-destructive"
                                onClick={() => setPendingAction({ kind: 'environment', item })}
                              >
                                <Trash2 />
                              </Button>
                            </InfoTooltip>
                          ) : (
                            '-'
                          )}
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={7} className="h-64 text-center text-muted-foreground">
                        {t('cacheManagement.noVirtualEnvironments')}
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TabsContent>
          )}
        </Tabs>
      </div>

      <ConfirmDialog
        isOpen={Boolean(pendingAction)}
        onOpenChange={(open) => {
          if (!open) setPendingAction(undefined);
        }}
        description={confirmDescription}
        confirmText={
          pendingAction?.kind === 'download'
            ? t('cacheManagement.cancelDownload')
            : t('common.delete')
        }
        confirmClassName="bg-destructive text-white hover:bg-destructive/90"
        onConfirm={() => void handleConfirmAction()}
        isLoading={actionLoading}
      />
    </PageContainer>
  );
}
