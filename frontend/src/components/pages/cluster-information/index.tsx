'use client';
import { useCallback, useEffect, useState } from 'react';
import {
  RefreshCw,
  Server,
  Cpu,
  Layers,
  Database,
  Settings2,
  MoreHorizontal,
  Network,
  Route,
} from 'lucide-react';
import PageContainer from '@/components/ui/page-container';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '@/components/ui/dropdown-menu';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';
import { useOperationCompletion } from '@/hooks/use-operation-completion';
import { formatFileSize } from '@/lib/utils';
import ClusterGpuPanel, { ResourceMeter } from './cluster-gpu-panel';
import { ServiceRelaunchPanel } from '../service-relaunch';
import {
  useWorkerRelaunches,
  WorkerRelaunchDialog,
  WorkerRelaunchHistory,
} from '../worker-relaunch';
import { finite, availableUsagePercent, vramSummary } from './utils';
import type { ClusterDataSource, ClusterDecorations } from './types';
import type { ClusterInfo, ClusterInformationItem, RouterNodeClusterInfo } from '@/types/services';

export default function ClusterInfoPage({
  dataSource,
  decorations,
}: { dataSource?: ClusterDataSource; decorations?: ClusterDecorations } = {}) {
  const { t } = useI18n();
  const { globalReady, clusterUIConfig, clusterVersion } = useGlobal();
  const { isAdmin } = useMenuAuth();
  const [localData, setData] = useState<ClusterInformationItem[]>([]);
  const [localLoaded, setLoaded] = useState(false);
  const [localError, setError] = useState(false);
  const [localUpdating, setUpdating] = useState(false);
  const [localUpdated, setUpdated] = useState('—');
  const [refresh, setRefresh] = useState(0);
  const external = dataSource !== undefined;
  const data = dataSource?.nodes ?? localData;
  const loaded = external ? dataSource.loaded : localLoaded;
  const updating = external ? dataSource.updating : localUpdating;
  const updated = external ? dataSource.updatedAt : localUpdated;
  const error = external ? dataSource.failed : localError;
  const [active, setActive] = useState('Worker');
  const [address, setAddress] = useState<string | null>(null);
  const relaunches = useWorkerRelaunches(globalReady && isAdmin);
  const refreshSnapshot = useCallback(() => {
    if (dataSource) dataSource.refresh();
    else setRefresh((value) => value + 1);
  }, [dataSource]);
  useOperationCompletion(relaunches.operations, refreshSnapshot);
  const routerEnabled = globalReady && clusterUIConfig?.token_router_enabled !== false;
  const workers = data.filter((n): n is ClusterInfo => n.node_type === 'Worker');
  const supervisors = data.filter((n): n is ClusterInfo => n.node_type === 'Supervisor');
  const routers = data.filter(
    (n): n is RouterNodeClusterInfo =>
      n.node_type === 'Router' && n.online && n.connectivity_status === 'online'
  );
  const gpuCount = workers.reduce((sum, n) => sum + (n.gpu_count || 0), 0);
  const reporting = workers.filter((n) => finite(n.gpu_utilization) && n.gpu_count > 0);
  const reportedCount = reporting.reduce((sum, n) => sum + n.gpu_count, 0);
  const gpuAverage =
    reportedCount > 0 && reportedCount === gpuCount
      ? reporting.reduce((sum, n) => sum + n.gpu_utilization! * n.gpu_count, 0) / reportedCount
      : null;
  const vram = vramSummary(workers);
  useEffect(() => {
    if (!globalReady || external) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      setUpdating(true);
      try {
        const response = await request.get<ClusterInformationItem[]>('/v1/cluster/info', {
          params: { detailed: true, include_routers: routerEnabled },
        });
        if (cancelled) return;
        if (!Array.isArray(response)) throw new Error('Invalid cluster information');
        setData(response);
        setError(false);
        setLoaded(true);
        setUpdated(new Date().toLocaleTimeString());
      } catch {
        if (!cancelled) {
          setError(true);
          setLoaded(true);
        }
      } finally {
        if (!cancelled) {
          setUpdating(false);
          timer = setTimeout(() => {
            void poll();
          }, 5000);
        }
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [globalReady, routerEnabled, refresh, external]);
  const tabs = [
    {
      key: 'Worker',
      icon: Server,
      label: t('clusterInfo.workers'),
      count: decorations?.summaryCount?.('workers', workers.length) ?? workers.length,
    },
    {
      key: 'Supervisor',
      icon: Network,
      label: t('clusterInfo.supervisor'),
      count: supervisors.length,
    },
    ...(routerEnabled
      ? [{ key: 'Router', icon: Route, label: t('clusterInfo.routers'), count: routers.length }]
      : []),
  ];
  const stats = [
    {
      label: t('clusterInfo.workers'),
      value: loaded
        ? (decorations?.summaryCount?.('workers', workers.length) ?? workers.length)
        : '—',
      suffix: t('clusterInfo.nodesUnit'),
      icon: Server,
      detail:
        t('clusterInfo.supervisor') +
        ' ' +
        supervisors.length +
        (routerEnabled ? ' · ' + t('clusterInfo.routers') + ' ' + routers.length : ''),
    },
    {
      label: t('clusterInfo.gpuCount'),
      value: loaded ? (decorations?.summaryCount?.('gpus', gpuCount) ?? gpuCount) : '—',
      suffix: 'GPU',
      icon: Cpu,
      detail: t('clusterInfo.allDevices'),
    },
    {
      label: t('clusterInfo.averageGpuLoad'),
      value: gpuAverage === null ? '—' : gpuAverage.toFixed(1),
      suffix: gpuAverage === null ? '' : '%',
      icon: Layers,
      detail: t('clusterInfo.deviceAverage'),
    },
    {
      label: t('clusterInfo.vramUsage'),
      value: vram ? formatFileSize(vram.used) : '—',
      suffix: '',
      icon: Database,
      detail: vram
        ? t('clusterInfo.capacity') + ' ' + formatFileSize(vram.total)
        : t('clusterInfo.unavailable'),
    },
  ];
  const controls = active === 'Supervisor' ? supervisors : routers;
  return (
    <PageContainer
      title={t('menu.clusterInfo')}
      subTitle={t('clusterInfo.overviewDescription')}
      headerClassName="gap-4 flex-wrap"
      extraContent={
        <div className="flex items-center gap-4">
          <span className="hidden items-center gap-2 text-xs text-muted-foreground sm:flex">
            <span
              className={'h-1.5 w-1.5 rounded-full ' + (error ? 'bg-amber-500' : 'bg-emerald-500')}
            />
            {updated}
          </span>
          <Button variant="outline" size="sm" disabled={updating} onClick={refreshSnapshot}>
            <RefreshCw className={'mr-2 h-3.5 w-3.5 ' + (updating ? 'animate-spin' : '')} />
            {t('clusterInfo.refresh')}
          </Button>
        </div>
      }
    >
      <div className="space-y-7">
        {dataSource?.notice}
        {error && (
          <p role="alert" className="border-l-2 border-amber-500 bg-amber-500/5 px-4 py-3 text-sm">
            {t('clusterInfo.fetchError')}
          </p>
        )}
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {stats.map((s, i) => (
            <Card key={s.label} className="rounded-lg py-5 shadow-none">
              <CardContent className="flex items-center gap-4 px-5">
                <div
                  className={
                    'shrink-0 rounded-lg p-3 ' +
                    [
                      'bg-sky-500/10 text-sky-600',
                      'bg-violet-500/10 text-violet-600',
                      'bg-emerald-500/10 text-emerald-600',
                      'bg-rose-500/10 text-rose-600',
                    ][i]
                  }
                >
                  <s.icon className="size-5" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm text-muted-foreground">{s.label}</p>
                  <p className="text-2xl font-semibold tabular-nums">
                    {s.value}
                    <span className="ml-1 text-xs font-normal text-muted-foreground">
                      {s.suffix}
                    </span>
                  </p>
                  <p className="mt-1 truncate text-xs text-muted-foreground" title={s.detail}>
                    {s.detail}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        <section>
          <div
            role="tablist"
            aria-label={t('clusterInfo.nodeType')}
            className="mb-5 flex gap-6 border-b"
          >
            {tabs.map((tab) => (
              <button
                key={tab.key}
                role="tab"
                aria-selected={active === tab.key}
                onClick={() => setActive(tab.key)}
                className={
                  'relative flex items-center gap-2 pb-3 text-sm transition-colors ' +
                  (active === tab.key
                    ? 'font-semibold text-foreground after:absolute after:inset-x-0 after:bottom-[-1px] after:h-0.5 after:bg-primary'
                    : 'text-muted-foreground hover:text-foreground')
                }
              >
                <tab.icon aria-hidden="true" className="h-4 w-4 shrink-0" />
                {tab.label}
                <span
                  className={
                    'rounded px-1.5 py-0.5 text-[11px] tabular-nums ' +
                    (active === tab.key ? 'bg-primary/10 text-primary' : 'bg-muted')
                  }
                >
                  {loaded ? tab.count : '—'}
                </span>
              </button>
            ))}
          </div>
          {!loaded ? (
            <div className="py-20 text-center text-sm text-muted-foreground">
              {t('clusterInfo.loading')}
            </div>
          ) : active === 'Worker' ? (
            <ClusterGpuPanel
              decorations={decorations}
              workers={workers}
              renderActions={
                isAdmin
                  ? (worker) => (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-8 w-8 text-muted-foreground"
                            aria-label={t('common.actions')}
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            disabled={
                              supervisors.some((n) => n.ip_address === worker.ip_address) ||
                              relaunches.busyAddresses.has(worker.ip_address)
                            }
                            onSelect={() => setAddress(worker.ip_address)}
                          >
                            <Settings2 className="mr-2 h-4 w-4" />
                            {t('clusterInfo.relaunch.title')}
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )
                  : undefined
              }
            />
          ) : (
            <div className="overflow-hidden rounded-lg border bg-card">
              <div className="grid grid-cols-1 divide-y">
                {controls.map((n) => (
                  <div
                    key={n.node_type === 'Router' ? n.node_id : n.ip_address}
                    className="grid items-center gap-6 p-5 lg:grid-cols-[1.5fr_1fr_1fr]"
                  >
                    <div>
                      <p className="font-mono text-sm font-medium">{n.ip_address}</p>
                      {n.node_type === 'Router' && n.node_id !== n.ip_address && (
                        <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
                          {n.node_id}
                        </p>
                      )}
                      <p className="mt-1.5 text-xs text-muted-foreground">
                        {n.software_version ||
                          (n.node_type === 'Supervisor' ? clusterVersion.version : '') ||
                          '—'}
                      </p>
                    </div>
                    <ResourceMeter
                      label={t('clusterInfo.cpuUsage')}
                      value={availableUsagePercent(n.cpu_count, n.cpu_available)}
                      detail={t('clusterInfo.cpuTotal') + ': ' + (n.cpu_count ?? '—')}
                    />
                    <ResourceMeter
                      label={t('clusterInfo.memoryInfo')}
                      value={availableUsagePercent(n.mem_total, n.mem_available)}
                      detail={
                        finite(n.mem_used) && finite(n.mem_total)
                          ? formatFileSize(n.mem_used) + ' / ' + formatFileSize(n.mem_total)
                          : t('clusterInfo.unavailable')
                      }
                    />
                  </div>
                ))}
                {controls.length === 0 && (
                  <div className="py-16 text-center text-sm text-muted-foreground">
                    {t(
                      active === 'Supervisor'
                        ? 'clusterInfo.noSupervisors'
                        : 'clusterInfo.noRouterInstances'
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
        {isAdmin && (
          <section className="space-y-4">
            <h2 className="text-base font-semibold">{t('clusterInfo.infrastructure')}</h2>
            <div className="space-y-4">
              <ServiceRelaunchPanel onCompleted={refreshSnapshot} />
              <WorkerRelaunchHistory operations={relaunches.operations} error={relaunches.error} />
            </div>
          </section>
        )}
        {isAdmin && (
          <WorkerRelaunchDialog
            address={address}
            onClose={() => setAddress(null)}
            onStarted={relaunches.addOperation}
          />
        )}
      </div>
    </PageContainer>
  );
}
