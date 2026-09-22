'use client';

import { useEffect, useMemo, useState } from 'react';
import { format } from 'date-fns';
import PageContainer from '@/components/ui/page-container';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { formatFileSize } from '@/lib/utils';
import type { ClusterInfo, ClusterInformationItem, RouterNodeClusterInfo } from '@/types/services';

type NodeResourceInfo = {
  cpu_count?: number | null;
  cpu_available?: number | null;
  mem_used?: number | null;
  mem_available?: number | null;
  mem_total?: number | null;
};

const isFiniteNumber = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const formatPercentage = (value: number | null): string =>
  value === null ? '-' : `${value.toFixed(2)}%`;

const calculateCpuUsageRate = (item: NodeResourceInfo): number | null => {
  if (!isFiniteNumber(item.cpu_count) || item.cpu_count <= 0) return null;
  if (!isFiniteNumber(item.cpu_available)) return null;

  const used = Math.max(0, Math.min(item.cpu_count, item.cpu_count - item.cpu_available));
  return (used / item.cpu_count) * 100;
};

const summarizeResources = (items: NodeResourceInfo[]) => {
  let cpuUsed = 0;
  let cpuTotal = 0;
  let memoryUsed = 0;
  let memoryTotal = 0;
  let cpuRateUsed = 0;
  let cpuRateTotal = 0;
  let memoryRateUsed = 0;
  let memoryRateTotal = 0;
  let cpuRateValid = items.length > 0;
  let memoryRateValid = items.length > 0;

  items.forEach((item) => {
    if (isFiniteNumber(item.cpu_count) && item.cpu_count > 0) {
      cpuTotal += item.cpu_count;
      if (isFiniteNumber(item.cpu_available)) {
        const used = Math.max(0, Math.min(item.cpu_count, item.cpu_count - item.cpu_available));
        cpuUsed += used;
        cpuRateUsed += used;
        cpuRateTotal += item.cpu_count;
      } else {
        cpuRateValid = false;
      }
    } else {
      cpuRateValid = false;
    }

    if (isFiniteNumber(item.mem_used)) {
      memoryUsed += Math.max(0, item.mem_used);
    }
    if (isFiniteNumber(item.mem_total) && item.mem_total > 0) {
      memoryTotal += item.mem_total;
      if (isFiniteNumber(item.mem_available)) {
        const used = Math.max(0, Math.min(item.mem_total, item.mem_total - item.mem_available));
        memoryRateUsed += used;
        memoryRateTotal += item.mem_total;
      } else {
        memoryRateValid = false;
      }
    } else {
      memoryRateValid = false;
    }
  });

  return {
    cpuUsed,
    cpuTotal,
    cpuUsageRate: cpuRateValid && cpuRateTotal > 0 ? (cpuRateUsed / cpuRateTotal) * 100 : null,
    memoryUsed,
    memoryTotal,
    memoryUsageRate:
      memoryRateValid && memoryRateTotal > 0 ? (memoryRateUsed / memoryRateTotal) * 100 : null,
  };
};

export default function ClusterInfoPage() {
  const [{ supervisors, workers, routers }, setData] = useState<{
    supervisors: ClusterInfo[];
    workers: ClusterInfo[];
    routers: RouterNodeClusterInfo[];
  }>({ supervisors: [], workers: [], routers: [] });
  const [lastUpdateTime, setLastUpdateTime] = useState('-');
  const { t } = useI18n();
  const { clusterVersion, clusterUIConfig, globalReady } = useGlobal();
  const tokenRouterEnabled = globalReady && clusterUIConfig?.token_router_enabled !== false;

  const supervisorSummary = useMemo(() => {
    const addresses: string[] = [];
    const metrics = summarizeResources(supervisors);
    supervisors.forEach((item) => addresses.push(item.ip_address));
    return [
      { label: t('clusterInfo.count'), value: supervisors.length },
      { label: t('clusterInfo.address'), value: addresses.join('、') || '-' },
      {
        label: t('clusterInfo.cpuInfo'),
        value: `${t('clusterInfo.rate')}${formatPercentage(metrics.cpuUsageRate)}`,
        total: `${t('clusterInfo.total')}${metrics.cpuTotal.toFixed(2)}`,
      },
      {
        label: t('clusterInfo.memoryInfo'),
        value: `${t('clusterInfo.used')}${formatFileSize(metrics.memoryUsed)}${
          metrics.memoryUsageRate === null
            ? ''
            : ` (${t('clusterInfo.rate')}${formatPercentage(metrics.memoryUsageRate)})`
        }`,
        total: `${t('clusterInfo.total')}${formatFileSize(metrics.memoryTotal)}`,
      },
      {
        label: t('clusterInfo.version'),
        value: `${t('clusterInfo.release')}${clusterVersion.version || '-'}`,
      },
    ];
  }, [clusterVersion, supervisors, t]);

  const workersSummary = useMemo(() => {
    const metrics = summarizeResources(workers);
    let gpuCount = 0;
    let gpuUtilization = 0;
    let gpuMemoryUsage = 0;
    let gpuMemoryTotal = 0;
    const nodesWithGpuLoad = workers.filter((item) => item.gpu_utilization != null).length;
    workers.forEach((item) => {
      gpuCount += item.gpu_count || 0;
      gpuUtilization += item.gpu_utilization || 0;
      const gpuTotal = Math.max(0, item.gpu_vram_total || 0);
      const gpuAvailable = Math.max(0, item.gpu_vram_available || 0);
      gpuMemoryUsage += Math.max(0, gpuTotal - gpuAvailable);
      gpuMemoryTotal += gpuTotal;
    });
    return [
      { label: t('clusterInfo.count'), value: workers.length },
      {
        label: t('clusterInfo.cpuInfo'),
        value: `${t('clusterInfo.rate')}${formatPercentage(metrics.cpuUsageRate)}`,
        total: `${t('clusterInfo.total')}${metrics.cpuTotal.toFixed(2)}`,
      },
      {
        label: t('clusterInfo.memoryInfo'),
        value: `${t('clusterInfo.used')}${formatFileSize(metrics.memoryUsed)}${
          metrics.memoryUsageRate === null
            ? ''
            : ` (${t('clusterInfo.rate')}${formatPercentage(metrics.memoryUsageRate)})`
        }`,
        total: `${t('clusterInfo.total')}${formatFileSize(metrics.memoryTotal)}`,
      },
      {
        label: t('clusterInfo.gpuInfo'),
        value: nodesWithGpuLoad
          ? `${t('clusterInfo.gpuLoad')}: ${(gpuUtilization / nodesWithGpuLoad).toFixed(2)}%`
          : `${t('clusterInfo.total')}${gpuCount}`,
        total: nodesWithGpuLoad ? `${t('clusterInfo.total')}${gpuCount}` : undefined,
      },
      {
        label: t('clusterInfo.gpuMemoryInfo'),
        value: `${t('clusterInfo.used')}${formatFileSize(gpuMemoryUsage)}`,
        total: `${t('clusterInfo.total')}${formatFileSize(gpuMemoryTotal)}`,
      },
    ];
  }, [t, workers]);

  const workerDetails = useMemo(
    () =>
      workers.map((item) => ({
        ...item,
        cpuUsage: formatPercentage(calculateCpuUsageRate(item)),
        memoryUsage: formatFileSize(item.mem_used || 0),
        memoryTotal: formatFileSize(item.mem_total || 0),
        gpuLoad:
          typeof item.gpu_utilization === 'number' ? `${item.gpu_utilization.toFixed(2)}%` : '-',
        gpuMemoryUsage: formatFileSize(
          Math.max(0, (item.gpu_vram_total || 0) - (item.gpu_vram_available || 0))
        ),
        gpuMemoryTotal: formatFileSize(item.gpu_vram_total || 0),
      })),
    [workers]
  );

  const routerSummary = useMemo(() => {
    const metrics = summarizeResources(routers);
    return [
      { label: t('clusterInfo.count'), value: routers.length },
      {
        label: t('clusterInfo.cpuInfo'),
        value: `${t('clusterInfo.rate')}${formatPercentage(metrics.cpuUsageRate)}`,
        total: `${t('clusterInfo.total')}${metrics.cpuTotal.toFixed(2)}`,
      },
      {
        label: t('clusterInfo.memoryInfo'),
        value: `${t('clusterInfo.used')}${formatFileSize(metrics.memoryUsed)}${
          metrics.memoryUsageRate === null
            ? ''
            : ` (${t('clusterInfo.rate')}${formatPercentage(metrics.memoryUsageRate)})`
        }`,
        total: `${t('clusterInfo.total')}${formatFileSize(metrics.memoryTotal)}`,
      },
    ];
  }, [routers, t]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const pollClusterInfo = async () => {
      try {
        const response = await request.get<ClusterInformationItem[]>('/v1/cluster/info', {
          params: { detailed: true, include_routers: tokenRouterEnabled },
        });
        if (cancelled) return;
        const dataList = Array.isArray(response) ? response : [];
        setLastUpdateTime(format(new Date(), 'yyyy-MM-dd HH:mm:ss'));
        setData({
          supervisors: dataList.filter(
            (item): item is ClusterInfo => item.node_type === 'Supervisor'
          ),
          workers: dataList.filter((item): item is ClusterInfo => item.node_type === 'Worker'),
          routers: tokenRouterEnabled
            ? dataList.filter(
                (item): item is RouterNodeClusterInfo =>
                  item.node_type === 'Router' &&
                  item.online &&
                  item.connectivity_status === 'online'
              )
            : [],
        });
      } catch (error) {
        if (!cancelled) console.error(error);
      } finally {
        if (!cancelled) {
          timer = setTimeout(() => {
            void pollClusterInfo();
          }, 5000);
        }
      }
    };

    void pollClusterInfo();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [tokenRouterEnabled]);

  const renderSummary = (
    rows: Array<{ label: string; value: string | number; total?: string; title?: string }>
  ) => (
    <div className="rounded-md border">
      <Table size="small">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[20%]">{t('clusterInfo.item')}</TableHead>
            <TableHead className="w-[22%]">{t('clusterInfo.value')}</TableHead>
            <TableHead className="w-[58%]" />
          </TableRow>
        </TableHeader>
        <TableBody className="[&_tr:nth-child(even)]:bg-muted/30">
          {rows.map((row) => (
            <TableRow key={row.label}>
              <TableCell>{row.label}</TableCell>
              <TableCell colSpan={row.total ? 1 : 2} className="max-w-0 truncate" title={row.title}>
                {row.value}
              </TableCell>
              {row.total && <TableCell title={row.title}>{row.total}</TableCell>}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );

  return (
    <PageContainer
      title={t('menu.clusterInfo')}
      subTitle={
        <>
          {t('clusterInfo.pageDescription')} {t('common.lastUpdateTime')}: {lastUpdateTime}
        </>
      }
    >
      <div className="space-y-6">
        <section>
          <h2 className="mb-3 text-lg font-bold text-primary">{t('clusterInfo.supervisor')}</h2>
          {renderSummary(supervisorSummary)}
        </section>

        <section>
          <h2 className="mb-3 text-lg font-bold text-primary">{t('clusterInfo.workers')}</h2>
          {renderSummary(workersSummary)}
        </section>

        {tokenRouterEnabled && (
          <section>
            <h2 className="mb-3 text-lg font-bold text-primary">{t('clusterInfo.routers')}</h2>
            {renderSummary(routerSummary)}
          </section>
        )}

        <section>
          <h2 className="mb-3 text-lg font-bold text-primary">{t('clusterInfo.workerDetails')}</h2>
          <div className="overflow-x-auto rounded-md border">
            <Table size="small">
              <TableHeader>
                <TableRow>
                  <TableHead>{t('clusterInfo.nodeType')}</TableHead>
                  <TableHead>{t('clusterInfo.address')}</TableHead>
                  <TableHead>{t('clusterInfo.version')}</TableHead>
                  <TableHead>{t('clusterInfo.cpuUsage')}</TableHead>
                  <TableHead>{t('clusterInfo.cpuTotal')}</TableHead>
                  <TableHead>{t('clusterInfo.memUsage')}</TableHead>
                  <TableHead>{t('clusterInfo.memTotal')}</TableHead>
                  <TableHead>{t('clusterInfo.gpuCount')}</TableHead>
                  <TableHead>{t('clusterInfo.gpuLoad')}</TableHead>
                  <TableHead>{t('clusterInfo.gpuMemUsage')}</TableHead>
                  <TableHead>{t('clusterInfo.gpuMemTotal')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {workerDetails.map((row) => (
                  <TableRow key={row.ip_address}>
                    <TableCell>{t('clusterInfo.worker')}</TableCell>
                    <TableCell>{row.ip_address}</TableCell>
                    <TableCell className="max-w-72 truncate" title={row.software_version || '-'}>
                      {row.software_version || '-'}
                    </TableCell>
                    <TableCell>{row.cpuUsage}</TableCell>
                    <TableCell>{row.cpu_count ?? '-'}</TableCell>
                    <TableCell>{row.memoryUsage}</TableCell>
                    <TableCell>{row.memoryTotal}</TableCell>
                    <TableCell>{row.gpu_count}</TableCell>
                    <TableCell>{row.gpuLoad}</TableCell>
                    <TableCell>{row.gpuMemoryUsage}</TableCell>
                    <TableCell>{row.gpuMemoryTotal}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </section>

        {tokenRouterEnabled && (
          <section>
            <h2 className="mb-3 text-lg font-bold text-primary">
              {t('clusterInfo.routerNodeDetails')}
            </h2>
            <div className="overflow-x-auto rounded-md border">
              <Table size="small">
                <TableHeader>
                  <TableRow>
                    <TableHead>{t('clusterInfo.nodeType')}</TableHead>
                    <TableHead>{t('clusterInfo.address')}</TableHead>
                    <TableHead>{t('clusterInfo.version')}</TableHead>
                    <TableHead>{t('clusterInfo.cpuUsage')}</TableHead>
                    <TableHead>{t('clusterInfo.cpuTotal')}</TableHead>
                    <TableHead>{t('clusterInfo.memUsage')}</TableHead>
                    <TableHead>{t('clusterInfo.memTotal')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {routers.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} className="h-24 text-center text-muted-foreground">
                        {t('clusterInfo.noRouterInstances')}
                      </TableCell>
                    </TableRow>
                  ) : (
                    routers.map((router) => {
                      const addressTitle =
                        router.node_id && router.node_id !== router.ip_address
                          ? `${router.ip_address}\n${router.node_id}`
                          : router.ip_address;
                      return (
                        <TableRow key={router.node_id}>
                          <TableCell>{t('clusterInfo.routerNode')}</TableCell>
                          <TableCell className="max-w-96 truncate" title={addressTitle}>
                            {router.ip_address || '-'}
                          </TableCell>
                          <TableCell
                            className="max-w-72 truncate"
                            title={router.software_version || '-'}
                          >
                            {router.software_version || '-'}
                          </TableCell>
                          <TableCell>{formatPercentage(calculateCpuUsageRate(router))}</TableCell>
                          <TableCell>
                            {typeof router.cpu_count === 'number'
                              ? router.cpu_count.toFixed(2)
                              : '-'}
                          </TableCell>
                          <TableCell>
                            {typeof router.mem_used === 'number'
                              ? formatFileSize(router.mem_used)
                              : '-'}
                          </TableCell>
                          <TableCell>
                            {typeof router.mem_total === 'number'
                              ? formatFileSize(router.mem_total)
                              : '-'}
                          </TableCell>
                        </TableRow>
                      );
                    })
                  )}
                </TableBody>
              </Table>
            </div>
          </section>
        )}
      </div>
    </PageContainer>
  );
}
