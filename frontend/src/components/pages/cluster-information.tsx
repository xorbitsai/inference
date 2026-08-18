'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const supervisorSummary = useMemo(() => {
    const addresses: string[] = [];
    let cpuUsage = 0;
    let cpuTotal = 0;
    let memUsage = 0;
    let memTotal = 0;
    supervisors.forEach((item) => {
      addresses.push(item.ip_address);
      cpuUsage += (item.cpu_count || 0) - (item.cpu_available || 0);
      cpuTotal += item.cpu_count || 0;
      memUsage += item.mem_used || 0;
      memTotal += item.mem_total || 0;
    });
    return [
      { label: t('clusterInfo.count'), value: supervisors.length },
      { label: t('clusterInfo.address'), value: addresses.join('、') || '-' },
      {
        label: t('clusterInfo.cpuInfo'),
        value: `${t('clusterInfo.usage')}${cpuUsage.toFixed(2)}`,
        total: `${t('clusterInfo.total')}${cpuTotal.toFixed(2)}`,
      },
      {
        label: t('clusterInfo.cpuMemoryInfo'),
        value: `${t('clusterInfo.usage')}${formatFileSize(memUsage)}`,
        total: `${t('clusterInfo.total')}${formatFileSize(memTotal)}`,
      },
      {
        label: t('clusterInfo.version'),
        value: `${t('clusterInfo.release')}${clusterVersion.version || '-'}`,
        total: `${t('clusterInfo.commit')}${clusterVersion['full-revisionid'] || '-'}`,
      },
    ];
  }, [clusterVersion, supervisors, t]);

  const workersSummary = useMemo(() => {
    let cpuUsage = 0;
    let cpuTotal = 0;
    let cpuMemUsage = 0;
    let cpuMemTotal = 0;
    let gpuCount = 0;
    let gpuUtilization = 0;
    let gpuMemoryUsage = 0;
    let gpuMemoryTotal = 0;
    const nodesWithGpuLoad = workers.filter((item) => item.gpu_utilization != null).length;
    workers.forEach((item) => {
      cpuUsage += (item.cpu_count || 0) - (item.cpu_available || 0);
      cpuTotal += item.cpu_count || 0;
      cpuMemUsage += item.mem_used || 0;
      cpuMemTotal += item.mem_total || 0;
      gpuCount += item.gpu_count || 0;
      gpuUtilization += item.gpu_utilization || 0;
      gpuMemoryUsage += (item.gpu_vram_total || 0) - (item.gpu_vram_available || 0);
      gpuMemoryTotal += item.gpu_vram_total || 0;
    });
    return [
      { label: t('clusterInfo.count'), value: workers.length },
      {
        label: t('clusterInfo.cpuInfo'),
        value: `${t('clusterInfo.usage')}${cpuUsage.toFixed(2)}`,
        total: `${t('clusterInfo.total')}${cpuTotal.toFixed(2)}`,
      },
      {
        label: t('clusterInfo.cpuMemoryInfo'),
        value: `${t('clusterInfo.usage')}${formatFileSize(cpuMemUsage)}`,
        total: `${t('clusterInfo.total')}${formatFileSize(cpuMemTotal)}`,
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
        value: `${t('clusterInfo.usage')}${formatFileSize(gpuMemoryUsage)}`,
        total: `${t('clusterInfo.total')}${formatFileSize(gpuMemoryTotal)}`,
      },
      {
        label: t('clusterInfo.version'),
        value: `${t('clusterInfo.release')}${clusterVersion.version || '-'}`,
        total: `${t('clusterInfo.commit')}${clusterVersion['full-revisionid'] || '-'}`,
      },
    ];
  }, [clusterVersion, t, workers]);

  const workerDetails = useMemo(
    () =>
      workers.map((item) => ({
        ...item,
        cpuUsage: ((item.cpu_count || 0) - (item.cpu_available || 0)).toFixed(2),
        cpuMemUsage: formatFileSize(item.mem_used || 0),
        cpuMemTotal: formatFileSize(item.mem_total || 0),
        gpuLoad:
          typeof item.gpu_utilization === 'number' ? `${item.gpu_utilization.toFixed(2)}%` : '-',
        gpuMemoryUsage: formatFileSize((item.gpu_vram_total || 0) - (item.gpu_vram_available || 0)),
        gpuMemoryTotal: formatFileSize(item.gpu_vram_total || 0),
      })),
    [workers]
  );

  const routerSummary = useMemo(() => {
    const cpuUsage = routers.reduce(
      (total, item) =>
        total +
        (typeof item.cpu_count === 'number' && typeof item.cpu_available === 'number'
          ? item.cpu_count - item.cpu_available
          : 0),
      0
    );
    const cpuTotal = routers.reduce((total, item) => total + (item.cpu_count || 0), 0);
    const memoryUsage = routers.reduce((total, item) => total + (item.mem_used || 0), 0);
    const memoryTotal = routers.reduce((total, item) => total + (item.mem_total || 0), 0);
    const versionPairs = new Set(
      routers.map((item) => `${item.software_version || '-'}@${item.software_revision || '-'}`)
    );
    const softwareVersions = [
      ...new Set(routers.map((item) => item.software_version).filter(Boolean)),
    ];
    const revisions = [...new Set(routers.map((item) => item.software_revision).filter(Boolean))];
    const versionsConsistent = versionPairs.size <= 1;
    const versionDetails = routers
      .map(
        (item) =>
          `${item.node_id}: ${item.software_version || '-'}@${item.software_revision || '-'}`
      )
      .join('\n');
    const release =
      routers.length === 0
        ? '-'
        : versionsConsistent
          ? softwareVersions[0] || '-'
          : t('clusterInfo.routerMultipleVersions');
    const revision =
      routers.length === 0
        ? '-'
        : versionsConsistent
          ? revisions[0] || '-'
          : t('clusterInfo.routerMultipleVersions');

    return [
      { label: t('clusterInfo.count'), value: routers.length },
      {
        label: t('clusterInfo.cpuInfo'),
        value: `${t('clusterInfo.usage')}${cpuUsage.toFixed(2)}`,
        total: `${t('clusterInfo.total')}${cpuTotal.toFixed(2)}`,
      },
      {
        label: t('clusterInfo.cpuMemoryInfo'),
        value: `${t('clusterInfo.usage')}${formatFileSize(memoryUsage)}`,
        total: `${t('clusterInfo.total')}${formatFileSize(memoryTotal)}`,
      },
      {
        label: t('clusterInfo.version'),
        value: `${t('clusterInfo.release')}${release}`,
        total: `${t('clusterInfo.commit')}${revision}`,
        title: versionDetails,
      },
    ];
  }, [routers, t]);

  const fetchClusterInfo = useCallback(async () => {
    try {
      const response = await request.get<ClusterInformationItem[]>('/v1/cluster/info', {
        params: { detailed: true, include_routers: tokenRouterEnabled },
      });
      setLastUpdateTime(format(new Date(), 'yyyy-MM-dd HH:mm:ss'));
      setData({
        supervisors: response.filter(
          (item): item is ClusterInfo => item.node_type === 'Supervisor'
        ),
        workers: response.filter((item): item is ClusterInfo => item.node_type === 'Worker'),
        routers: tokenRouterEnabled
          ? response.filter(
              (item): item is RouterNodeClusterInfo =>
                item.node_type === 'Router' && item.online && item.connectivity_status === 'online'
            )
          : [],
      });
    } catch (error) {
      console.error(error);
    } finally {
      timerRef.current = setTimeout(fetchClusterInfo, 5000);
    }
  }, [tokenRouterEnabled]);

  useEffect(() => {
    void fetchClusterInfo();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [fetchClusterInfo]);

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
                    <TableCell>{row.cpuUsage}</TableCell>
                    <TableCell>{row.cpu_count ?? '-'}</TableCell>
                    <TableCell>{row.cpuMemUsage}</TableCell>
                    <TableCell>{row.cpuMemTotal}</TableCell>
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
                    <TableHead>{t('clusterInfo.cpuUsage')}</TableHead>
                    <TableHead>{t('clusterInfo.cpuTotal')}</TableHead>
                    <TableHead>{t('clusterInfo.memUsage')}</TableHead>
                    <TableHead>{t('clusterInfo.memTotal')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {routers.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} className="h-24 text-center text-muted-foreground">
                        {t('clusterInfo.noRouterInstances')}
                      </TableCell>
                    </TableRow>
                  ) : (
                    routers.map((router) => {
                      const cpuUsage =
                        typeof router.cpu_count === 'number' &&
                        typeof router.cpu_available === 'number'
                          ? (router.cpu_count - router.cpu_available).toFixed(2)
                          : '-';
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
                          <TableCell>{cpuUsage}</TableCell>
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
