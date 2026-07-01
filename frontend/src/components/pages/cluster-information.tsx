'use client';

import { useEffect, useRef, useState, useMemo } from 'react';
import { format } from 'date-fns';
import request from '@/lib/request';
import { formatFileSize } from '@/lib/utils';
import { useI18n } from '@/contexts/i18n-context';
import PageContainer from '@/components/ui/page-container';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { ClusterInfo } from '@/types/services';
import { useGlobal } from '@/contexts/global-context';

export default function ClusterInfo() {
  const [{ supervisors, workers }, setData] = useState<{
    supervisors: ClusterInfo[];
    workers: ClusterInfo[];
  }>({
    supervisors: [],
    workers: [],
  });
  const [lastUpdateTime, setLastUpdateTime] = useState('-');
  const { t } = useI18n();
  const { clusterVersion } = useGlobal();
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const supervisorSummary = useMemo(() => {
    const address: string[] = [];
    let cpuUsage = 0;
    let cpuTotal = 0;
    let memUsage = 0;
    let memTotal = 0;
    supervisors.forEach((item) => {
      address.push(item.ip_address);
      cpuUsage += (item.cpu_count || 0) - (item.cpu_available || 0);
      cpuTotal += item.cpu_count || 0;
      memUsage += item.mem_used || 0;
      memTotal += item.mem_total || 0;
    });
    return [
      {
        label: t('clusterInfo.count'),
        value: supervisors.length,
      },
      {
        label: t('clusterInfo.address'),
        value: address.join('、'),
      },
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
  }, [supervisors, clusterVersion]);
  const workersSummary = useMemo(() => {
    let cpuUsage = 0;
    let cpuTotal = 0;
    let cpuMemUsage = 0;
    let cpuMemTotal = 0;
    let gpuCount = 0;
    let gpuUtilization = 0;
    let gpuMemoryUsage = 0;
    let gpuMemoryTotal = 0;

    const nodesWithGpuLoad = workers.filter((obj) => obj['gpu_utilization'] != null).length;
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
      {
        label: t('clusterInfo.count'),
        value: workers.length,
      },
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
        ...(nodesWithGpuLoad
          ? {
              value: `${t('clusterInfo.gpuLoad')}: ${(gpuUtilization / nodesWithGpuLoad).toFixed(
                2
              )}%`,
              total: `${t('clusterInfo.total')}${gpuCount}`,
            }
          : {
              value: `${t('clusterInfo.total')}${gpuCount}`,
            }),
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
  }, [workers, clusterVersion]);
  const workerDetails = useMemo(() => {
    return workers.map((item) => ({
      ...item,
      cpuUsage: ((item.cpu_count || 0) - (item.cpu_available || 0)).toFixed(2),
      cpuMemUsage: formatFileSize(item.mem_used || 0),
      cpuMemTotal: formatFileSize(item.mem_total || 0),
      gpuLoad:
        typeof item.gpu_utilization === 'number' ? `${item.gpu_utilization.toFixed(2)}%` : '-',
      gpuMemoryUsage: formatFileSize((item.gpu_vram_total || 0) - (item.gpu_vram_available || 0)),
      gpuMemoryTotal: formatFileSize(item.gpu_vram_total || 0),
    }));
  }, [workers]);
  const fetchClusterInfo = async () => {
    try {
      const res = await request.get<ClusterInfo[]>('/v1/cluster/info', {
        params: { detailed: true },
      });
      setLastUpdateTime(format(new Date(), 'yyyy-MM-dd HH:mm:ss'));
      setData({
        supervisors: res.filter((item) => item.node_type === 'Supervisor'),
        workers: res.filter((item) => item.node_type === 'Worker'),
      });
      timerRef.current = setTimeout(fetchClusterInfo, 5000);
    } catch (err) {
      console.log(err)
      if(timerRef.current){
        clearTimeout(timerRef.current); 
      }
    }
  };

  useEffect(() => {
    fetchClusterInfo();
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);
  return (
    <PageContainer
      title={t('menu.clusterInfo')}
      subTitle={
        <>
          {t('menu.clusterInfoDesc')}
          {t('common.lastUpdateTime')}: {lastUpdateTime}
        </>
      }
    >
      <div className="space-y-6">
        <div>
          <div className="text-primary text-lg mb-3 font-bold">{t('clusterInfo.supervisor')}</div>
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
                {supervisorSummary.map((row) => (
                  <TableRow key={row.label}>
                    <TableCell>{row.label}</TableCell>
                    {row.total ? (
                      <>
                        <TableCell>{row.value}</TableCell>
                        <TableCell>{row.total}</TableCell>
                      </>
                    ) : (
                      <TableCell colSpan={2}>{row.value}</TableCell>
                    )}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
        
        <div>
          <div className="text-primary text-lg mb-3 font-bold">{t('clusterInfo.workers')}</div>
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
                {workersSummary.map((row) => (
                  <TableRow key={row.label}>
                    <TableCell>{row.label}</TableCell>
                    {row.total ? (
                      <>
                        <TableCell>{row.value}</TableCell>
                        <TableCell>{row.total}</TableCell>
                      </>
                    ) : (
                      <TableCell colSpan={2}>{row.value}</TableCell>
                    )}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>

        <div>
          <div className="text-primary text-lg mb-3 font-bold">
            {t('clusterInfo.workerDetails')}
          </div>
          <div className="rounded-md border">
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
                {workerDetails.map((row, index) => (
                  <TableRow key={`worker${index}`}>
                    <TableCell>{t('clusterInfo.worker')}</TableCell>
                    <TableCell>{row.ip_address}</TableCell>
                    <TableCell>{row.cpuUsage}</TableCell>
                    <TableCell>{row.cpu_count}</TableCell>
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
        </div>
      </div>
    </PageContainer>
  );
}
