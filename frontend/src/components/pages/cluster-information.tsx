'use client';

import { useEffect, useRef, useState, useMemo, useCallback } from 'react';

import request from '@/lib/request';
import { formatBytes } from '@/lib/utils';
import { useI18n } from '@/contexts/i18n-context';
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
  const { t } = useI18n();
  const { clusterVersion } = useGlobal();
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const buildSummary = useCallback((list: ClusterInfo[]) => {
    const count = list.length;

    let cpuUsage = 0;
    let cpuTotal = 0;

    let memUsage = 0;
    let memTotal = 0;

    let gpuCount = 0;
    let gpuUsage = 0;

    let gpuMemoryUsage = 0;
    let gpuMemoryTotal = 0;

    list.forEach((item) => {
      cpuUsage += item.cpu_count - item.cpu_available;
      cpuTotal += item.cpu_count;
      memUsage += item.mem_used;
      memTotal += item.mem_total;
      gpuCount += item.gpu_count;
      gpuUsage += item.gpu_utilization ?? 0;
      gpuMemoryUsage += item.gpu_vram_total - item.gpu_vram_available;
      gpuMemoryTotal += item.gpu_vram_total;
    });

    return {
      count,
      cpuUsage,
      cpuTotal,
      memUsage,
      memTotal,
      gpuCount,
      gpuUsage,
      gpuMemoryUsage,
      gpuMemoryTotal,
      address: list[0]?.ip_address,
    };
  }, []);

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
        value: `${t('clusterInfo.usage')}${formatBytes(memUsage)}`,
        total: `${t('clusterInfo.total')}${formatBytes(memTotal)}`,
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
        value: `${t('clusterInfo.usage')}${formatBytes(cpuMemUsage)}`,
        total: `${t('clusterInfo.total')}${formatBytes(cpuMemTotal)}`,
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
        value: `${t('clusterInfo.usage')}${formatBytes(gpuMemoryUsage)}`,
        total: `${t('clusterInfo.total')}${formatBytes(gpuMemoryTotal)}`,
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
      cpuMemUsage: formatBytes(item.mem_used || 0),
      cpuMemTotal: formatBytes(item.mem_total || 0),
      gpuLoad: item.gpu_utilization !== null ? `${item.gpu_utilization.toFixed(2)}%` : '-',
      gpuMemoryUsage: formatBytes((item.gpu_vram_total || 0) - (item.gpu_vram_available || 0)),
      gpuMemoryTotal: formatBytes(item.gpu_vram_total || 0),
    }));
  }, [workers]);
  const fetchClusterInfo = async () => {
    try {
      const res = await request.get<ClusterInfo[]>('/v1/cluster/info', {
        params: { detailed: true },
      });

      setData({
        supervisors: res.filter((item) => item.node_type === 'Supervisor'),
        workers: res.filter((item) => item.node_type === 'Worker'),
      });
    } catch (err) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  useEffect(() => {
    fetchClusterInfo();
    timerRef.current = setInterval(() => {
      fetchClusterInfo();
    }, 5000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);
  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-1">{t('menu.clusterInfo')}</h1>
        <p className="text-muted-foreground">{t('menu.clusterInfoDesc')}</p>
      </div>
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
        <div className="text-primary text-lg mb-3 font-bold">{t('clusterInfo.workerDetails')}</div>
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
  );
}
