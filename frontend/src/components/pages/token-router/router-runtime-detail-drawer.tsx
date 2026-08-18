'use client';

import { format } from 'date-fns';
import type { ReactNode } from 'react';

import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { useI18n } from '@/contexts/i18n-context';
import { formatFileSize } from '@/lib/utils';
import type { RouterClusterInfo } from '@/types/services';
import { RouterStatusBadge } from './router-status-badge';

interface Props {
  router: RouterClusterInfo | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function formatTimestamp(value?: number | null) {
  return typeof value === 'number' ? format(new Date(value * 1000), 'yyyy-MM-dd HH:mm:ss') : '-';
}

function formatDuration(value?: number) {
  if (typeof value !== 'number') return '-';
  const totalSeconds = Math.max(0, Math.floor(value));
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [
    days ? `${days}d` : '',
    hours ? `${hours}h` : '',
    minutes ? `${minutes}m` : '',
    `${seconds}s`,
  ]
    .filter(Boolean)
    .join(' ');
}

function formatBytes(value?: number) {
  return typeof value === 'number' ? formatFileSize(value) : '-';
}

function DetailSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold text-primary">{title}</h3>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">{children}</dl>
    </section>
  );
}

function DetailItem({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-1 break-all text-sm">{value ?? '-'}</dd>
    </div>
  );
}

export function RouterRuntimeDetailDrawer({ router, open, onOpenChange }: Props) {
  const { t } = useI18n();
  const resources = router?.process?.resources;
  const workerPids = router?.process?.tokenization?.worker_pids;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-3xl">
        <SheetHeader>
          <SheetTitle>{t('clusterInfo.routerRuntimeDetails')}</SheetTitle>
        </SheetHeader>
        {router && (
          <div className="mt-6 space-y-4 pb-6">
            <DetailSection title={t('clusterInfo.routerBasicInfo')}>
              <DetailItem label={t('clusterInfo.routerUid')} value={router.router_uid} />
              <DetailItem label={t('clusterInfo.routerInstanceId')} value={router.instance_id} />
              <DetailItem
                label={t('clusterInfo.routerHostname')}
                value={router.metadata?.hostname || '-'}
              />
              <DetailItem label={t('clusterInfo.endpoint')} value={router.endpoint || '-'} />
              <DetailItem
                label={t('clusterInfo.status')}
                value={<RouterStatusBadge status={router.router_status} />}
              />
              <DetailItem
                label={t('clusterInfo.routerOnlineStatus')}
                value={
                  router.online ? t('clusterInfo.routerOnline') : t('clusterInfo.routerOffline')
                }
              />
              <DetailItem
                label={t('clusterInfo.routerRegisteredAt')}
                value={formatTimestamp(router.registered_at)}
              />
              <DetailItem
                label={t('clusterInfo.routerLastHeartbeat')}
                value={formatTimestamp(router.last_heartbeat)}
              />
            </DetailSection>

            <DetailSection title={t('clusterInfo.routerProcessInfo')}>
              <DetailItem
                label={t('clusterInfo.routerMainPid')}
                value={router.process?.pid ?? '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerTokenizerWorkerPids')}
                value={workerPids?.length ? workerPids.join(', ') : '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerChildProcessCount')}
                value={resources?.child_process_count ?? '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerThreadCount')}
                value={resources?.thread_count ?? '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerStartedAt')}
                value={formatTimestamp(resources?.started_at ?? router.metadata?.started_at)}
              />
              <DetailItem
                label={t('clusterInfo.routerUptime')}
                value={formatDuration(resources?.uptime_seconds)}
              />
            </DetailSection>

            <DetailSection title={t('clusterInfo.routerPerformanceInfo')}>
              <DetailItem
                label={t('clusterInfo.routerProcessCpu')}
                value={
                  typeof resources?.cpu_percent === 'number'
                    ? `${resources.cpu_percent.toFixed(2)}%`
                    : '-'
                }
              />
              <DetailItem
                label={t('clusterInfo.routerCpuCores')}
                value={
                  typeof resources?.cpu_cores === 'number' ? resources.cpu_cores.toFixed(3) : '-'
                }
              />
              <DetailItem
                label={t('clusterInfo.routerProcessRss')}
                value={formatBytes(resources?.rss_bytes)}
              />
              <DetailItem
                label={t('clusterInfo.routerMainRss')}
                value={formatBytes(resources?.main_process_rss_bytes)}
              />
              <DetailItem
                label={t('clusterInfo.routerChildRss')}
                value={formatBytes(resources?.child_process_rss_bytes)}
              />
              <DetailItem
                label={t('clusterInfo.routerSampledAt')}
                value={formatTimestamp(resources?.sampled_at)}
              />
            </DetailSection>

            <DetailSection title={t('clusterInfo.routerVersionInfo')}>
              <DetailItem
                label={t('clusterInfo.routerProtocolVersion')}
                value={router.protocol_version || router.version || '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerSoftwareVersion')}
                value={router.software_version || '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerSoftwareRevision')}
                value={router.software_revision || '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerPythonVersion')}
                value={router.metadata?.python_version || '-'}
              />
              <DetailItem
                label={t('clusterInfo.routerPlatform')}
                value={router.metadata?.platform || '-'}
              />
            </DetailSection>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
