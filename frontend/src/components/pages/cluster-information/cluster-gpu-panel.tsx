'use client';
import { useEffect, useState, type ReactNode } from 'react';
import { Search, LayoutGrid, List, Server, Cpu, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useI18n } from '@/contexts/i18n-context';
import { formatFileSize } from '@/lib/utils';
import { finite, availableUsagePercent, usagePercent } from './utils';
import type { ClusterDecorations } from './types';
import type { ClusterInfo, ClusterGPUInfo } from '@/types/services';
const display = (n: number | null) =>
  finite(n) ? Number(Math.max(0, Math.min(100, n)).toFixed(1)) + '%' : '—';
export const memoryRate = (g: ClusterGPUInfo) => usagePercent(g.memory_used, g.memory_total);
function Bar({ value, label }: { value: number | null; label: string }) {
  const safe = finite(value) ? Math.max(0, Math.min(100, value)) : null;
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={safe ?? undefined}
      className="h-2 overflow-hidden rounded-full bg-muted"
    >
      <div
        className={
          'h-full rounded-full transition-[width] ' +
          (safe !== null && safe > 90 ? 'bg-red-500' : 'bg-primary')
        }
        style={{ width: (safe ?? 0) + '%' }}
      />
    </div>
  );
}
export function ResourceMeter({
  label,
  value,
  detail,
}: {
  label: string;
  value: number | null;
  detail?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="mb-2 flex items-center justify-between gap-3 text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">{display(value)}</span>
      </div>
      <Bar value={value} label={label} />
      {detail && <p className="mt-2 truncate text-[11px] text-muted-foreground">{detail}</p>}
    </div>
  );
}
function DeviceMatrix({
  worker,
  list,
  decorations,
}: {
  worker: ClusterInfo;
  list: boolean;
  decorations?: ClusterDecorations;
}) {
  const { t } = useI18n();
  const [selected, setSelected] = useState<string | null>(null);
  const devices = worker.gpus ?? [];
  const [firstVisible, setFirstVisible] = useState(0);
  const [columns, setColumns] = useState(4);
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia('(min-width: 640px)');
    const update = () => setColumns(media.matches ? 4 : 2);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  const pageSize = columns * 2;
  const pageCount = Math.max(1, Math.ceil(devices.length / pageSize));
  const activePage = Math.min(Math.floor(firstVisible / pageSize), pageCount - 1);
  const pageDevices = devices.slice(activePage * pageSize, (activePage + 1) * pageSize);
  const current = pageDevices.find((g) => g.index === selected) ?? pageDevices[0];
  const movePage = (offset: number) => {
    const nextOffset = Math.max(0, Math.min(pageCount - 1, activePage + offset)) * pageSize;
    setFirstVisible(nextOffset);
    setSelected(devices[nextOffset]?.index ?? null);
  };
  const mem = (g: ClusterGPUInfo) =>
    memoryRate(g) === null
      ? t('clusterInfo.unavailable')
      : formatFileSize(g.memory_used!) + ' / ' + formatFileSize(g.memory_total!);
  if (!devices.length)
    return (
      <div className="flex items-center gap-3 border-t bg-muted/15 px-5 py-8">
        <Cpu className="h-5 w-5 shrink-0 text-muted-foreground/50" />
        <p className="text-xs text-muted-foreground">{t('clusterInfo.deviceUnavailable')}</p>
      </div>
    );
  return (
    <div className="border-t">
      <div className="flex items-center justify-between gap-3 px-5 pb-3 pt-4">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          {t('clusterInfo.devices')}{' '}
          <span className="ml-1 tabular-nums">
            {decorations?.workerGpuCount?.(worker) ?? devices.length}
          </span>
        </span>
        <div className="flex items-center gap-3">
          {!list && (
            <span className="hidden text-[11px] text-muted-foreground sm:inline">
              {t('clusterInfo.selectDevice')}
            </span>
          )}
          {!list && pageCount > 1 && (
            <div className="flex items-center gap-2">
              <span
                aria-live="polite"
                className="min-w-20 text-right text-xs tabular-nums text-muted-foreground"
              >
                {activePage * pageSize + 1}–{Math.min((activePage + 1) * pageSize, devices.length)}{' '}
                / {devices.length}
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-7 w-7 p-0"
                aria-label={t('clusterInfo.previousDevices')}
                disabled={activePage === 0}
                onClick={() => movePage(-1)}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 w-7 p-0"
                aria-label={t('clusterInfo.nextDevices')}
                disabled={activePage === pageCount - 1}
                onClick={() => movePage(1)}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </div>
      {list ? (
        <div className="overflow-x-auto px-5 pb-4">
          <table className="w-full text-left text-xs">
            <thead className="border-b text-muted-foreground">
              <tr>
                <th className="pb-3 font-normal">GPU</th>
                <th className="pb-3 font-normal">{t('clusterInfo.gpuModel')}</th>
                <th className="min-w-32 pb-3 pr-8 font-normal">{t('clusterInfo.gpuLoad')}</th>
                <th className="min-w-40 pb-3 font-normal">{t('clusterInfo.gpuMemoryInfo')}</th>
              </tr>
            </thead>
            <tbody>
              {devices.map((g) => (
                <tr key={g.index} className="border-b last:border-0">
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono">{g.index.toUpperCase()}</span>
                      {decorations?.gpuBadge?.(worker, g)}
                    </div>
                  </td>
                  <td className="pr-5 text-muted-foreground">{g.name}</td>
                  <td className="py-3 pr-8">
                    <ResourceMeter label={t('clusterInfo.gpuLoad')} value={g.utilization} />
                  </td>
                  <td className="py-3">
                    <ResourceMeter label={mem(g)} value={memoryRate(g)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <div
            key={activePage + ':' + pageSize}
            style={{ gridTemplateRows: pageCount > 1 ? 'repeat(2, minmax(0, 1fr))' : undefined }}
            className="grid grid-cols-2 gap-3 px-5 pb-4 sm:grid-cols-4 animate-in fade-in duration-200 motion-reduce:animate-none"
          >
            {pageDevices.map((g) => (
              <div key={g.index} className="relative">
                <button
                  aria-label={g.index.toUpperCase()}
                  aria-pressed={current?.index === g.index}
                  onClick={() => setSelected(g.index)}
                  className={
                    'w-full min-w-0 rounded-md border p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ' +
                    (current?.index === g.index
                      ? 'border-primary/50 bg-primary/[0.035]'
                      : 'border-border/70 bg-background hover:border-primary/30 hover:bg-muted/25')
                  }
                >
                  <div className="flex items-center gap-2 pr-10">
                    <span className="font-mono text-[10px] font-medium text-muted-foreground">
                      {g.index.toUpperCase()}
                    </span>
                    <span
                      aria-hidden="true"
                      className={
                        'h-1 w-1 shrink-0 rounded-full ' +
                        (finite(g.utilization) && g.utilization > 90 ? 'bg-red-500' : 'bg-primary')
                      }
                    />
                  </div>
                  <p className="mb-2 mt-3 text-[22px] font-medium leading-none tracking-tight tabular-nums">
                    {display(g.utilization)}
                  </p>
                  <Bar value={g.utilization} label={t('clusterInfo.gpuLoad')} />
                  <div className="mb-1.5 mt-3 flex justify-between text-[10px] text-muted-foreground">
                    <span>VRAM</span>
                    <span className="tabular-nums">{display(memoryRate(g))}</span>
                  </div>
                  <Bar value={memoryRate(g)} label={t('clusterInfo.gpuMemoryInfo')} />
                </button>
                {decorations?.gpuBadge && (
                  <div className="absolute right-2 top-2">{decorations.gpuBadge(worker, g)}</div>
                )}
              </div>
            ))}
          </div>
          {current && (
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-dashed bg-muted/15 px-5 py-3 text-[11px]">
              <span className="font-mono font-medium">{current.index.toUpperCase()}</span>
              <span className="text-muted-foreground">{current.name}</span>
              <span className="text-muted-foreground">
                VRAM <span className="ml-1 font-medium text-foreground">{mem(current)}</span>
              </span>
              <span className="text-muted-foreground">
                {t('clusterInfo.availableMemory')}{' '}
                <span className="ml-1 font-medium text-foreground">
                  {memoryRate(current) !== null && finite(current.memory_available)
                    ? formatFileSize(current.memory_available)
                    : '—'}
                </span>
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
export default function ClusterGpuPanel({
  workers,
  renderActions,
  decorations,
}: {
  workers: ClusterInfo[];
  decorations?: ClusterDecorations;
  renderActions?: (worker: ClusterInfo) => ReactNode;
}) {
  const { t } = useI18n();
  const [query, setQuery] = useState('');
  const [list, setList] = useState(false);
  const filtered = workers.filter((w) =>
    (w.ip_address + ' ' + (w.gpus ?? []).map((g) => g.name).join(' '))
      .toLowerCase()
      .includes(query.toLowerCase())
  );
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">{t('clusterInfo.deviceHint')}</p>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              className="h-9 w-56 pl-8 text-xs"
              aria-label={t('clusterInfo.searchNodes')}
              placeholder={t('clusterInfo.searchNodes')}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <div className="flex items-center rounded-md border p-0.5">
            <Button
              size="sm"
              className="h-7 w-8 p-0"
              variant={!list ? 'secondary' : 'ghost'}
              aria-label={t('clusterInfo.gridView')}
              aria-pressed={!list}
              onClick={() => setList(false)}
            >
              <LayoutGrid className="h-3.5 w-3.5" />
            </Button>
            <Button
              size="sm"
              className="h-7 w-8 p-0"
              variant={list ? 'secondary' : 'ghost'}
              aria-label={t('clusterInfo.listView')}
              aria-pressed={list}
              onClick={() => setList(true)}
            >
              <List className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </div>
      {filtered.length === 0 && (
        <p className="rounded-lg border py-16 text-center text-sm text-muted-foreground">
          {t(query ? 'clusterInfo.noMatches' : 'clusterInfo.noWorkers')}
        </p>
      )}
      {filtered.map((w) => (
        <article key={w.ip_address} className="overflow-hidden rounded-lg border bg-card">
          <div className="flex flex-wrap items-center gap-x-8 gap-y-3 px-5 py-4">
            <div className="mr-auto flex min-w-0 items-start gap-3">
              <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="break-all font-mono text-[13px] font-semibold">{w.ip_address}</h3>
                  {decorations?.workerBadge?.(w)}
                </div>
                <p
                  className="mt-1 truncate text-[11px] text-muted-foreground"
                  title={w.software_version ?? ''}
                >
                  {decorations?.workerGpuCount?.(w) ?? w.gpu_count} GPU{' '}
                  <span className="mx-1.5 text-border">/</span>
                  {w.software_version || '—'}
                </p>
              </div>
            </div>
            <div className="flex w-full items-center justify-between gap-6 sm:w-auto sm:justify-end">
              <div className="text-xs">
                <p className="flex items-center gap-3">
                  <span className="text-muted-foreground">{t('clusterInfo.cpuUsage')}</span>
                  <span className="font-medium tabular-nums">
                    {display(availableUsagePercent(w.cpu_count, w.cpu_available))}
                  </span>
                </p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {w.cpu_count ?? '—'} {t('clusterInfo.coresUnit')}
                </p>
              </div>
              <div className="text-xs">
                <p className="flex items-center gap-3">
                  <span className="text-muted-foreground">{t('clusterInfo.memoryInfo')}</span>
                  <span className="font-medium tabular-nums">
                    {display(availableUsagePercent(w.mem_total, w.mem_available))}
                  </span>
                </p>
                <p className="mt-1 text-[11px] tabular-nums text-muted-foreground">
                  {finite(w.mem_used) && finite(w.mem_total)
                    ? formatFileSize(w.mem_used) + ' / ' + formatFileSize(w.mem_total)
                    : t('clusterInfo.unavailable')}
                </p>
              </div>
              {renderActions && <div className="shrink-0 border-l pl-3">{renderActions(w)}</div>}
            </div>
          </div>
          <DeviceMatrix worker={w} list={list} decorations={decorations} />
        </article>
      ))}
    </div>
  );
}
