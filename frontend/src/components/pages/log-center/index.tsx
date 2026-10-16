'use client';

import { useState } from 'react';
import { AlertTriangle, ArrowDown, Database, Radio, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import PageContainer from '@/components/ui/page-container';
import { Select } from '@/components/ui/select';
import { DEFAULT_LOG_TIME_RANGE, LOG_REFRESH_OPTIONS } from '@/constants/logs';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

import ElasticLogs from './elastic-logs';
import RuntimeLogs from './runtime-logs';
import { TimeRangePicker } from './time-range-picker';
import type {
  HistoricalLogHandoff,
  LogCenterMode,
  RuntimeNodeHandoff,
  TimeRangeValue,
} from './types';

const DEFAULT_RUNTIME_MAX_LINES = 100;
const MAX_RUNTIME_DISPLAY_LINES = 10000;

export default function LogCenter() {
  const { t } = useI18n();
  const { clusterUIConfig, globalReady } = useGlobal();
  const [mode, setMode] = useState<LogCenterMode>('runtime');
  const [historicalHandoff, setHistoricalHandoff] = useState<HistoricalLogHandoff>();
  const [runtimeHandoff, setRuntimeHandoff] = useState<RuntimeNodeHandoff>();
  const [runtimeMaxLinesInput, setRuntimeMaxLinesInput] = useState(
    String(DEFAULT_RUNTIME_MAX_LINES)
  );
  const [runtimeFollowTail, setRuntimeFollowTail] = useState(true);
  const [historicalTimeRange, setHistoricalTimeRange] =
    useState<TimeRangeValue>(DEFAULT_LOG_TIME_RANGE);
  const [historicalRefreshInterval, setHistoricalRefreshInterval] = useState(0);

  if (!globalReady) return <PageContainer loading />;

  const indexedEnabled = Boolean(clusterUIConfig?.es_enabled);

  const openHistoricalLogs = (request: Omit<HistoricalLogHandoff, 'token'>) => {
    setHistoricalHandoff({ ...request, token: Date.now() });
    setMode('historical');
  };

  const openRuntimeLogs = (nodeName: string) => {
    setRuntimeHandoff({ nodeName, token: Date.now() });
    setMode('runtime');
  };

  const normalizeRuntimeMaxLines = () => {
    const parsed = Number(runtimeMaxLinesInput);
    const normalized =
      Number.isSafeInteger(parsed) && parsed > 0
        ? Math.min(parsed, MAX_RUNTIME_DISPLAY_LINES)
        : DEFAULT_RUNTIME_MAX_LINES;
    setRuntimeMaxLinesInput(String(normalized));
  };

  return (
    <PageContainer
      title={t('menu.logCenter')}
      subTitle={t('logCenter.workspaceDescription')}
      className="h-full gap-4"
    >
      <div className="mb-4 rounded-md border bg-background">
        <div className="flex flex-wrap items-center gap-3 border-b px-4 py-3">
          <span className="text-sm font-medium">{t('logCenter.dataMode')}</span>
          <div className="flex gap-2">
            <Button
              aria-pressed={mode === 'runtime'}
              variant={mode === 'runtime' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMode('runtime')}
            >
              <Radio className="size-4" />
              {t('logCenter.liveTracking')}
            </Button>
            <Button
              aria-pressed={mode === 'historical'}
              variant={mode === 'historical' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMode('historical')}
            >
              <Database className="size-4" />
              {t('logCenter.historicalSearch')}
              {!indexedEnabled && <AlertTriangle className="size-3.5 text-amber-500" />}
            </Button>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2 text-sm text-muted-foreground">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-foreground">{t('logCenter.dataSource')}:</span>
            <span>
              {mode === 'runtime'
                ? t('logCenter.runtimeDataSource')
                : t('logCenter.historicalDataSource')}
            </span>
            {mode === 'runtime' && (
              <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
                {t('logCenter.updateEveryTwoSeconds')}
              </span>
            )}
          </div>

          {mode === 'runtime' ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-foreground">
                {t('logCenter.runtimeSettings')}
              </span>
              <label htmlFor="runtime-log-max-lines" className="text-sm">
                {t('logCenter.maxLines')}
              </label>
              <Input
                id="runtime-log-max-lines"
                type="number"
                min={1}
                max={MAX_RUNTIME_DISPLAY_LINES}
                step={1}
                className="h-8 w-24"
                value={runtimeMaxLinesInput}
                onChange={(event) => setRuntimeMaxLinesInput(event.target.value)}
                onBlur={normalizeRuntimeMaxLines}
              />
              <Button
                variant={runtimeFollowTail ? 'secondary' : 'outline'}
                size="sm"
                onClick={() => setRuntimeFollowTail((value) => !value)}
              >
                <ArrowDown className="size-4" />
                {t('logCenter.autoFollow')}
              </Button>
            </div>
          ) : (
            indexedEnabled && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium text-foreground">
                  {t('logCenter.historicalSettings')}
                </span>
                <TimeRangePicker value={historicalTimeRange} onChange={setHistoricalTimeRange} />
                <Select
                  value={historicalRefreshInterval}
                  onChange={(value) => setHistoricalRefreshInterval(Number(value || 0))}
                  options={LOG_REFRESH_OPTIONS.map((option) => ({
                    value: option.value,
                    label: t(option.labelKey),
                    prefix: <RefreshCw className="size-4" />,
                  }))}
                  allowClear={false}
                  className="w-36"
                />
              </div>
            )
          )}
        </div>
      </div>

      <div className={cn(mode !== 'runtime' && 'hidden')} aria-hidden={mode !== 'runtime'}>
        <RuntimeLogs
          active={mode === 'runtime'}
          historicalSearchEnabled={indexedEnabled}
          targetNodeRequest={runtimeHandoff}
          maxLinesInput={runtimeMaxLinesInput}
          followTail={runtimeFollowTail}
          onFollowTailChange={setRuntimeFollowTail}
          onViewHistorical={openHistoricalLogs}
        />
      </div>
      <div className={cn(mode !== 'historical' && 'hidden')} aria-hidden={mode !== 'historical'}>
        <ElasticLogs
          active={mode === 'historical'}
          enabled={indexedEnabled}
          handoff={historicalHandoff}
          timeRange={historicalTimeRange}
          onTimeRangeChange={setHistoricalTimeRange}
          refreshInterval={historicalRefreshInterval}
          onRefreshIntervalChange={setHistoricalRefreshInterval}
          onViewRuntimeNode={openRuntimeLogs}
        />
      </div>
    </PageContainer>
  );
}
