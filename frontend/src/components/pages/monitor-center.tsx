'use client';

import { Check, ChevronDown, Clock, RotateCw } from 'lucide-react';
import { useMemo, useState, useRef, useEffect } from 'react';
import { format } from 'date-fns';
import { useTheme } from 'next-themes';

import PageContainer from '@/components/ui/page-container';
import { Button } from '@/components/ui/button';
import { Select } from '@/components/ui/select';
import { DateTimePicker } from '@/components/ui/date-time-picker';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { TIME_RANGES, REFRESH } from '@/constants/monitor';
import { cn } from '@/lib/utils';

type TimeRangeValue = {
  from: string;
  to: string;
};

type TimeRangeOption = {
  label: string;
  value: string;
};

const DEFAULT_TIME_RANGE: TimeRangeValue = {
  from: 'now-1h',
  to: 'now',
};
const DEFAULT_GRAFANA_REFRESH = '1m';
const DEFAULT_REFRESH_INTERVAL = 60000;

const parseDateTime = (value: string) => {
  const timestamp = Number(value);
  return new Date(Number.isNaN(timestamp) ? value : timestamp);
};

const formatDateTime = (value: string) => format(parseDateTime(value), 'yyyy-MM-dd HH:mm');

const toMilliseconds = (value: string) => String(new Date(value).getTime());

const joinGrafanaPath = (baseUrl: string, path: string) => {
  return `${baseUrl.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
};

const buildQueryParam = (key: string, value: string) => {
  return `${key}=${encodeURIComponent(value)}`;
};

function TimeRangePicker({
  options,
  value,
  onChange,
}: {
  options: TimeRangeOption[];
  value: TimeRangeValue;
  onChange: (value: TimeRangeValue) => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [absoluteFrom, setAbsoluteFrom] = useState('');
  const [absoluteTo, setAbsoluteTo] = useState('');

  const selectedRelativeOption = options.find(
    (option) => option.value === value.from && value.to === 'now'
  );
  const isAbsoluteRange = value.to !== 'now';
  const hasAbsoluteRange = Boolean(absoluteFrom && absoluteTo);
  const absoluteRangeInvalid =
    hasAbsoluteRange && new Date(absoluteTo).getTime() <= new Date(absoluteFrom).getTime();
  const canApplyAbsoluteRange = hasAbsoluteRange && !absoluteRangeInvalid;
  const triggerLabel = isAbsoluteRange
    ? `${formatDateTime(value.from)} ~ ${formatDateTime(value.to)}`
    : selectedRelativeOption?.label || options[0]?.label;
  const handleApplyAbsoluteRange = () => {
    if (!canApplyAbsoluteRange) return;
    onChange({
      from: toMilliseconds(absoluteFrom),
      to: toMilliseconds(absoluteTo),
    });
    setOpen(false);
  };

  const handleSelectRelativeRange = (option: TimeRangeOption) => {
    onChange({
      from: option.value,
      to: 'now',
    });
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            'h-10 min-w-0 justify-between px-3',
            isAbsoluteRange ? 'w-[24rem] max-w-[calc(100vw-9rem)]' : 'w-40'
          )}
        >
          <span className="flex min-w-0 items-center gap-2">
            <Clock className="size-4 text-muted-foreground" />
            <span className="truncate">{triggerLabel}</span>
          </span>
          <ChevronDown className="size-4 text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[min(calc(100vw-2rem),30rem)] p-0">
        <div className="grid max-h-[28rem] grid-cols-1 overflow-hidden sm:grid-cols-[18rem_12rem]">
          <div className="border-b p-4 sm:border-b-0 sm:border-r">
            <div className="mb-4 text-sm font-medium text-muted-foreground">
              {t('monitorCenter.absoluteRange')}
            </div>
            <div className="space-y-3">
              <DateTimePicker
                value={absoluteFrom}
                onChange={setAbsoluteFrom}
                aria-label={t('common.from')}
                placeholder={t('common.from')}
                showClear={false}
                showSelectedTime={false}
              />
              <DateTimePicker
                value={absoluteTo}
                onChange={setAbsoluteTo}
                aria-label={t('common.to')}
                placeholder={t('common.to')}
                inputClassName={absoluteRangeInvalid ? 'border-destructive' : undefined}
                showClear={false}
                showSelectedTime={false}
              />
              {absoluteRangeInvalid && (
                <div className="text-xs text-destructive">
                  {t('monitorCenter.invalidTimeRange')}
                </div>
              )}
              <Button
                className="w-full"
                disabled={!canApplyAbsoluteRange}
                onClick={handleApplyAbsoluteRange}
              >
                {t('monitorCenter.applyTimeRange')}
              </Button>
            </div>
          </div>
          <div className="overflow-y-auto py-2">
            {options.map((option) => {
              const selected = value.from === option.value && value.to === 'now';

              return (
                <button
                  key={option.value}
                  type="button"
                  className={cn(
                    'flex h-11 w-full items-center justify-between px-4 text-left text-sm transition-colors hover:bg-accent',
                    selected && 'bg-accent text-accent-foreground'
                  )}
                  onClick={() => handleSelectRelativeRange(option)}
                >
                  <span>{option.label}</span>
                  {selected && <Check className="size-4 text-primary" />}
                </button>
              );
            })}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

const MonitorCenter = () => {
  const { clusterUIConfig, globalReady } = useGlobal();
  const { t } = useI18n();
  const { theme } = useTheme();
  const [timeRange, setTimeRange] = useState<TimeRangeValue>(DEFAULT_TIME_RANGE);
  const [grafanaRefresh, setGrafanaRefresh] = useState(DEFAULT_GRAFANA_REFRESH);
  const [refreshInterval, setRefreshInterval] = useState(DEFAULT_REFRESH_INTERVAL);
  const [refreshKey, setRefreshKey] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const TIME_RANGES_OPTIONS = useMemo(
    () => TIME_RANGES.map((item) => ({ label: t(item.labelKey), value: item.value })),
    [t]
  );
  const REFRESH_OPTIONS = useMemo(
    () =>
      REFRESH.map((item) => ({
        prefix: <RotateCw className="size-4" />,
        label: t(item.labelKey),
        value: item.value,
        refreshInterval: item.refreshInterval,
      })),
    [t]
  );
  const dashboardUrl = useMemo(() => {
    const {
      grafana_url,
      grafana_dashboard_uid,
      grafana_datasource,
      grafana_alert_datasource,
      cluster_name,
    } = clusterUIConfig;
    if (!grafana_url) {
      return '';
    }
    const url = joinGrafanaPath(grafana_url, `/d/${grafana_dashboard_uid}`);
    // Using url.searchParams.set() would generate "?orgId=1&kiosk=&theme=light", but we need "?orgId=1&kiosk&theme=light" (without the extra "=" after kiosk).
    const queryParams = [
      buildQueryParam('orgId', '1'),
      'kiosk',
      buildQueryParam('theme', theme || 'light'),
      buildQueryParam('from', timeRange.from),
      buildQueryParam('to', timeRange.to),
    ];

    if (grafana_datasource) {
      queryParams.push(buildQueryParam('var-datasource', grafana_datasource));
    }

    if (grafana_alert_datasource) {
      queryParams.push(buildQueryParam('var-alert_datasource', grafana_alert_datasource));
    }

    if (cluster_name) {
      queryParams.push(buildQueryParam('var-cluster', cluster_name));
    }

    if (grafanaRefresh && grafanaRefresh !== 'off') {
      queryParams.push(buildQueryParam('refresh', grafanaRefresh));
    }

    queryParams.push(buildQueryParam('_t', String(refreshKey)));

    return `${url}?${queryParams.join('&')}`;
  }, [clusterUIConfig, theme, timeRange, grafanaRefresh, refreshKey]);

  const handleTimeRange = (value: TimeRangeValue) => {
    setTimeRange(value);
    setRefreshKey((current) => current + 1);
  };
  const handleRefreshValue = (value?: string) => {
    const item = REFRESH_OPTIONS.find((item) => item.value === value);
    if (!item) return;

    setGrafanaRefresh(item.value);
    setRefreshInterval(item.refreshInterval);
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('monitoring_grafana_refresh', item.value);
      sessionStorage.setItem('monitoring_refresh_interval', String(item.refreshInterval));
    }
    setRefreshKey((current) => current + 1);
  };

  useEffect(() => {
    const savedGrafanaRefresh = sessionStorage.getItem('monitoring_grafana_refresh');
    const savedRefreshInterval = sessionStorage.getItem('monitoring_refresh_interval');

    if (savedGrafanaRefresh) {
      setGrafanaRefresh(savedGrafanaRefresh);
    }

    if (savedRefreshInterval) {
      setRefreshInterval(Number(savedRefreshInterval));
    }
  }, []);

  // Auto refresh
  useEffect(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (refreshInterval > 0) {
      timerRef.current = setInterval(() => setRefreshKey((k) => k + 1), refreshInterval);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [refreshInterval]);

  if (!globalReady) {
    return <PageContainer loading />;
  }

  if (!clusterUIConfig?.grafana_url) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground font-medium">
        {t('monitorCenter.notConfigured')}
      </div>
    );
  }

  return (
    <PageContainer
      title={t('menu.monitorCenter')}
      extraContent={
        <div className="flex items-center justify-between gap-3">
          <TimeRangePicker
            options={TIME_RANGES_OPTIONS}
            value={timeRange}
            onChange={handleTimeRange}
          />
          <Button
            variant="outline"
            size="icon"
            aria-label={t('monitorCenter.manualRefresh')}
            onClick={() => setRefreshKey((current) => current + 1)}
          >
            <RotateCw className="size-4" />
          </Button>
          <Select
            options={REFRESH_OPTIONS}
            className="w-32"
            value={grafanaRefresh}
            onChange={handleRefreshValue}
            allowClear={false}
          />
        </div>
      }
    >
      <div className="h-[calc(100vh-8rem)] min-h-[30rem]">
        <iframe
          key={refreshKey}
          src={dashboardUrl}
          className="h-full w-full rounded-md border"
          title="Grafana Monitoring"
        />
      </div>
    </PageContainer>
  );
};
export default MonitorCenter;
