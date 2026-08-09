export const LOG_FONT_SIZE_CLASS = 'text-xs';
export const LOG_PAGE_SIZE = 200;

export const DEFAULT_LOG_TIME_RANGE = {
  from: 'now-1h',
  to: 'now',
};

export const LOG_LEVELS = ['ERROR', 'WARNING', 'INFO', 'DEBUG'] as const;
export const LOG_TYPES = ['worker', 'supervisor'] as const;

export const LOG_TIME_RANGES = [
  { labelKey: 'monitorCenter.time.15m', from: 'now-15m', to: 'now' },
  { labelKey: 'monitorCenter.time.1h', from: 'now-1h', to: 'now' },
  { labelKey: 'monitorCenter.time.6h', from: 'now-6h', to: 'now' },
  { labelKey: 'monitorCenter.time.24h', from: 'now-24h', to: 'now' },
  { labelKey: 'monitorCenter.time.2d', from: 'now-2d', to: 'now' },
  { labelKey: 'monitorCenter.time.7d', from: 'now-7d', to: 'now' },
] as const;

export const LOG_REFRESH_OPTIONS = [
  { labelKey: 'monitorCenter.refresh.off', value: 0 },
  { labelKey: 'monitorCenter.refresh.10s', value: 10000 },
  { labelKey: 'monitorCenter.refresh.30s', value: 30000 },
  { labelKey: 'monitorCenter.refresh.1m', value: 60000 },
  { labelKey: 'monitorCenter.refresh.5m', value: 300000 },
] as const;

export const LOG_LEVEL_TEXT_CLASSES: Record<string, string> = {
  ERROR: 'text-destructive',
  WARNING: 'text-amber-600 dark:text-amber-400',
  DEBUG: 'text-muted-foreground',
};
