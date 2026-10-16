export const LOG_FONT_SIZE_CLASS = 'text-xs';
export const LOG_PAGE_SIZE = 200;

export const DEFAULT_LOG_TIME_RANGE = {
  from: 'now-1h',
  to: 'now',
};

export const LOG_LEVELS = ['ERROR', 'WARNING', 'INFO', 'DEBUG'] as const;
export const LOG_TYPES = ['worker', 'supervisor', 'model_request'] as const;

export const LOG_TIME_RANGES = [
  { labelKey: 'monitorCenter.time.15m', from: 'now-15m', to: 'now' },
  { labelKey: 'monitorCenter.time.1h', from: 'now-1h', to: 'now' },
  { labelKey: 'monitorCenter.time.6h', from: 'now-6h', to: 'now' },
  { labelKey: 'monitorCenter.time.24h', from: 'now-24h', to: 'now' },
  { labelKey: 'monitorCenter.time.2d', from: 'now-2d', to: 'now' },
  { labelKey: 'monitorCenter.time.7d', from: 'now-7d', to: 'now' },
] as const;

export const AUDIT_TIME_RANGES = [
  { labelKey: 'monitorCenter.time.15m', from: 'now-15m', to: 'now' },
  { labelKey: 'monitorCenter.time.30m', from: 'now-30m', to: 'now' },
  { labelKey: 'monitorCenter.time.1h', from: 'now-1h', to: 'now' },
  { labelKey: 'monitorCenter.time.6h', from: 'now-6h', to: 'now' },
  { labelKey: 'monitorCenter.time.12h', from: 'now-12h', to: 'now' },
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

export const LOG_LEVEL_BADGE_CLASSES: Record<string, string> = {
  ERROR:
    'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300',
  WARNING:
    'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-300',
  INFO: 'border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950/50 dark:text-blue-300',
  DEBUG:
    'border-violet-200 bg-violet-50 text-violet-700 dark:border-violet-900 dark:bg-violet-950/50 dark:text-violet-300',
  UNKNOWN:
    'border-slate-200 bg-slate-50 text-slate-700 dark:border-slate-800 dark:bg-slate-950/50 dark:text-slate-300',
};

export const NODE_ROLE_BADGE_CLASSES: Record<string, string> = {
  supervisor:
    'border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950/50 dark:text-blue-300',
  worker:
    'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300',
  local:
    'border-violet-200 bg-violet-50 text-violet-700 dark:border-violet-900 dark:bg-violet-950/50 dark:text-violet-300',
  unknown:
    'border-slate-200 bg-slate-50 text-slate-700 dark:border-slate-800 dark:bg-slate-950/50 dark:text-slate-300',
};
