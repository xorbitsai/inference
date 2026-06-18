export const TIME_RANGES = [
  { labelKey: 'monitorCenter.time.5m', value: 'now-5m' },
  { labelKey: 'monitorCenter.time.15m', value: 'now-15m' },
  { labelKey: 'monitorCenter.time.30m', value: 'now-30m' },
  { labelKey: 'monitorCenter.time.1h', value: 'now-1h' },
  { labelKey: 'monitorCenter.time.3h', value: 'now-3h' },
  { labelKey: 'monitorCenter.time.6h', value: 'now-6h' },
  { labelKey: 'monitorCenter.time.12h', value: 'now-12h' },
  { labelKey: 'monitorCenter.time.24h', value: 'now-24h' },
  { labelKey: 'monitorCenter.time.2d', value: 'now-2d' },
  { labelKey: 'monitorCenter.time.7d', value: 'now-7d' },
] as const;

export const REFRESH = [
  { labelKey: 'monitorCenter.refresh.off', refreshInterval: 0, value: 'off' },
  { labelKey: 'monitorCenter.refresh.10s', refreshInterval: 10000, value: '10s' },
  { labelKey: 'monitorCenter.refresh.30s', refreshInterval: 30000, value: '30s' },
  { labelKey: 'monitorCenter.refresh.1m', refreshInterval: 60000, value: '1m' },
  { labelKey: 'monitorCenter.refresh.5m', refreshInterval: 300000, value: '5m' },
]