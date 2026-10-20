export interface GenerationProgressTiming {
  elapsed_seconds?: number;
  eta_seconds: number | null;
}

export function normalizeProgressTiming(value: unknown): GenerationProgressTiming {
  const data = (value && typeof value === 'object' ? value : {}) as Record<string, unknown>;
  const seconds = (value: unknown): number | undefined =>
    typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
  return {
    elapsed_seconds: seconds(data.elapsed_seconds),
    eta_seconds: seconds(data.eta_seconds) ?? null,
  };
}

export function formatProgressDuration(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.floor(seconds);
  return [Math.floor(total / 3600), Math.floor((total % 3600) / 60), total % 60]
    .map((part) => String(part).padStart(2, '0'))
    .join(':');
}
