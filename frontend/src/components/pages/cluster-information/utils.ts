export const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export function usagePercent(used: unknown, total: unknown): number | null {
  return finite(used) && finite(total) && total > 0
    ? Math.max(0, Math.min(100, (used / total) * 100))
    : null;
}

export function availableUsagePercent(total: unknown, available: unknown): number | null {
  return finite(total) && finite(available) ? usagePercent(total - available, total) : null;
}
