type VramSample = {
  gpu_count: number;
  gpus?: readonly { memory_total?: unknown; memory_available?: unknown }[];
  gpu_vram_total?: unknown;
  gpu_vram_available?: unknown;
};

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

/** Prefer per-device bytes; legacy aggregates are used only by older API payloads. */
export function vramSummary(
  workers: readonly VramSample[]
): { total: number; used: number } | null {
  if (workers.length === 0) return null;
  let total = 0;
  let used = 0;
  for (const worker of workers) {
    const devices = worker.gpus;
    if (devices !== undefined && (!Array.isArray(devices) || devices.length !== worker.gpu_count)) {
      return null;
    }
    const readings =
      devices === undefined
        ? worker.gpu_count === 0
          ? []
          : [
              {
                memory_total: worker.gpu_vram_total,
                memory_available: worker.gpu_vram_available,
              },
            ]
        : devices;
    for (const reading of readings) {
      if (
        !finite(reading.memory_total) ||
        reading.memory_total <= 0 ||
        !finite(reading.memory_available)
      ) {
        return null;
      }
      total += reading.memory_total;
      used += Math.max(0, reading.memory_total - reading.memory_available);
    }
  }
  return { total, used };
}
