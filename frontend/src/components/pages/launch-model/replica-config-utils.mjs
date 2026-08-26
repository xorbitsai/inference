/**
 * @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

/**
 * @param {unknown} value
 * @returns {number[] | undefined}
 */
export const parseReplicaGpuIndexes = (value) => {
  if (typeof value !== 'string' || !value) return undefined;

  const result = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .map(Number)
    .filter((num) => !Number.isNaN(num));

  return result.length ? result : undefined;
};

/**
 * Convert API/CLI replica placement entries into the UI's flat row format.
 * `n_gpu` is retained as hidden row metadata so a numeric GPU count without
 * explicit GPU indexes survives a command import and subsequent launch.
 *
 * @param {unknown[]} replicaConfig
 * @returns {Array<{
 *   replica_uid: unknown,
 *   worker_ip: unknown,
 *   gpu_idx: string,
 *   n_gpu: unknown,
 * }>}
 */
export const transformReplicaConfigToFormRows = (replicaConfig) =>
  replicaConfig.map((entry) => {
    const device = isRecord(entry) && Array.isArray(entry.devices) ? entry.devices[0] : {};
    const deviceRecord = isRecord(device) ? device : {};

    return {
      replica_uid: (isRecord(entry) && entry.replica_uid) || '',
      worker_ip: deviceRecord.worker_ip || '',
      gpu_idx: Array.isArray(deviceRecord.gpu_idx) ? deviceRecord.gpu_idx.join(',') : '',
      n_gpu: deviceRecord.n_gpu,
    };
  });

/**
 * Convert the UI's flat replica placement rows back into API/CLI entries.
 * Explicit GPU indexes take precedence; otherwise the imported `n_gpu` value
 * is preserved, falling back to automatic allocation for newly-created rows.
 *
 * @param {unknown[]} rows
 * @returns {Array<{
 *   replica_uid: string | undefined,
 *   devices: Array<{
 *     worker_ip: unknown,
 *     n_gpu: number | 'auto',
 *     gpu_idx: number[] | undefined,
 *   }>,
 * }>}
 */
export const transformReplicaFormRowsToConfig = (rows) =>
  rows
    .filter((row) => isRecord(row) && row.worker_ip)
    .map((row) => {
      const gpuIdx = parseReplicaGpuIndexes(row.gpu_idx);
      const preservedNGPU =
        row.n_gpu === 'auto' ||
        (typeof row.n_gpu === 'number' && Number.isInteger(row.n_gpu) && row.n_gpu >= 0)
          ? row.n_gpu
          : 'auto';

      return {
        replica_uid:
          typeof row.replica_uid === 'string' ? row.replica_uid.trim() || undefined : undefined,
        devices: [
          {
            worker_ip: row.worker_ip,
            n_gpu: gpuIdx ? gpuIdx.length : preservedNGPU,
            gpu_idx: gpuIdx,
          },
        ],
      };
    });
