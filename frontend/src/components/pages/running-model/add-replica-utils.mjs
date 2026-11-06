/**
 * @typedef {{
 *   replica_uid?: string,
 *   devices: Array<{
 *     worker_ip: string,
 *     n_gpu: number | 'auto',
 *     gpu_idx?: number[],
 *   }>,
 * }} ReplicaConfig
 */

/**
 * @typedef {{
 *   label: string,
 *   value: string,
 *   description?: string,
 *   gpuCount?: number,
 * }} WorkerOption
 */

/**
 * Filter Workers using the effective device type. CPU replicas can run on any
 * Worker, while GPU replicas require a Worker with an advertised GPU. Workers
 * from fallback discovery have no GPU metadata, so keep them selectable.
 *
 * @param {WorkerOption[]} workerOptions
 * @param {'auto' | 'GPU' | 'CPU'} device
 * @param {'auto' | 'GPU' | 'CPU'} defaultDevice
 * @returns {WorkerOption[]}
 */
export function filterWorkerOptions(workerOptions, device, defaultDevice) {
  const effectiveDevice = device === 'auto' ? defaultDevice : device;
  if (effectiveDevice !== 'GPU') return workerOptions;

  return workerOptions.filter((option) => option.gpuCount === undefined || option.gpuCount > 0);
}

/**
 * Build one placement entry per new replica.
 *
 * Workers are assigned in selection order and reused round-robin when more
 * replicas than Workers are requested. GPU indexes keep the existing scale-up
 * behavior: the list is split evenly between replicas.
 *
 * @param {{
 *   replicaCount: number,
 *   workerAddresses: string[],
 *   device: 'auto' | 'GPU' | 'CPU',
 *   gpuIndexes: number[],
 * }} options
 * @returns {ReplicaConfig[] | undefined}
 */
export function buildReplicaConfigs({ replicaCount, workerAddresses, device, gpuIndexes }) {
  if (workerAddresses.length === 0) return undefined;

  const gpuCountPerReplica = gpuIndexes.length / replicaCount;
  return Array.from({ length: replicaCount }, (_, index) => {
    const assignedGpuIndexes = gpuIndexes.slice(
      index * gpuCountPerReplica,
      (index + 1) * gpuCountPerReplica
    );

    return {
      devices: [
        {
          worker_ip: workerAddresses[index % workerAddresses.length],
          n_gpu: device === 'CPU' ? 0 : device === 'GPU' ? assignedGpuIndexes.length || 1 : 'auto',
          ...(assignedGpuIndexes.length > 0 ? { gpu_idx: assignedGpuIndexes } : {}),
        },
      ],
    };
  });
}

/**
 * Validate every row before constructing any placement entry. GPU indexes are
 * local to a Worker; sharing/occupancy is checked by the server, not guessed here.
 * @param {{
 *   replicaCount: number,
 *   rows: Array<{worker_ip: string, gpu_idx: string, replica_uid?: string}>,
 *   workerAddresses: string[],
 *   device: 'auto' | 'GPU' | 'CPU',
 * }} options
 * @returns {{configs: ReplicaConfig[], error?: never} | {configs?: never, error: {
 *   code: 'count' | 'worker' | 'gpu' | 'duplicateGpu' | 'duplicateAlias', row: number,
 * }}}
 */
export function buildCustomReplicaConfigs({ replicaCount, rows, workerAddresses, device }) {
  if (!Number.isInteger(replicaCount) || replicaCount < 1 || rows.length !== replicaCount) {
    return { error: { code: 'count', row: 1 } };
  }
  const aliases = new Set();
  /** @type {ReplicaConfig[]} */
  const configs = [];
  for (const [index, row] of rows.entries()) {
    const rowNumber = index + 1;
    if (!row.worker_ip || !workerAddresses.includes(row.worker_ip)) {
      return { error: { code: 'worker', row: rowNumber } };
    }
    const gpuText = device === 'CPU' ? '' : row.gpu_idx.trim();
    if (gpuText && !/^\d+(?:\s*,\s*\d+)*$/.test(gpuText)) {
      return { error: { code: 'gpu', row: rowNumber } };
    }
    const indexes = gpuText ? gpuText.split(',').map(Number) : [];
    if (indexes.some((value) => !Number.isSafeInteger(value))) {
      return { error: { code: 'gpu', row: rowNumber } };
    }
    if (new Set(indexes).size !== indexes.length) {
      return { error: { code: 'duplicateGpu', row: rowNumber } };
    }
    const alias = row.replica_uid?.trim();
    if (alias && aliases.has(alias)) {
      return { error: { code: 'duplicateAlias', row: rowNumber } };
    }
    if (alias) aliases.add(alias);
    configs.push({
      ...(alias ? { replica_uid: alias } : {}),
      devices: [
        {
          worker_ip: row.worker_ip,
          n_gpu: device === 'CPU' ? 0 : indexes.length || 'auto',
          ...(indexes.length ? { gpu_idx: indexes } : {}),
        },
      ],
    });
  }
  return { configs };
}
