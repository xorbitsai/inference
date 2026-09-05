/**
 * @typedef {{
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
