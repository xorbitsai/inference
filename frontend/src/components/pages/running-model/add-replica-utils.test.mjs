import assert from 'node:assert/strict';
import test from 'node:test';

import { buildReplicaConfigs, filterWorkerOptions } from './add-replica-utils.mjs';

const workerOptions = [
  { label: 'CPU Worker', value: 'cpu:9978', gpuCount: 0 },
  { label: 'GPU Worker', value: 'gpu:9978', gpuCount: 2 },
  { label: 'Fallback Worker', value: 'unknown:9978' },
];

test('filters out CPU-only workers for GPU replicas', () => {
  assert.deepEqual(
    filterWorkerOptions(workerOptions, 'GPU', 'CPU').map((option) => option.value),
    ['gpu:9978', 'unknown:9978']
  );
});

test('uses the current model device when device selection is automatic', () => {
  assert.deepEqual(
    filterWorkerOptions(workerOptions, 'auto', 'GPU').map((option) => option.value),
    ['gpu:9978', 'unknown:9978']
  );
  assert.equal(filterWorkerOptions(workerOptions, 'auto', 'CPU').length, 3);
});

test('keeps every worker available for CPU replicas', () => {
  assert.equal(filterWorkerOptions(workerOptions, 'CPU', 'GPU').length, 3);
});

test('assigns replicas to selected workers in order', () => {
  const configs = buildReplicaConfigs({
    replicaCount: 2,
    workerAddresses: ['worker-a:9978', 'worker-b:9978'],
    device: 'auto',
    gpuIndexes: [],
  });

  assert.deepEqual(
    configs.map((config) => config.devices[0]),
    [
      { worker_ip: 'worker-a:9978', n_gpu: 'auto' },
      { worker_ip: 'worker-b:9978', n_gpu: 'auto' },
    ]
  );
});

test('reuses selected workers round-robin for additional replicas', () => {
  const configs = buildReplicaConfigs({
    replicaCount: 5,
    workerAddresses: ['worker-a:9978', 'worker-b:9978'],
    device: 'CPU',
    gpuIndexes: [],
  });

  assert.deepEqual(
    configs.map((config) => config.devices[0].worker_ip),
    ['worker-a:9978', 'worker-b:9978', 'worker-a:9978', 'worker-b:9978', 'worker-a:9978']
  );
  assert.ok(configs.every((config) => config.devices[0].n_gpu === 0));
});

test('splits explicit GPU indexes evenly between replicas', () => {
  const configs = buildReplicaConfigs({
    replicaCount: 2,
    workerAddresses: ['worker-a:9978', 'worker-b:9978'],
    device: 'GPU',
    gpuIndexes: [0, 1, 2, 3],
  });

  assert.deepEqual(configs[0].devices[0], {
    worker_ip: 'worker-a:9978',
    n_gpu: 2,
    gpu_idx: [0, 1],
  });
  assert.deepEqual(configs[1].devices[0], {
    worker_ip: 'worker-b:9978',
    n_gpu: 2,
    gpu_idx: [2, 3],
  });
});
