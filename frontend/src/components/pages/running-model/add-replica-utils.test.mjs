import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildCustomReplicaConfigs,
  buildReplicaConfigs,
  filterWorkerOptions,
} from './add-replica-utils.mjs';

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

const customConfig = (rows, overrides = {}) =>
  buildCustomReplicaConfigs({
    rows,
    replicaCount: rows.length,
    workerAddresses: ['a:9978', 'b:9978'],
    device: 'GPU',
    ...overrides,
  });

test('maps independent Worker-local GPU lists and aliases without equal splitting', () => {
  const result = customConfig([
    { worker_ip: 'a:9978', gpu_idx: '0, 1', replica_uid: ' new-a ' },
    { worker_ip: 'b:9978', gpu_idx: '0,1' },
    { worker_ip: 'b:9978', gpu_idx: '4,5,6,7', replica_uid: ' ' },
  ]);
  assert.deepEqual(result.configs, [
    { replica_uid: 'new-a', devices: [{ worker_ip: 'a:9978', n_gpu: 2, gpu_idx: [0, 1] }] },
    { devices: [{ worker_ip: 'b:9978', n_gpu: 2, gpu_idx: [0, 1] }] },
    { devices: [{ worker_ip: 'b:9978', n_gpu: 4, gpu_idx: [4, 5, 6, 7] }] },
  ]);
});

test('defers same-Worker sharing policy to the server', () => {
  assert.ok(
    customConfig([
      { worker_ip: 'a:9978', gpu_idx: '0' },
      { worker_ip: 'a:9978', gpu_idx: '0' },
    ]).configs
  );
});

for (const device of ['auto', 'GPU', 'CPU']) {
  test(`uses explicit ${device} allocation semantics without inherited GPU metadata`, () => {
    const result = customConfig([{ worker_ip: 'a:9978', gpu_idx: ' ', n_gpu: 4 }], { device });
    assert.deepEqual(result.configs, [
      { devices: [{ worker_ip: 'a:9978', n_gpu: device === 'CPU' ? 0 : 'auto' }] },
    ]);
  });
}

test('CPU placement never sends stale GPU indexes', () => {
  assert.deepEqual(
    customConfig([{ worker_ip: 'a:9978', gpu_idx: '0,1' }], { device: 'CPU' }).configs,
    [{ devices: [{ worker_ip: 'a:9978', n_gpu: 0 }] }]
  );
});

for (const gpu_idx of ['-1', '1.5', '1,,2', '1,', 'x', '1e2', '9007199254740992']) {
  test(`rejects GPU input ${gpu_idx} without silently filtering values`, () => {
    const result = customConfig([{ worker_ip: 'a:9978', gpu_idx }]);
    assert.deepEqual(result.error, { code: 'gpu', row: 1 });
    assert.equal(result.configs, undefined);
  });
}

test('identifies the invalid row for duplicate indexes and aliases', () => {
  assert.deepEqual(
    customConfig([
      { worker_ip: 'a:9978', gpu_idx: '0' },
      { worker_ip: 'b:9978', gpu_idx: '1,01' },
    ]).error,
    { code: 'duplicateGpu', row: 2 }
  );
  assert.deepEqual(
    customConfig([
      { worker_ip: 'a:9978', gpu_idx: '', replica_uid: 'same' },
      { worker_ip: 'b:9978', gpu_idx: '', replica_uid: ' same ' },
    ]).error,
    { code: 'duplicateAlias', row: 2 }
  );
});

test('rejects incomplete, unavailable or mismatched rows instead of dropping them', () => {
  for (const worker_ip of ['', 'missing:9978']) {
    assert.deepEqual(customConfig([{ worker_ip, gpu_idx: '' }]).error, { code: 'worker', row: 1 });
  }
  assert.equal(
    customConfig([{ worker_ip: 'a:9978', gpu_idx: '' }], { replicaCount: 2 }).error.code,
    'count'
  );
  for (const replicaCount of [0, -1, 1.5, Infinity]) {
    assert.equal(customConfig([], { replicaCount }).error.code, 'count');
  }
});
