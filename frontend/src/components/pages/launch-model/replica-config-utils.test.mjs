import assert from 'node:assert/strict';
import test from 'node:test';
import {
  transformReplicaConfigToFormRows,
  transformReplicaFormRowsToConfig,
} from './replica-config-utils.mjs';

const roundTrip = (replicaConfig) =>
  transformReplicaFormRowsToConfig(transformReplicaConfigToFormRows(replicaConfig));

test('preserves numeric n_gpu when gpu_idx is omitted', () => {
  const replicaConfig = [
    {
      replica_uid: 'primary',
      devices: [{ worker_ip: 'worker-0:9978', n_gpu: 1 }],
    },
  ];

  const result = roundTrip(replicaConfig);

  assert.equal(result[0].devices[0].n_gpu, 1);
  assert.equal(result[0].devices[0].gpu_idx, undefined);
});

test('preserves auto n_gpu when gpu_idx is omitted', () => {
  const replicaConfig = [
    {
      replica_uid: 'primary',
      devices: [{ worker_ip: 'worker-0:9978', n_gpu: 'auto' }],
    },
  ];

  const result = roundTrip(replicaConfig);

  assert.equal(result[0].devices[0].n_gpu, 'auto');
  assert.equal(result[0].devices[0].gpu_idx, undefined);
});

test('derives n_gpu from explicit gpu_idx instead of retained metadata', () => {
  const result = transformReplicaFormRowsToConfig([
    {
      replica_uid: 'primary',
      worker_ip: 'worker-0:9978',
      gpu_idx: '0, 2',
      n_gpu: 1,
    },
  ]);

  assert.equal(result[0].devices[0].n_gpu, 2);
  assert.deepEqual(result[0].devices[0].gpu_idx, [0, 2]);
});
