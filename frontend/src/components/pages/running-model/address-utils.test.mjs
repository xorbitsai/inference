import assert from 'node:assert/strict';
import test from 'node:test';
import { formatReplicaRuntimeAddress, parseActorAddress } from './address-utils.mjs';

test('parseActorAddress handles xoscar-style unbracketed IPv6 addresses', () => {
  assert.deepEqual(parseActorAddress('2001:db8::1:42135'), {
    host: '2001:db8::1',
    port: '42135',
  });
  assert.deepEqual(parseActorAddress(':::42135'), {
    host: '::',
    port: '42135',
  });
});

test('parseActorAddress continues to handle bracketed IPv6 addresses', () => {
  assert.deepEqual(parseActorAddress('[2001:db8::1]:42135'), {
    host: '2001:db8::1',
    port: '42135',
  });
  assert.deepEqual(parseActorAddress('[2001:db8::1]'), {
    host: '2001:db8::1',
    port: undefined,
  });
});

test('formatReplicaRuntimeAddress preserves the model port for IPv6 workers', () => {
  assert.equal(
    formatReplicaRuntimeAddress({
      worker_address: '2001:db8::10:12345',
      model_address: '2001:db8::10:42135',
    }),
    '[2001:db8::10]:42135'
  );
  assert.equal(
    formatReplicaRuntimeAddress({
      worker_address: ':::12345',
      model_address: ':::42135',
    }),
    '[::]:42135'
  );
  assert.equal(
    formatReplicaRuntimeAddress({
      worker_address: '[2001:db8::10]:12345',
      model_address: '[2001:db8::10]:42135',
    }),
    '[2001:db8::10]:42135'
  );
});
