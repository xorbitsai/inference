import assert from 'node:assert/strict';
import test from 'node:test';

import { MAX_SEED, parseScalarSeed } from './seed-utils';

test('parseScalarSeed accepts random and explicit seeds', () => {
  assert.equal(parseScalarSeed(-1), -1);
  assert.equal(parseScalarSeed('11'), 11);
  assert.equal(parseScalarSeed(MAX_SEED), MAX_SEED);
  assert.equal(parseScalarSeed(''), undefined);
});

test('parseScalarSeed rejects invalid seeds', () => {
  assert.throws(() => parseScalarSeed(-2), /must be -1/);
  assert.throws(() => parseScalarSeed('1.5'), /must be -1/);
  assert.throws(() => parseScalarSeed(MAX_SEED + 1), /must be -1/);
});
