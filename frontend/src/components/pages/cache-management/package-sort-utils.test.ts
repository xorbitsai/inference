import assert from 'node:assert/strict';
import test from 'node:test';

import { sortPackages } from './package-sort-utils';

const packages = [
  { name: 'zeta', version: '1.0', size_bytes: 300 },
  { name: 'Alpha', version: '1.0', size_bytes: 100 },
  { name: 'beta', version: '1.0', size_bytes: 200 },
];

test('sorts package names alphabetically ascending by default configuration', () => {
  assert.deepEqual(
    sortPackages(packages, { key: 'name', direction: 'asc' }).map((item) => item.name),
    ['Alpha', 'beta', 'zeta']
  );
});

test('sorts package names and sizes in both directions', () => {
  assert.deepEqual(
    sortPackages(packages, { key: 'name', direction: 'desc' }).map((item) => item.name),
    ['zeta', 'beta', 'Alpha']
  );
  assert.deepEqual(
    sortPackages(packages, { key: 'size', direction: 'asc' }).map((item) => item.size_bytes),
    [100, 200, 300]
  );
  assert.deepEqual(
    sortPackages(packages, { key: 'size', direction: 'desc' }).map((item) => item.size_bytes),
    [300, 200, 100]
  );
});
