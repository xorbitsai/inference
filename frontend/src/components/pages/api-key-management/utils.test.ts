import assert from 'node:assert/strict';
import test from 'node:test';

import { getApiKeyDisplayName, type ApiKey } from './utils';

const makeKey = (id: number, name: string | null) => ({ id, name }) as ApiKey;

test('uses a trimmed API key name as the display label', () => {
  assert.equal(getApiKeyDisplayName(makeKey(7, ' Production ')), 'Production');
});

test('falls back to a non-secret identifier for missing names', () => {
  assert.equal(getApiKeyDisplayName(makeKey(8, null)), 'api-key-8');
  assert.equal(getApiKeyDisplayName(makeKey(9, '   ')), 'api-key-9');
});
