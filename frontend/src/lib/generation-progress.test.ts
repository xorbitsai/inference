import assert from 'node:assert/strict';
import test from 'node:test';

import { formatProgressDuration, normalizeProgressTiming } from './generation-progress';

test('generation duration formatting and unknown timing are handled', () => {
  assert.equal(formatProgressDuration(3661.9), '01:01:01');
  assert.equal(formatProgressDuration(undefined), '—');
  assert.equal(normalizeProgressTiming({ eta_seconds: null }).eta_seconds, null);
  assert.equal(normalizeProgressTiming({ eta_seconds: Infinity }).eta_seconds, null);
  assert.equal(normalizeProgressTiming({ elapsed_seconds: -1 }).elapsed_seconds, undefined);
});
