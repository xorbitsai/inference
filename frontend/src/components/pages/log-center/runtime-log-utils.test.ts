import assert from 'node:assert/strict';
import test from 'node:test';
import { tailLogLines } from './runtime-log-utils';

test('tailLogLines keeps the newest physical lines with or without a trailing newline', () => {
  assert.equal(tailLogLines('one\ntwo\nthree\n', 2), 'two\nthree\n');
  assert.equal(tailLogLines('one\ntwo\nthree', 2), 'two\nthree');
  assert.equal(tailLogLines('one\ntwo\n', 3), 'one\ntwo\n');
});

test('tailLogLines counts multiline errors and blank lines', () => {
  const logs = 'error\n  traceback\n\nnext\n';
  assert.equal(tailLogLines(logs, 3), '  traceback\n\nnext\n');
  assert.equal(tailLogLines(logs, 1), 'next\n');
  assert.equal(tailLogLines('\n', 1), '\n');
  assert.equal(tailLogLines('\nnext\n', 2), '\nnext\n');
});
