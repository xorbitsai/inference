import assert from 'node:assert/strict';
import test from 'node:test';

import { getCorrelationId } from './correlation-utils';

test('prefers request_id over other correlation sources', () => {
  assert.equal(
    getCorrelationId({
      request_id: ' request-id ',
      correlation_id: 'correlation-id',
      message: '[request message-id] Enter call',
    }),
    'request-id'
  );
});

test('uses correlation_id when application logs do not have request_id', () => {
  assert.equal(getCorrelationId({ correlation_id: ' correlation-id ' }), 'correlation-id');
});

test('extracts only the fixed historical request marker from message', () => {
  assert.equal(
    getCorrelationId({ message: '[request historical-id] Enter get_model, args: []' }),
    'historical-id'
  );
  assert.equal(getCorrelationId({ message: 'actor id 0b359038-38eb-4361-ab5a-b82c0472685d' }), '');
  assert.equal(getCorrelationId({ message: '[Request wrong-case] Enter get_model' }), '');
});

test('rejects empty, oversized, and control-character IDs', () => {
  assert.equal(getCorrelationId({ request_id: '   ' }), '');
  assert.equal(getCorrelationId({ correlation_id: 'x'.repeat(257) }), '');
  assert.equal(getCorrelationId({ request_id: 'bad\nid' }), '');
});
