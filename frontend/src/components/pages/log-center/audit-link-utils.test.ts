import assert from 'node:assert/strict';
import test from 'node:test';

import { buildAuditCenterHref } from './audit-link-utils';

test('buildAuditCenterHref includes an encoded request ID and a 24-hour window', () => {
  const href = buildAuditCenterHref('request/id with spaces', '2026-09-30T08:07:22.341Z');
  const url = new URL(href, 'https://xinference.example');

  assert.equal(url.pathname, '/audit-center');
  assert.equal(url.searchParams.get('request_id'), 'request/id with spaces');
  assert.equal(url.searchParams.get('time_from'), '2026-09-29T08:07:22.341Z');
  assert.equal(url.searchParams.get('time_to'), '2026-10-01T08:07:22.341Z');
});

test('buildAuditCenterHref omits time bounds for an invalid anchor', () => {
  const href = buildAuditCenterHref('external-request-id', 'not-a-date');
  const url = new URL(href, 'https://xinference.example');

  assert.equal(url.searchParams.get('request_id'), 'external-request-id');
  assert.equal(url.searchParams.has('time_from'), false);
  assert.equal(url.searchParams.has('time_to'), false);
});
