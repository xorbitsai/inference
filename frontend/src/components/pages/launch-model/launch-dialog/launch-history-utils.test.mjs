import assert from 'node:assert/strict';
import test from 'node:test';
import {
  getLaunchHistoryItemKey,
  getLaunchHistoryStorageKey,
  getModelLaunchHistory,
  mergeLaunchHistories,
  migrateLegacyLaunchHistory,
  normalizeLaunchHistory,
  normalizeTimestamp,
  selectLatestModelLaunchHistory,
} from './launch-history-utils.mjs';

const item = (overrides = {}) => ({
  data: { model_name: 'llama' },
  model_name: 'llama',
  model_uid: 'uid-1',
  created_by: 'alice',
  updated_at: 100,
  autostart_enabled: false,
  source: 'server',
  pending_sync: false,
  is_owner: true,
  ...overrides,
});

test('normalizes numeric and ISO timestamps to milliseconds', () => {
  assert.equal(normalizeTimestamp(1234), 1234);
  assert.equal(normalizeTimestamp('2026-09-09T00:00:00Z'), 1788912000000);
});

test('rejects invalid timestamps and malformed records', () => {
  assert.equal(normalizeTimestamp('not-a-date'), null);
  assert.deepEqual(
    normalizeLaunchHistory([null, {}, item({ data: [] }), item({ updated_at: 'x' })]),
    []
  );
});

test('uses model, uid, and creator as the stable identity', () => {
  assert.equal(getLaunchHistoryItemKey(item()), 'llama::uid-1::alice');
  assert.equal(
    getLaunchHistoryItemKey(item({ model_uid: '', created_by: '' })),
    'llama::__default__::__anonymous__'
  );
});

test('server records replace local mirrors with the same identity', () => {
  const server = item({ data: { value: 'server' }, updated_at: 200 });
  const local = item({ data: { value: 'local' }, source: 'local', pending_sync: true });
  assert.deepEqual(mergeLaunchHistories([server], [local]), [server]);
});

test('newer pending local records replace stale server mirrors', () => {
  const server = item({ data: { value: 'server' }, updated_at: 100 });
  const local = item({
    data: { value: 'local' },
    updated_at: 200,
    source: 'local',
    pending_sync: true,
  });
  assert.deepEqual(mergeLaunchHistories([server], [local]), [local]);
});

test('an empty server result preserves pending local records only', () => {
  const pending = item({ source: 'local', pending_sync: true });
  const mirror = item({ model_uid: 'uid-2', pending_sync: false });
  assert.deepEqual(mergeLaunchHistories([], [pending, mirror]), [pending]);
});

test('same model uid from different creators remains separate', () => {
  const alice = item();
  const bob = item({ created_by: 'bob', is_owner: false });
  assert.equal(mergeLaunchHistories([alice, bob], []).length, 2);
});

test('filters records to the selected model and sorts newest first', () => {
  const result = getModelLaunchHistory(
    [
      item({ updated_at: 1 }),
      item({ model_name: 'qwen' }),
      item({ model_uid: 'uid-2', updated_at: 3 }),
    ],
    'llama'
  );
  assert.deepEqual(
    result.map((entry) => entry.model_uid),
    ['uid-2', 'uid-1']
  );
});

test('builds versioned user-scoped storage keys', () => {
  assert.equal(getLaunchHistoryStorageKey('anonymous'), 'xinference:launch-history:v2:anonymous');
  assert.equal(
    getLaunchHistoryStorageKey('alice@example.com'),
    'xinference:launch-history:v2:alice%40example.com'
  );
  assert.equal(getLaunchHistoryStorageKey(''), null);
});

test('migrates valid legacy records only in no-auth mode', () => {
  const legacy = [item({ created_by: undefined, source: undefined, pending_sync: undefined })];
  const migrated = migrateLegacyLaunchHistory(legacy, false);
  assert.equal(migrated.length, 1);
  assert.equal(migrated[0].created_by, '');
  assert.equal(migrated[0].source, 'local');
  assert.equal(migrated[0].pending_sync, true);
  assert.equal(migrated[0].is_owner, true);
});

test('does not migrate the unscoped legacy cache in authenticated mode', () => {
  assert.deepEqual(migrateLegacyLaunchHistory([item()], true), []);
});

test('normalizes backend autostart and ownership flags', () => {
  const [normalized] = normalizeLaunchHistory([
    item({ updated_at: '2026-09-09T00:00:00Z', autostart_enabled: true, is_owner: false }),
  ]);
  assert.equal(normalized.updated_at, 1788912000000);
  assert.equal(normalized.autostart_enabled, true);
  assert.equal(normalized.is_owner, false);
});

test('applies the latest refreshed model history only before the user edits', () => {
  const history = [
    item({ updated_at: 1 }),
    item({ model_uid: 'uid-2', updated_at: 3 }),
    item({ model_name: 'qwen', updated_at: 5 }),
  ];

  assert.equal(selectLatestModelLaunchHistory(history, 'llama', false)?.model_uid, 'uid-2');
  assert.equal(selectLatestModelLaunchHistory(history, 'llama', true), null);
});
