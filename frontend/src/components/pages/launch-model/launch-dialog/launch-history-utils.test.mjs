import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildLaunchTemplateData,
  getLaunchHistoryItemKey,
  getLaunchHistoryStorageKey,
  getModelLaunchHistory,
  getOtherModelLaunchHistory,
  mergeLaunchHistories,
  migrateLegacyLaunchHistory,
  normalizeLaunchHistory,
  normalizeTimestamp,
  selectLatestModelLaunchHistory,
} from './launch-history-utils.mjs';

const item = (overrides = {}) => ({
  data: { model_name: 'llama', model_type: 'LLM' },
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

test('merges server history with pending local records across models', () => {
  const server = item({ model_name: 'qwen', model_uid: 'server-qwen', updated_at: 3 });
  const pending = item({
    model_name: 'llama',
    model_uid: 'pending-llama',
    source: 'local',
    pending_sync: true,
    updated_at: 4,
  });
  const staleMirror = item({
    model_name: 'mistral',
    model_uid: 'stale-mistral',
    source: 'local',
    pending_sync: false,
    updated_at: 5,
  });

  assert.deepEqual(mergeLaunchHistories([server], [pending, staleMirror]), [pending, server]);
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

test('filters owned records from other models and sorts newest first', () => {
  const result = getOtherModelLaunchHistory(
    [
      item({ model_name: 'llama', updated_at: 5 }),
      item({ model_name: 'qwen', model_uid: 'qwen-old', updated_at: 2 }),
      item({ model_name: 'mistral', model_uid: 'mistral-new', updated_at: 4 }),
      item({ model_name: 'gemma', is_owner: false, updated_at: 6 }),
    ],
    'llama',
    'LLM'
  );

  assert.deepEqual(
    result.map((entry) => entry.model_uid),
    ['mistral-new', 'qwen-old']
  );
});

test('searches other model history by model name and model uid case-insensitively', () => {
  const history = [
    item({ model_name: 'Qwen3-Coder', model_uid: 'coder-1' }),
    item({ model_name: 'mistral', model_uid: 'PROD-UID' }),
  ];

  assert.deepEqual(
    getOtherModelLaunchHistory(history, 'llama', 'LLM', 'QWEN').map((entry) => entry.model_name),
    ['Qwen3-Coder']
  );
  assert.deepEqual(
    getOtherModelLaunchHistory(history, 'llama', 'LLM', 'prod-uid').map(
      (entry) => entry.model_name
    ),
    ['mistral']
  );
});

test('builds a safe cross-model template without mutating history data', () => {
  const historyData = {
    model_name: 'old-model',
    model_uid: 'old-uid',
    model_type: 'LLM',
    model_engine: 'vllm',
    n_gpu: 2,
    envs: { TOKEN: 'value' },
    id: 9,
    created_by: 'alice',
    updated_by: 'alice',
    created_at: '2026-09-01T00:00:00Z',
    updated_at: '2026-09-02T00:00:00Z',
    autostart_enabled: true,
    autostart_priority: 1,
    autostart_max_retries: 2,
    autostart_retry_interval_seconds: 3,
    is_owner: true,
    source: 'server',
    pending_sync: false,
  };
  const original = structuredClone(historyData);

  const template = buildLaunchTemplateData(historyData, 'current-model', 'LLM');

  assert.deepEqual(historyData, original);
  assert.equal(template.model_name, 'current-model');
  assert.equal(template.model_type, 'LLM');
  assert.equal('model_uid' in template, false);
  assert.equal(template.model_engine, 'vllm');
  assert.deepEqual(template.envs, { TOKEN: 'value' });
  assert.notEqual(template.envs, historyData.envs);
  for (const key of [
    'id',
    'created_by',
    'updated_by',
    'created_at',
    'updated_at',
    'autostart_enabled',
    'autostart_priority',
    'autostart_max_retries',
    'autostart_retry_interval_seconds',
    'is_owner',
    'source',
    'pending_sync',
  ]) {
    assert.equal(key in template, false, `${key} should not be copied to the launch template`);
  }
});

test('filters out history from a different model type', () => {
  const history = [
    item({ model_name: 'qwen', model_uid: 'qwen-llm' }),
    item({
      model_name: 'whisper',
      model_uid: 'whisper-audio',
      data: { model_name: 'whisper', model_type: 'audio' },
    }),
  ];

  assert.deepEqual(
    getOtherModelLaunchHistory(history, 'llama', 'LLM').map((entry) => entry.model_uid),
    ['qwen-llm']
  );
});

test('rejects invalid or incompatible cross-model template inputs', () => {
  assert.equal(buildLaunchTemplateData(null, 'llama', 'LLM'), null);
  assert.equal(buildLaunchTemplateData([], 'llama', 'LLM'), null);
  assert.equal(buildLaunchTemplateData({}, '', 'LLM'), null);
  assert.equal(
    buildLaunchTemplateData(
      { model_name: 'whisper', model_type: 'audio', model_path: '/audio/path' },
      'llama',
      'LLM'
    ),
    null
  );
});
