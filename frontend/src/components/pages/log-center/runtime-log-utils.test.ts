import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildHistoricalHandoffQueryState,
  filterRuntimeLogEntries,
  getNodeDisplayName,
  getRuntimeHistoricalSearch,
  getRuntimeLogSourceColorIndex,
  mergeRuntimeLogEntries,
  normalizeRuntimeLogSource,
  parseRuntimeLogEntries,
  parseRuntimeLogPayload,
  runtimeLogSourceSearchText,
  tailLogLines,
  tailRuntimeLogEntries,
  trimRuntimeLogBuffer,
} from './runtime-log-utils';
import {
  formatLogTime,
  formatLogTimeTitle,
  getLogNodeFilterValue,
  getLogNodeName,
  getLogNodeRole,
  inferNodeRoleFromName,
  normalizeLogNodeRole,
  resolveHistoricalNodeRole,
} from './utils';

test('normalizeRuntimeLogSource uses structured role and node name', () => {
  assert.deepEqual(
    normalizeRuntimeLogSource({
      id: 'supervisor',
      label: 'Supervisor',
      role: 'supervisor',
      node_name: 'xinference-supervisor:9999',
    }),
    {
      id: 'supervisor',
      label: 'Supervisor',
      role: 'supervisor',
      nodeName: 'xinference-supervisor:9999',
      displayNodeName: 'xinference-supervisor:9999',
    }
  );
});

test('normalizeRuntimeLogSource remains compatible with legacy source responses', () => {
  assert.deepEqual(normalizeRuntimeLogSource({ id: 'supervisor', label: 'Supervisor' }), {
    id: 'supervisor',
    label: 'Supervisor',
    role: 'supervisor',
    nodeName: 'Supervisor',
    displayNodeName: 'Supervisor',
  });
  assert.deepEqual(normalizeRuntimeLogSource({ id: 'local', label: 'Local' }), {
    id: 'local',
    label: 'Local',
    role: 'local',
    nodeName: 'Local',
    displayNodeName: 'Local',
  });
  assert.deepEqual(
    normalizeRuntimeLogSource({
      id: 'xinference-worker-4090-2:30001',
      label: 'Worker xinference-worker-4090-2:30001',
    }),
    {
      id: 'xinference-worker-4090-2:30001',
      label: 'Worker xinference-worker-4090-2:30001',
      role: 'worker',
      nodeName: 'xinference-worker-4090-2:30001',
      displayNodeName: 'xinference-worker-4090-2:30001',
    }
  );
});

test('getNodeDisplayName preserves the complete Xinference address', () => {
  assert.equal(getNodeDisplayName(' xinference-supervisor:9999 '), 'xinference-supervisor:9999');
  assert.equal(
    getNodeDisplayName('xinference-worker-4090-2:30001'),
    'xinference-worker-4090-2:30001'
  );
  assert.equal(getNodeDisplayName('hostname-without-port'), 'hostname-without-port');
  assert.equal(getNodeDisplayName('[::1]:9999'), '[::1]:9999');
  assert.equal(getNodeDisplayName('2001:db8::1'), '2001:db8::1');
});

test('runtime log search matches source role labels and node names', () => {
  const source = normalizeRuntimeLogSource({
    id: 'supervisor',
    label: 'Supervisor',
    role: 'supervisor',
    node_name: 'xinference-supervisor:9999',
  });
  const entries = parseRuntimeLogEntries('2026-10-03T01:02:03.000Z INFO module ready\n', source.id);
  const searchText = {
    [source.id]: runtimeLogSourceSearchText(source, '主管节点'),
  };

  assert.equal(filterRuntimeLogEntries(entries, '主管节点', searchText).length, 1);
  assert.equal(
    filterRuntimeLogEntries(entries, 'xinference-supervisor:9999', searchText).length,
    1
  );
  assert.equal(filterRuntimeLogEntries(entries, 'xinference-supervisor', searchText).length, 1);
});

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

test('parseRuntimeLogEntries parses text, JSON, and multiline exceptions', () => {
  const text = [
    '2026-10-03T01:02:03.004Z INFO xinference.core pid:1 started',
    '2026-10-03T01:02:04.005Z ERROR xinference.worker pid:2 failed',
    'Traceback (most recent call last):',
    '  RuntimeError: boom',
    '',
  ].join('\n');
  const entries = parseRuntimeLogEntries(text, 'worker-a');
  assert.equal(entries.length, 2);
  assert.equal(entries[0].level, 'INFO');
  assert.equal(entries[0].message, 'xinference.core pid:1 started');
  assert.match(entries[1].message, /RuntimeError: boom/);

  const json = JSON.stringify({
    '@timestamp': '2026-10-03T01:02:05.006Z',
    level: 'WARNING',
    message: 'low memory',
  });
  const jsonEntry = parseRuntimeLogEntries(`${json}\n`, 'worker-b')[0];
  assert.equal(jsonEntry.level, 'WARNING');
  assert.equal(jsonEntry.message, json);
});

test('parseRuntimeLogEntries reparses a completed line after a chunk boundary', () => {
  const first = '{"@timestamp":"2026-10-03T01:02:03.004Z","level":"INFO"';
  assert.equal(parseRuntimeLogEntries(first, 'worker-a')[0].timestamp, '');
  const completed = `${first},"message":"ready"}\n`;
  assert.equal(
    parseRuntimeLogEntries(completed, 'worker-a')[0].timestamp,
    '2026-10-03T01:02:03.004Z'
  );
});

test('mergeRuntimeLogEntries sorts by timestamp and stable source order', () => {
  const workerA = parseRuntimeLogEntries(
    '2026-10-03T01:02:04.000Z INFO module worker-a\n',
    'worker-a'
  );
  const workerB = parseRuntimeLogEntries(
    [
      '2026-10-03T01:02:03.000Z INFO module earlier',
      '2026-10-03T01:02:04.000Z INFO module same-time',
      '',
    ].join('\n'),
    'worker-b'
  );
  const merged = mergeRuntimeLogEntries({ 'worker-a': workerA, 'worker-b': workerB }, [
    'worker-a',
    'worker-b',
  ]);
  assert.deepEqual(
    merged.map((entry) => entry.source),
    ['worker-b', 'worker-a', 'worker-b']
  );
});

test('tail and search preserve a complete multiline entry', () => {
  const entries = parseRuntimeLogEntries(
    [
      '2026-10-03T01:02:03.000Z INFO module old',
      '2026-10-03T01:02:04.000Z ERROR module failed',
      'Traceback:',
      '  needle',
      '',
    ].join('\n'),
    'worker-a'
  );
  const tailed = tailRuntimeLogEntries(entries, 2);
  assert.equal(tailed.length, 1);
  assert.match(tailed[0].raw, /Traceback/);
  assert.equal(filterRuntimeLogEntries(tailed, 'needle').length, 1);
  assert.equal(filterRuntimeLogEntries(tailed, 'worker-a').length, 1);
});

test('trimRuntimeLogBuffer starts at a complete log entry', () => {
  const logs = [
    '2026-10-03T01:02:03.000Z INFO module first',
    'continuation',
    '2026-10-03T01:02:04.000Z INFO module second',
    '',
  ].join('\n');
  const trimmed = trimRuntimeLogBuffer(logs, 70);
  assert.equal(trimmed, '2026-10-03T01:02:04.000Z INFO module second\n');
});

test('source colors and unified timestamp formatting are stable', () => {
  assert.equal(
    getRuntimeLogSourceColorIndex('worker-a', 6),
    getRuntimeLogSourceColorIndex('worker-a', 6)
  );
  assert.ok(getRuntimeLogSourceColorIndex('worker-a', 6) < 6);

  const timestamp = '2026-10-03T01:02:03.004Z';
  assert.match(formatLogTime(timestamp), /^\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.004$/);
  assert.match(formatLogTimeTitle(timestamp) || '', /2026-10-0[23].*· 2026-10-03T01:02:03\.004Z/);
  assert.equal(formatLogTime(''), '--');
  assert.equal(formatLogTimeTitle(''), undefined);
});

test('unified log node helpers prefer address and keep the filter field separate', () => {
  const row = {
    address: 'xinference-worker-4090-2:30001',
    node: 'legacy-node',
    node_name: 'legacy-node-name',
    node_role: 'worker',
  };

  assert.equal(getLogNodeName(row, 'node'), 'xinference-worker-4090-2:30001');
  assert.equal(getLogNodeFilterValue(row, 'node'), 'legacy-node');
  assert.equal(getLogNodeRole(row), 'worker');
});

test('unified log node helpers retain legacy fallback order', () => {
  assert.equal(
    getLogNodeName({ node_name: 'xinference-worker-4090-2:30001', node: 'legacy-node' }),
    'xinference-worker-4090-2:30001'
  );
  assert.equal(getLogNodeName({ node: 'legacy-node' }), 'legacy-node');
  assert.equal(getLogNodeName({ address: '', node_name: '', node: 'legacy-node' }), 'legacy-node');
  assert.equal(getLogNodeName({}), '-');
  assert.equal(
    getLogNodeFilterValue({ address: 'xinference-supervisor:9999' }, 'address.keyword'),
    'xinference-supervisor:9999'
  );
  assert.equal(getLogNodeFilterValue({ address: 'xinference-supervisor:9999' }, 'node'), '');
});

test('unified log node helpers preserve configured node fields and infer legacy roles', () => {
  assert.equal(
    getLogNodeName({ 'node.keyword': 'xinference-supervisor:9999' }, 'node.keyword'),
    'xinference-supervisor:9999'
  );
  assert.equal(inferNodeRoleFromName('xinference-supervisor:9999'), 'supervisor');
  assert.equal(inferNodeRoleFromName('xinference-worker-4090-2:30001'), 'worker');
  assert.equal(inferNodeRoleFromName('custom-node:1234'), '');
});

test('log node role normalization accepts known values without guessing unknown roles', () => {
  assert.equal(normalizeLogNodeRole('xinference-worker'), 'worker');
  assert.equal(normalizeLogNodeRole('SUPERVISOR'), 'supervisor');
  assert.equal(normalizeLogNodeRole('local_log'), 'local');
  assert.equal(normalizeLogNodeRole('unknown'), 'unknown');
  assert.equal(normalizeLogNodeRole('scheduler'), '');
  assert.equal(getLogNodeRole({ node_role: 'scheduler', role: 'worker' }), 'worker');
});

test('historical node role resolution follows the structured fallback order', () => {
  assert.equal(
    resolveHistoricalNodeRole({
      nodeName: 'xinference-supervisor',
      historicalRole: 'worker',
      resultRole: 'supervisor',
      runtimeRole: 'local',
    }),
    'worker'
  );
  assert.equal(
    resolveHistoricalNodeRole({
      nodeName: 'p-gpu-s4090-002',
      resultRole: 'worker',
      runtimeRole: 'supervisor',
    }),
    'worker'
  );
  assert.equal(
    resolveHistoricalNodeRole({
      nodeName: 'runtime-worker',
      runtimeRole: 'worker',
    }),
    'worker'
  );
  assert.equal(resolveHistoricalNodeRole({ nodeName: 'xinference-supervisor' }), 'supervisor');
  assert.equal(resolveHistoricalNodeRole({ nodeName: 'p-gpu-s4090-002' }), 'unknown');
});

test('parseRuntimeLogPayload exposes structured request metadata when available', () => {
  assert.deepEqual(parseRuntimeLogPayload('{"request_id":"request-1","message":"failed"}'), {
    request_id: 'request-1',
    message: 'failed',
  });
  assert.equal(parseRuntimeLogPayload('plain text log'), undefined);
  assert.equal(parseRuntimeLogPayload('["not", "an", "object"]'), undefined);
});

test('historical handoff replaces filters that are unrelated to the runtime record', () => {
  const previous = {
    searchText: 'old query',
    appliedSearch: 'old query',
    selectedLevels: ['ERROR'],
    selectedLogType: 'worker',
    selectedNodes: ['old-worker'],
    pageFrom: 200,
    fieldFilters: [{ key: 'module', value: 'old.module', op: '+' as const }],
  };

  const next = {
    ...previous,
    ...buildHistoricalHandoffQueryState({
      token: 1,
      nodeName: 'xinference-supervisor:9999',
      level: 'WARN',
      requestId: 'xinf-123',
    }),
  };

  assert.deepEqual(next, {
    searchText: 'xinf-123',
    appliedSearch: 'xinf-123',
    selectedLevels: ['WARNING'],
    selectedLogType: '',
    selectedNodes: [],
    pageFrom: 0,
    fieldFilters: [],
  });
});

test('runtime historical search uses the structured message without a request ID', () => {
  const raw = JSON.stringify({
    '@timestamp': '2026-10-05T08:00:00.000Z',
    level: 'INFO',
    module: 'xinference.core.worker',
    pid: 123,
    node: 'worker-host',
    message: 'Model loaded successfully',
  });
  const [entry] = parseRuntimeLogEntries(`${raw}\n`, 'worker-a');

  assert.deepEqual(getRuntimeHistoricalSearch(entry), {
    query: 'Model loaded successfully',
  });
});

test('runtime historical search keeps request IDs and plain-text message fallbacks', () => {
  const structured = JSON.stringify({
    '@timestamp': '2026-10-05T08:00:00.000Z',
    level: 'INFO',
    request_id: ' xinf-456 ',
    message: 'ignored when request ID is available',
  });
  const [structuredEntry] = parseRuntimeLogEntries(`${structured}\n`, 'worker-a');
  const [plainEntry] = parseRuntimeLogEntries(
    '2026-10-05T08:00:00.000Z INFO Model ready on worker\ncontinuation\n',
    'worker-a'
  );

  assert.deepEqual(getRuntimeHistoricalSearch(structuredEntry), { requestId: 'xinf-456' });
  assert.deepEqual(getRuntimeHistoricalSearch(plainEntry), { query: 'Model ready on worker' });
});
