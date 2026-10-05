/* eslint-disable @typescript-eslint/no-require-imports -- Tests replace component modules before import. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import React, { act, useState } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';

import type { HistoricalLogHandoff, TimeRangeValue } from './types';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  Element: dom.window.Element,
  HTMLElement: dom.window.HTMLElement,
  Node: dom.window.Node,
  Event: dom.window.Event,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });

const toolbarModule = require('./log-toolbar') as typeof import('./log-toolbar');
const tableModule = require('./log-table') as typeof import('./log-table');
const filterBarModule = require('./filter-chip-bar') as typeof import('./filter-chip-bar');
const paginationModule = require('./pagination') as typeof import('./pagination');
const i18nModule = require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');

let toolbarProps: React.ComponentProps<typeof toolbarModule.LogToolbar>;
let tableProps: React.ComponentProps<typeof tableModule.LogTable>;

require.cache[require.resolve('./log-toolbar')]!.exports = {
  ...toolbarModule,
  LogToolbar: (props: typeof toolbarProps) => {
    toolbarProps = props;
    return null;
  },
};
require.cache[require.resolve('./log-table')]!.exports = {
  ...tableModule,
  LogTable: (props: typeof tableProps) => {
    tableProps = props;
    return null;
  },
};
require.cache[require.resolve('./filter-chip-bar')]!.exports = {
  ...filterBarModule,
  FilterChipBar: () => null,
};
require.cache[require.resolve('./pagination')]!.exports = {
  ...paginationModule,
  LogPagination: () => null,
};
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  ...i18nModule,
  useI18n: () => ({ t: (key: string) => key }),
};

const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const ElasticLogs = require('./elastic-logs').default as typeof import('./elastic-logs').default;
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');

let root: Root;
let requests: string[];
const originalGet = request.get;

function Harness({ handoff }: { handoff?: HistoricalLogHandoff }) {
  const [timeRange, setTimeRange] = useState<TimeRangeValue>({ from: 'now-1h', to: 'now' });
  return (
    <ElasticLogs
      active
      enabled
      handoff={handoff}
      timeRange={timeRange}
      onTimeRangeChange={setTimeRange}
      refreshInterval={0}
      onRefreshIntervalChange={() => {}}
      onViewRuntimeNode={() => {}}
    />
  );
}

async function settle(delay = 0) {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, delay));
  });
}

beforeEach(async () => {
  requests = [];
  request.get = (async (url: string) => {
    requests.push(url);
    if (url === '/v1/cluster/logs/nodes') {
      return {
        nodes: ['xinference-supervisor:9999'],
        node_field: 'address',
        node_roles: { 'xinference-supervisor:9999': 'supervisor' },
      };
    }
    if (url === '/v1/cluster/runtime-logs/sources') return { sources: [] };
    if (url.startsWith('/v1/cluster/logs?')) return { hits: [], total: 0 };
    throw new Error(`Unexpected request: ${url}`);
  }) as typeof request.get;

  const container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<Harness />));
  await settle();
});

afterEach(async () => {
  await act(async () => root.unmount());
  request.get = originalGet;
  document.body.replaceChildren();
});

test('runtime handoff clears old historical filters and cancels a pending search', async () => {
  await act(async () => {
    toolbarProps.onSelectedLogTypeChange?.('worker');
    toolbarProps.onSearchTextChange('stale pending query');
    tableProps.onFieldFilter?.('module', 'old.module', '+');
  });
  await settle();

  assert.ok(
    requests.some((url) => {
      if (!url.startsWith('/v1/cluster/logs?')) return false;
      const params = new URL(url, 'http://localhost').searchParams;
      return (
        params.get('log_type') === 'worker' &&
        params.getAll('filters').includes('+module:old.module')
      );
    }),
    'precondition: the previous historical filters should have been applied'
  );

  await act(async () => {
    root.render(
      <Harness
        handoff={{
          token: 1,
          nodeName: 'xinference-supervisor:9999',
          timestamp: '2026-10-05T08:00:00.000Z',
          level: 'INFO',
          requestId: 'xinf-123',
        }}
      />
    );
  });
  await settle(600);

  const latest = [...requests].reverse().find((url) => url.startsWith('/v1/cluster/logs?'));
  assert.ok(latest);
  const params = new URL(latest, 'http://localhost').searchParams;
  assert.equal(params.get('q'), 'xinf-123');
  assert.equal(params.get('log_type'), null);
  assert.deepEqual(params.getAll('filters'), []);
  assert.equal(params.get('node'), 'xinference-supervisor:9999');
});
