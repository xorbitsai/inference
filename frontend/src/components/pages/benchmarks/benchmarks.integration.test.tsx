/* eslint-disable @typescript-eslint/no-require-imports -- Initialize jsdom before React. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom declarations are not installed.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';

const dom = new JSDOM('<!doctype html><html><body></body></html>', {
  url: 'http://localhost/benchmarks?model=model-two',
});
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
  sessionStorage: dom.window.sessionStorage,
  Event: dom.window.Event,
  IS_REACT_ACT_ENVIRONMENT: true,
  ResizeObserver: class {
    constructor(
      private callback: (entries: { contentRect: { width: number; height: number } }[]) => void
    ) {}
    observe() {
      this.callback([{ contentRect: { width: 720, height: 256 } }]);
    }
    unobserve() {}
    disconnect() {}
  },
});
dom.window.HTMLElement.prototype.getBoundingClientRect = () => ({
  width: 720,
  height: 256,
  x: 0,
  y: 0,
  top: 0,
  left: 0,
  right: 720,
  bottom: 256,
  toJSON: () => ({}),
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
require('@/contexts/i18n-context');
const en = require('@/i18n/locales/en').default;
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({
    t: (key: string) => key.split('.').reduce((value, part) => value?.[part], en) ?? key,
  }),
};
let allowed = true;
require('@/hooks/use-menu-auth');
require.cache[require.resolve('@/hooks/use-menu-auth')]!.exports = {
  useMenuAuth: () => ({ canWriteModels: allowed, hasModelsRead: true }),
};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const Benchmarks = require('./index').default;
const originalGet = request.get;
const originalPost = request.post;
let host: HTMLDivElement;
let root: Root;
let calls: string[];
let posts: { url: string; data: unknown }[];
const metric = { mean: 20, p50: 18, p95: 30, p99: 35 };
const stage = {
  concurrency: 4,
  successful: 9,
  failed: 1,
  request_throughput: 3,
  input_throughput: 120,
  output_throughput: null as number | null,
  latency_ms: metric,
  ttft_ms: metric,
  tpot_ms: null as typeof metric | null,
  itl_ms: metric,
  prompt_tokens: metric,
  output_tokens: null,
};
const record = (id: string, state = 'completed') => ({
  id,
  name: `test-${id}`,
  created_at: '2026-10-29T01:00:00Z',
  state,
  progress: 20,
  config: {
    model_uid: 'model-two',
    input_tokens: 256,
    output_tokens: 128,
    requests: 100,
    warmup: 3,
    seed: 42,
  },
  snapshot: { hardware: [{ gpu_count: 1 }] },
  runner: { name: 'xinference', version: '1', metrics_version: '1' },
  results: [stage],
  logs: ['stage complete'],
  error: null,
});
let records = [record('one'), record('two')];
beforeEach(() => {
  allowed = true;
  calls = [];
  posts = [];
  records = [record('one'), record('two')];
  request.get = async <T,>(url: string) => {
    calls.push(url);
    if (url === '/v1/models')
      return {
        data: [
          { id: 'model-one', model_type: 'LLM', model_ability: ['chat'] },
          { id: 'model-two', model_type: 'LLM', model_ability: ['chat'] },
          { id: 'embedding', model_type: 'embedding' },
        ],
      } as T;
    return records as T;
  };
  request.post = async <T,>(url: string, data: unknown) => {
    posts.push({ url, data });
    if (url === '/v1/benchmarks') {
      const next = record('new', 'queued');
      records = [...records, next];
      return next as T;
    }
    records = records.map((row) => (row.id === 'new' ? { ...row, state: 'stopped' } : row));
    return records[2] as T;
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  request.get = originalGet;
  request.post = originalPost;
});

it('creates a server job from the model entry and compares saved measurements', async () => {
  await act(async () => root.render(<Benchmarks />));
  assert.equal(host.querySelector('select')?.value, 'model-two');
  assert.equal(host.querySelectorAll('option').length, 3);
  await act(async () => {
    for (const checkbox of host.querySelectorAll<HTMLInputElement>('input[type=checkbox]'))
      checkbox.click();
  });
  assert.ok(host.textContent?.includes('Compare (2)'));
  const tables = host.querySelectorAll('table');
  assert.equal(tables.length, 2);
  assert.ok(tables[1].textContent?.includes('9 / 1'));
  assert.ok(tables[1].textContent?.includes('20.00 / 30.00 / 35.00'));
  assert.ok(tables[1].textContent?.includes('—'));
  assert.ok(host.textContent?.includes('gpu_count'));
  await act(async () => {
    host
      .querySelector('form')!
      .dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
  });
  assert.deepEqual(posts[0], {
    url: '/v1/benchmarks',
    data: {
      model_uid: 'model-two',
      name: '',
      runner: 'xinference',
      input_tokens: 256,
      output_tokens: 128,
      concurrency: [1, 4, 16],
      requests: 100,
      warmup: 3,
      seed: 42,
    },
  });
  await act(async () =>
    (host.querySelector('button[aria-label="Stop"]') as HTMLButtonElement).click()
  );
  assert.equal(posts[1].url, '/v1/benchmarks/new/stop');
});

it('does not expose history or start requests without model-management permission', async () => {
  allowed = false;
  await act(async () => root.render(<Benchmarks />));
  assert.equal(host.querySelector('form'), null);
  assert.deepEqual(calls, []);
});

it('aligns numeric concurrency points and keeps missing metrics separate from zero', () => {
  const { buildChartData } = require('./chart-data') as typeof import('./chart-data');
  const first = {
    ...record('a'),
    results: [
      { ...stage, concurrency: 16, output_throughput: 0 },
      { ...stage, concurrency: 1, output_throughput: 10 },
    ],
  };
  const second = {
    ...record('b'),
    results: [
      { ...stage, concurrency: 4, output_throughput: 30 },
      { ...stage, concurrency: 16, output_throughput: Number.NaN },
    ],
  };
  assert.deepEqual(buildChartData([first, second], 'output_throughput', 'mean'), [
    { concurrency: 1, run0: 10, run1: null },
    { concurrency: 4, run0: null, run1: 30 },
    { concurrency: 16, run0: 0, run1: null },
  ]);
  assert.deepEqual(
    first.results.map((p) => p.concurrency),
    [16, 1]
  );
  assert.equal(buildChartData([first], 'ttft_ms', 'p99')[0].run0, 35);
  assert.equal(buildChartData([first], 'tpot_ms', 'mean')[0].run0, null);
});

it('overlays saved runs, switches latency percentiles, and flags different workloads', async () => {
  records = [
    {
      ...record('one'),
      results: [
        {
          ...stage,
          concurrency: 1,
          output_throughput: 10,
          ttft_ms: { ...metric, mean: 10, p99: 20 },
        },
        {
          ...stage,
          concurrency: 4,
          output_throughput: 30,
          ttft_ms: { ...metric, mean: 20, p99: 80 },
        },
      ],
    },
    {
      ...record('two'),
      config: { ...record('two').config, input_tokens: 1024 },
      results: [
        {
          ...stage,
          concurrency: 1,
          output_throughput: 15,
          ttft_ms: { ...metric, mean: 30, p99: 40 },
        },
        {
          ...stage,
          concurrency: 4,
          output_throughput: 40,
          ttft_ms: { ...metric, mean: 40, p99: 100 },
        },
      ],
    },
  ];
  await act(async () => root.render(<Benchmarks />));
  await act(async () => {
    for (const checkbox of host.querySelectorAll<HTMLInputElement>('input[type=checkbox]'))
      checkbox.click();
  });
  const charts = host.querySelector('[data-testid="benchmark-charts"]')!;
  const figures = charts.querySelectorAll('figure');
  assert.equal(figures.length, 3);
  assert.equal(figures[0].querySelectorAll('.recharts-line-curve').length, 2);
  const lines = figures[0].querySelectorAll('.recharts-line-curve');
  assert.notEqual(lines[0].getAttribute('stroke'), lines[1].getAttribute('stroke'));
  assert.ok(charts.querySelector('[role=status]')?.textContent?.includes('different workloads'));
  assert.equal(figures[2].querySelector('svg'), null);
  const before = figures[1].querySelector('.recharts-line-curve')!.getAttribute('d');
  await act(async () => {
    const select = charts.querySelector('select')!;
    select.value = 'p99';
    select.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
  });
  assert.notEqual(figures[1].querySelector('.recharts-line-curve')!.getAttribute('d'), before);
  const { haveDifferentWorkloads } = require('./chart-data') as typeof import('./chart-data');
  assert.equal(haveDifferentWorkloads([record('one'), record('two')]), false);
  assert.equal(
    haveDifferentWorkloads([
      record('one'),
      { ...record('two'), runner: { name: 'other', version: '1', metrics_version: '2' } },
    ]),
    true
  );
});
