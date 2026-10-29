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
  output_throughput: null,
  latency_ms: metric,
  ttft_ms: metric,
  tpot_ms: null,
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
