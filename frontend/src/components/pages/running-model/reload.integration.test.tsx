/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { beforeEach, afterEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
  Element: dom.window.Element,
  HTMLElement: dom.window.HTMLElement,
  Node: dom.window.Node,
  MutationObserver: dom.window.MutationObserver,
  Event: dom.window.Event,
  CustomEvent: dom.window.CustomEvent,
  getComputedStyle: dom.window.getComputedStyle,
  requestAnimationFrame: (callback: FrameRequestCallback) => setTimeout(callback, 0),
  cancelAnimationFrame: (id: number) => clearTimeout(id),
  IS_REACT_ACT_ENVIRONMENT: true,
  ResizeObserver: class {
    observe() {}
    unobserve() {}
    disconnect() {}
  },
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
for (const key of Reflect.ownKeys(dom.window)) {
  if (!(key in globalThis))
    Object.defineProperty(globalThis, key, Object.getOwnPropertyDescriptor(dom.window, key)!);
}
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
require.extensions['.css'] = () => {};

const translate = (key: string) => key;
require('@/contexts/i18n-context');
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({ t: translate }),
};
const globalContext = { clusterAuth: { auth: false } };
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => globalContext,
};
require('@/hooks/use-menu-auth');
require.cache[require.resolve('@/hooks/use-menu-auth')]!.exports = {
  useMenuAuth: () => ({ isAdmin: false, hasRouterRead: false, canAccessRouterPage: false }),
};
let navigations: string[] = [];
const router = { push: (url: string) => navigations.push(url) };
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = { useRouter: () => router };
require('@/components/pages/running-model-detail/components/try-api-drawer');
require.cache[
  require.resolve('@/components/pages/running-model-detail/components/try-api-drawer')
]!.exports = { TryApiDrawer: () => null, getTryApiAbility: () => 'chat' };
require('./add-replica-dialog');
require.cache[require.resolve('./add-replica-dialog')]!.exports = {
  __esModule: true,
  default: () => null,
};

const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const RunningModel = require('./index').default as typeof import('./index').default;
const originalGet = request.get;
const originalPost = request.post;
let root: Root;
let host: HTMLDivElement;
let model: Record<string, unknown>;
let requests: string[];
let posts: { url: string; body: unknown }[];

beforeEach(() => {
  navigations = [];
  requests = [];
  posts = [];
  model = {
    id: 'model/one',
    model_name: 'qwen3',
    model_type: 'LLM',
    model_engine: 'vLLM',
    model_format: 'pytorch',
    quantization: 'none',
    model_size_in_billions: '0_6',
    model_ability: ['chat'],
    replica: 1,
    n_worker: 1,
    accelerators: ['0'],
  };
  request.get = async <T,>(url: string) => {
    requests.push(url);
    if (url === '/v1/models') return { data: [model] } as T;
    if (url === '/v1/cluster/info') return { workers: [] } as T;
    if (url === '/v1/autostart/models/summary') return { models: [] } as T;
    if (url.endsWith('/reload/config'))
      return {
        enabled: true,
        can_upgrade_environment: false,
        parameters: { max_num_seqs: 'positive_int' },
        model_config: { max_num_seqs: 4 },
      } as T;
    if (url.endsWith('/reload')) return { status: 'idle' } as T;
    return [] as T;
  };
  request.post = async <T,>(url: string, body: unknown) => {
    posts.push({ url, body });
    return { status: 'reloading', stage: 'queued' } as T;
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

function reconfigureButton() {
  return [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.getAttribute('aria-label') === 'runningModels.reload.title'
  );
}

it('reconfigures the selected model from Running Models without navigating to chat', async () => {
  await act(async () => root.render(<RunningModel />));
  assert.ok(reconfigureButton());
  await act(async () => reconfigureButton()!.click());
  assert.deepEqual(navigations, []);
  assert.ok(requests.includes('/v1/models/model%2Fone/reload/config'));
  const input = document.querySelector<HTMLInputElement>('input[aria-label="max_num_seqs"]')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      '8'
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  const apply = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'runningModels.reload.apply'
  )!;
  await act(async () => apply.click());
  assert.deepEqual(posts, [
    { url: '/v1/models/model%2Fone/reload', body: { model_config: { max_num_seqs: 8 } } },
  ]);
});

for (const overrides of [{ model_engine: 'Transformers' }, { replica: 2 }, { n_worker: 2 }]) {
  it(`hides reconfiguration for unsupported model setup ${JSON.stringify(overrides)}`, async () => {
    Object.assign(model, overrides);
    await act(async () => root.render(<RunningModel />));
    assert.equal(reconfigureButton(), undefined);
  });
}

it('keeps reload polling scheduled when the list rerenders', async (context) => {
  const get = request.get;
  let statusReads = 0;
  request.get = async <T,>(url: string) => {
    if (url.endsWith('/reload')) {
      statusReads++;
      return { status: statusReads === 1 ? 'reloading' : 'ready' } as T;
    }
    return get<T>(url);
  };
  await act(async () => root.render(<RunningModel />));
  context.mock.timers.enable({ apis: ['setTimeout'] });
  await act(async () => reconfigureButton()!.click());
  for (let index = 0; index < 4; index++) {
    await act(async () => context.mock.timers.tick(200));
    await act(async () => root.render(<RunningModel />));
  }
  assert.equal(statusReads, 1);
  await act(async () => context.mock.timers.tick(200));
  assert.equal(statusReads, 2);
  assert.equal(document.querySelector('[role="dialog"]'), null);
});
