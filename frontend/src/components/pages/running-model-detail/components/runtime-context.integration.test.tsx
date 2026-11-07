/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { beforeEach, afterEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { RunningModelDetail } from '@/types/services';
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

require('@/contexts/i18n-context');
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({ t: (key: string) => key }),
};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { RuntimeContext, useModelRuntime } =
  require('./runtime-context') as typeof import('./runtime-context');
function RuntimeDetails({ uid, model }: { uid: string; model: RunningModelDetail | null }) {
  const runtime = useModelRuntime(uid, model);
  return <RuntimeContext runtime={runtime} registeredContextLength={model?.context_length} />;
}
const originalGet = request.get;
let root: Root;
let host: HTMLDivElement;
let model: RunningModelDetail | null;
beforeEach(() => {
  model = { model_type: 'LLM', context_length: 40960 } as RunningModelDetail;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  request.get = originalGet;
});
async function render(uid = 'demo') {
  await act(async () => root.render(<RuntimeDetails uid={uid} model={model} />));
}
it('shows the serving limit and each replica, using the native endpoint', async () => {
  request.get = async <T,>(url: string) => {
    assert.equal(url, '/v1/models/model%2Fname/runtime');
    return {
      runtime_context_length: 4096,
      replicas: [
        { replica_uid: 'rep0', runtime_context_length: 8192 },
        { replica_uid: 'rep1', runtime_context_length: 4096 },
      ],
    } as T;
  };
  await render('model/name');
  assert.match(host.textContent!, /runtimeContextLength/);
  assert.match(host.textContent!, /registeredContextLength: 40960/);
  assert.match(host.textContent!, /rep0: 8192/);
  assert.match(host.textContent!, /rep1: 4096/);
});
it('shows unknown when the engine or endpoint cannot report a limit', async () => {
  request.get = async <T,>() => ({ runtime_context_length: null, replicas: [] }) as T;
  await render();
  assert.match(host.textContent!, /unknownContextLength/);
  request.get = async () => {
    throw new Error('older server');
  };
  await render('another');
  assert.match(host.textContent!, /unknownContextLength/);
});
it('ignores a previous model response after navigation', async () => {
  let resolveOld: (value: unknown) => void = () => {};
  request.get = <T,>(url: string) =>
    url.includes('/old/')
      ? new Promise<T>((resolve) => {
          resolveOld = resolve as (value: unknown) => void;
        })
      : Promise.resolve({ runtime_context_length: 4096, replicas: [] } as T);
  await render('old');
  await render('new');
  await act(async () => resolveOld({ runtime_context_length: 131072, replicas: [] }));
  assert.match(host.textContent!, /4096/);
  assert.doesNotMatch(host.textContent!, /131072/);
});

it('fetches once on entry and after reload, without periodic requests', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  let calls = 0;
  request.get = async <T,>() => {
    calls += 1;
    return { runtime_context_length: calls === 1 ? 33604 : 8192, replicas: [] } as T;
  };
  await render();
  assert.equal(calls, 1);
  assert.match(host.textContent!, /33604/);
  await act(async () => t.mock.timers.tick(60000));
  await render();
  assert.equal(calls, 1);
  model = { ...model! };
  await render();
  assert.equal(calls, 2);
  assert.match(host.textContent!, /8192/);
  assert.doesNotMatch(host.textContent!, /33604/);
});
it('waits for a loaded model before fetching runtime information', async () => {
  let calls = 0;
  request.get = async <T,>() => {
    calls += 1;
    return { runtime_context_length: 33604, replicas: [] } as T;
  };
  model = null;
  await render();
  assert.equal(calls, 0);
  model = { model_type: 'LLM', context_length: 40960 } as RunningModelDetail;
  await render();
  assert.equal(calls, 1);
  assert.match(host.textContent!, /33604/);
});

it('shows the loaded value as the primary context and shares it with chat', async () => {
  const Module = require('node:module');
  function mockModule(name: string, exports: unknown) {
    const filename = require.resolve(name);
    require.cache[filename] = Object.assign(new Module(filename), {
      exports: { __esModule: true, ...(exports as object) },
      loaded: true,
    });
  }
  mockModule('next/navigation', { useRouter: () => ({ back() {} }) });
  mockModule('@/components/ui/page-container', {
    default: ({ children }: { children: React.ReactNode }) => <main>{children}</main>,
  });
  mockModule('../capability-config', { CAPABILITY_CONFIGS: {} });
  mockModule('../panels/capability-task-panel', { default: () => null });
  mockModule('../panels/chat-panel', {
    ChatPanel: ({ runtimeContextLength }: { runtimeContextLength?: number | null }) => (
      <div data-chat-context={runtimeContextLength ?? 'unknown'} />
    ),
  });
  mockModule('./try-api-drawer', { TryApiDrawer: () => null });
  let reloadComplete = () => {};
  mockModule('./reload-dialog', {
    ReloadDialog: ({ onReady }: { onReady: () => void }) => {
      reloadComplete = onReady;
      return null;
    },
  });
  const RunningModelDetail = require('../index').default as typeof import('../index').default;
  let loadedLength = 33604;
  let calls = 0;
  request.get = async <T,>(url: string) => {
    if (url.endsWith('/runtime')) {
      calls += 1;
      return { runtime_context_length: loadedLength, replicas: [] } as T;
    }
    return {
      model_type: 'LLM',
      model_name: 'qwen3',
      model_ability: ['chat'],
      context_length: 40960,
      model_engine: 'MLX',
    } as T;
  };
  await act(async () => root.render(<RunningModelDetail modelUid="qwen3" />));
  assert.equal(calls, 1);
  assert.match(host.textContent!, /runtimeContextLength33604/);
  assert.match(host.textContent!, /registeredContextLength: 40960/);
  assert.doesNotMatch(host.textContent!, /runningModels.detail.contextLength/);
  assert.equal(
    host.querySelector('[data-chat-context]')?.getAttribute('data-chat-context'),
    '33604'
  );
  loadedLength = 8192;
  await act(async () => reloadComplete());
  assert.equal(calls, 2);
  assert.equal(
    host.querySelector('[data-chat-context]')?.getAttribute('data-chat-context'),
    '8192'
  );
});
