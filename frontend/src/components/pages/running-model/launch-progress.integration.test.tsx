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
const menuAuth = {
  isAdmin: false,
  hasRouterRead: false,
  canAccessRouterPage: false,
  canWriteModels: false,
};
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => globalContext,
};
require('@/hooks/use-menu-auth');
require.cache[require.resolve('@/hooks/use-menu-auth')]!.exports = {
  useMenuAuth: () => menuAuth,
};
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = { useRouter: () => ({ push() {} }) };
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const RunningModel = require('./index').default as typeof import('./index').default;
const { translations } = require('@/i18n/translations') as typeof import('@/i18n/translations');
const originalGet = request.get;
const originalPost = request.post;
const originalDelete = request.delete;
let root: Root;
let host: HTMLDivElement;
let models: Record<string, unknown>[];
let instances: Record<string, unknown>[];
let progress: Record<string, unknown>;
let reads: string[];
let posts: string[];
let deletes: string[];
let failDelete: boolean;
let failReads: Set<string>;

const uid = 'model/one';
const progressUrl = '/v1/models/model%2Fone/progress';
function instance(status = 'LOADING', created = 1) {
  return {
    model_uid: uid,
    model_name: 'demo',
    model_ability: ['chat'],
    replica: 2,
    status,
    can_cancel_launch: status === 'CREATING' || status === 'LOADING',
    instance_created_ts: created,
    replica_statuses: [
      {
        replica_id: 0,
        replica_model_uid: 'one-rep0',
        worker_address: 'worker-a:1234',
        status,
        created_ts: 1,
      },
      {
        replica_id: 1,
        replica_model_uid: 'one-rep1',
        worker_address: 'worker-b:1234',
        status,
        created_ts: 1,
      },
    ],
  };
}
function running() {
  return {
    id: uid,
    model_name: 'demo',
    model_type: 'LLM',
    model_engine: 'vLLM',
    model_format: 'pytorch',
    quantization: 'none',
    model_size_in_billions: 7,
    model_ability: ['chat'],
    replica: 2,
    n_worker: 1,
    accelerators: ['0'],
  };
}
function replicaProgress(id: number, stage: string, value: number) {
  return {
    replica_id: id,
    replica_model_uid: `one-rep${id}`,
    stage,
    progress: value,
    download_files: [],
  };
}
function button(label: string) {
  return [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (item) => item.textContent === label || item.getAttribute('aria-label') === label
  );
}
async function mount() {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(async () => root.render(<RunningModel />));
}

beforeEach(async () => {
  localStorage.clear();
  globalContext.clusterAuth.auth = false;
  menuAuth.canWriteModels = false;
  models = [];
  instances = [instance()];
  progress = {
    stage: 'loading',
    replicas: [replicaProgress(0, 'loading', 0.8), replicaProgress(1, 'loading', 0.5)],
  };
  reads = [];
  posts = [];
  deletes = [];
  failDelete = false;
  failReads = new Set();
  request.get = async <T,>(url: string) => {
    reads.push(url);
    if (failReads.has(url)) throw new Error('temporary failure');
    if (url === '/v1/models') return structuredClone({ data: models }) as T;
    if (url === '/v1/models/instances') return structuredClone(instances) as T;
    if (url.endsWith('/progress')) return structuredClone(progress) as T;
    if (url === '/v1/cluster/info') return { workers: [] } as T;
    if (url === '/v1/autostart/models/summary') return { models: [] } as T;
    return [] as T;
  };
  request.post = async <T,>(url: string) => {
    posts.push(url);
    if (url.endsWith('/cancel')) instances = [];
    return {} as T;
  };
  request.delete = async <T,>(url: string) => {
    deletes.push(url);
    if (failDelete) throw new Error('delete failed');
    instances = instances.filter(
      (item) => `/v1/models/${encodeURIComponent(String(item.model_uid))}` !== url
    );
    return null as T;
  };
});

afterEach(async () => {
  if (root) await act(async () => root.unmount());
  document.body.innerHTML = '';
  request.get = originalGet;
  request.post = originalPost;
  request.delete = originalDelete;
});

it('discovers a loading instance with independent replica progress and no inference controls', async () => {
  progress = {
    replicas: [
      replicaProgress(0, 'downloading', 0.3),
      {
        ...replicaProgress(1, 'installing_dependencies', 0.8),
        dependency_install_completed: 2,
        dependency_install_total: 5,
        dependency_install_plan: ['example==2.0'],
      },
    ],
  };
  await mount();
  assert.match(host.textContent!, /model\/one/);
  assert.match(host.textContent!, /runningModels.statuses.LOADING/);
  assert.match(host.textContent!, /worker-a:1234/);
  assert.match(host.textContent!, /worker-b:1234/);
  assert.match(host.textContent!, /launchModel.stageDownloading/);
  assert.match(host.textContent!, /launchModel.stageInstallingDependencies/);
  assert.match(host.textContent!, /30%/);
  assert.match(host.textContent!, /80%/);
  assert.match(host.textContent!, /example==2.0/);
  assert.equal(button('runningModels.tryApi'), undefined);
  assert.equal(button('runningModels.addReplica'), undefined);
  assert.equal(button('runningModels.detail.modelDetails'), undefined);
  assert.ok(button('runningModels.cancelLaunch'));
  assert.ok(reads.includes(progressUrl));
  assert.deepEqual(posts, []);
});

it('restores an unfinished launch after leaving and returning without browser storage', async () => {
  await mount();
  assert.match(host.textContent!, /80%/);
  await act(async () => root.unmount());
  host.remove();
  localStorage.clear();
  await mount();
  assert.match(host.textContent!, /80%/);
  assert.match(host.textContent!, /runningModels.statuses.LOADING/);
  assert.equal(reads.filter((url) => url === '/v1/models/instances').length, 2);
  assert.deepEqual(posts, []);
});

it('follows stage changes and replaces the pending card with the ready model exactly once', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  await mount();
  progress = { replicas: [replicaProgress(0, 'installing_dependencies', 0.9)] };
  await act(async () => context.mock.timers.tick(1000));
  assert.match(host.textContent!, /launchModel.stageInstallingDependencies/);
  progress = { replicas: [replicaProgress(0, 'loading', 1)] };
  await act(async () => context.mock.timers.tick(1000));
  assert.match(host.textContent!, /99%/);
  assert.equal(button('runningModels.tryApi'), undefined);
  instances = [instance('READY')];
  models = [running()];
  await act(async () => context.mock.timers.tick(2000));
  assert.match(host.textContent!, /runningModels.statuses.READY/);
  assert.equal(host.querySelectorAll('h4').length, 2); // one list card and its detail heading
  assert.ok(button('runningModels.tryApi'));
  assert.ok(button('runningModels.addReplica'));
  assert.ok(button('runningModels.detail.modelDetails'));
  assert.equal(button('runningModels.cancelLaunch'), undefined);
  assert.equal(host.querySelector('[role="progressbar"]'), null);
  const count = reads.filter((url) => url === progressUrl).length;
  await act(async () => context.mock.timers.tick(4000));
  assert.equal(reads.filter((url) => url === progressUrl).length, count);
});

it('retains failed instances and server replica errors across remounts', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const failed = instance('ERROR');
  failed.replica_statuses[1] = {
    ...failed.replica_statuses[1],
    error_message: 'GPU out of memory',
  } as (typeof failed.replica_statuses)[number];
  instances = [failed];
  await mount();
  assert.match(host.textContent!, /GPU out of memory/);
  assert.match(host.textContent!, /runningModels.launchFailed/);
  assert.equal(button('runningModels.tryApi'), undefined);
  await act(async () => context.mock.timers.tick(4000));
  assert.equal(reads.filter((url) => url === progressUrl).length, 0);
  await act(async () => root.unmount());
  host.remove();
  await mount();
  assert.match(host.textContent!, /GPU out of memory/);
});

it('stops launch polling on failure even if a replica still reports an earlier loading state', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  await mount();
  instances = [{ ...instance(), status: 'ERROR', can_cancel_launch: false }];
  await act(async () => context.mock.timers.tick(2000));
  const count = reads.filter((url) => url === progressUrl).length;
  await act(async () => context.mock.timers.tick(4000));
  assert.equal(reads.filter((url) => url === progressUrl).length, count);
  assert.match(host.textContent!, /runningModels.launchFailed/);
  assert.equal(button('runningModels.cancelLaunch'), undefined);
});

it('preserves progress through transient failures and keeps polling', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  await mount();
  failReads = new Set(['/v1/models', '/v1/models/instances', progressUrl]);
  await act(async () => context.mock.timers.tick(2000));
  assert.match(host.textContent!, /80%/);
  assert.match(host.textContent!, /runningModels.statuses.LOADING/);
  assert.match(host.textContent!, /runningModels.statusUnavailable/);
  failReads.clear();
  instances = [instance('ERROR')];
  await act(async () => context.mock.timers.tick(2000));
  assert.match(host.textContent!, /runningModels.launchFailed/);
});

it('discovers launches started elsewhere while the model list is already open', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  instances = [];
  await mount();
  assert.match(host.textContent!, /runningModels.noRunningModels/);
  instances = [instance('CREATING')];
  await act(async () => context.mock.timers.tick(2000));
  assert.match(host.textContent!, /runningModels.statuses.CREATING/);
  assert.match(host.textContent!, /80%/);
});

it('ignores late progress from a previously selected instance', async () => {
  instances = [instance(), { ...instance(), model_uid: 'model-two', model_name: 'second' }];
  const get = request.get;
  let resolveOld!: (value: unknown) => void;
  request.get = (<T,>(url: string) =>
    url === progressUrl
      ? new Promise<unknown>((resolve) => {
          resolveOld = resolve;
        })
      : get<T>(url)) as typeof request.get;
  await mount();
  const card = [...host.querySelectorAll('h4')].find((item) => item.textContent === 'model-two')!;
  await act(async () => card.click());
  assert.match(host.textContent!, /80%/);
  await act(async () => resolveOld({ replicas: [replicaProgress(0, 'loading', 0.73)] }));
  assert.doesNotMatch(host.textContent!, /73%/);
});

it('clears old progress when a model UID is reused for a new launch', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  await mount();
  const get = request.get;
  request.get = (<T,>(url: string) =>
    url === progressUrl
      ? Promise.reject(new Error('new progress record not ready'))
      : get<T>(url)) as typeof request.get;
  instances = [instance('CREATING', 2)];
  await act(async () => context.mock.timers.tick(2000));
  assert.doesNotMatch(host.textContent!, /80%/);
  assert.match(host.textContent!, /runningModels.statuses.CREATING/);
});

it('keeps existing running models when the instance API is unavailable and preserves Token Routers', async () => {
  models = [
    running(),
    {
      id: 'router',
      model_name: 'router',
      model_engine: 'token_router',
      model_kind: 'virtual',
      router_status: 'ready',
    },
    {
      id: 'hidden-router',
      model_name: 'hidden',
      model_engine: 'token_router',
      router_status: 'disabled',
    },
  ];
  failReads.add('/v1/models/instances');
  await mount();
  assert.match(host.textContent!, /demo/);
  assert.match(host.textContent!, /Token Router/);
  assert.doesNotMatch(host.textContent!, /hidden-router/);
  assert.ok(button('runningModels.tryApi'));
});

it('cancels the selected launch using its encoded UID', async () => {
  await mount();
  await act(async () => button('runningModels.cancelLaunch')!.click());
  const confirm = [
    ...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button'),
  ].find((item) => item.textContent === 'runningModels.cancelLaunch')!;
  assert.ok(confirm);
  await act(async () => confirm.click());
  assert.deepEqual(posts, ['/v1/models/model%2Fone/cancel']);
  assert.match(host.textContent!, /runningModels.noRunningModels/);
});

it('can cancel an initial launch with both ready and loading replicas', async () => {
  const mixed = instance('READY');
  mixed.can_cancel_launch = true;
  mixed.replica_statuses[1].status = 'LOADING';
  instances = [mixed];
  models = [running()];
  await mount();
  assert.ok(button('runningModels.cancelLaunch'));
  assert.equal(button('runningModels.terminateConfirmOk'), undefined);
  assert.equal(button('runningModels.addReplica'), undefined);
  await act(async () => button('runningModels.cancelLaunch')!.click());
  const confirm = [
    ...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button'),
  ].find((item) => item.textContent === 'runningModels.cancelLaunch')!;
  await act(async () => confirm.click());
  assert.deepEqual(posts, ['/v1/models/model%2Fone/cancel']);
});

it('keeps an established model usable during scale-up without offering initial-launch cancellation', async () => {
  const scaling = instance('READY');
  scaling.replica_statuses[1].status = 'CREATING';
  instances = [scaling];
  models = [running()];
  await mount();
  assert.equal(button('runningModels.cancelLaunch'), undefined);
  assert.equal(button('runningModels.terminateConfirmOk'), undefined);
  assert.ok(button('runningModels.tryApi'));
  assert.deepEqual(posts, []);
});

it('does not infer cancellation safety when an older server omits the capability', async () => {
  const legacy: Partial<ReturnType<typeof instance>> = instance();
  delete legacy.can_cancel_launch;
  instances = [legacy];
  await mount();
  assert.equal(button('runningModels.cancelLaunch'), undefined);
});

it('clears a failed launch without a running model and keeps it removed after remount', async () => {
  instances = [instance('ERROR')];
  await mount();
  await act(async () => button('runningModels.clearFailedRecord')!.click());
  const confirm = document.querySelector<HTMLElement>('[role="alertdialog"]')!;
  assert.match(confirm.textContent!, /runningModels.clearFailedRecordConfirm/);
  const clear = [...confirm.querySelectorAll('button')].find(
    (item) => item.textContent === 'runningModels.clearFailedRecord'
  )!;
  await act(async () => clear.click());
  assert.deepEqual(deletes, ['/v1/models/model%2Fone']);
  assert.match(host.textContent!, /runningModels.noRunningModels/);
  await act(async () => root.unmount());
  host.remove();
  await mount();
  assert.match(host.textContent!, /runningModels.noRunningModels/);
});

it('retains the failed record when deletion fails and allows retry', async () => {
  instances = [instance('ERROR')];
  failDelete = true;
  await mount();
  await act(async () => button('runningModels.clearFailedRecord')!.click());
  const clear = [
    ...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button'),
  ].find((item) => item.textContent === 'runningModels.clearFailedRecord')!;
  await act(async () => clear.click());
  assert.match(host.textContent!, /runningModels.launchFailed/);
  assert.ok(document.querySelector('[role="alertdialog"]'));
  failDelete = false;
  await act(async () => clear.click());
  assert.match(host.textContent!, /runningModels.noRunningModels/);
});

it('requires model write permission to clear failed records', async () => {
  globalContext.clusterAuth.auth = true;
  instances = [instance('ERROR')];
  await mount();
  assert.equal(button('runningModels.clearFailedRecord'), undefined);
  menuAuth.canWriteModels = true;
  await act(async () => root.render(<RunningModel />));
  assert.ok(button('runningModels.clearFailedRecord'));
});

it('keeps the confirmed deletion target when the selected model changes', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  instances = [instance('ERROR'), { ...instance('ERROR'), model_uid: 'second' }];
  await mount();
  await act(async () => button('runningModels.clearFailedRecord')!.click());
  // The first row may disappear in another browser while its dialog is open.
  instances = instances.slice(1);
  await act(async () => context.mock.timers.tick(2000));
  const clear = [
    ...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button'),
  ].find((item) => item.textContent === 'runningModels.clearFailedRecord')!;
  await act(async () => clear.click());
  assert.deepEqual(deletes, ['/v1/models/model%2Fone']);
  assert.match(host.textContent!, /second/);
});

it('waits for each list poll to finish and stops scheduling after unmount', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const get = request.get;
  let resolveInstances!: (value: unknown) => void;
  request.get = (<T,>(url: string) =>
    url === '/v1/models/instances'
      ? new Promise<unknown>((resolve) => {
          reads.push(url);
          resolveInstances = resolve;
        })
      : get<T>(url)) as typeof request.get;
  await mount();
  await act(async () => context.mock.timers.tick(10000));
  assert.equal(reads.filter((url) => url === '/v1/models/instances').length, 1);
  await act(async () => root.unmount());
  const count = reads.length;
  await act(async () => resolveInstances([instance()]));
  await act(async () => context.mock.timers.tick(10000));
  assert.equal(reads.length, count);
});

it('provides lifecycle labels in every maintained UI language', () => {
  for (const locale of Object.values(translations)) {
    const labels = locale.runningModels;
    for (const label of Object.values(labels.statuses)) assert.ok(label.length);
    assert.ok(labels.cancelLaunch);
    assert.ok(labels.cancelLaunchConfirm.includes('{{modelUid}}'));
    assert.ok(labels.clearFailedRecord);
    assert.ok(labels.clearFailedRecordConfirm.includes('{{modelUid}}'));
    assert.ok(labels.launchFailed);
  }
});
