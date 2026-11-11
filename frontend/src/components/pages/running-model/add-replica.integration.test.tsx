/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
import { AxiosError, AxiosHeaders, type AxiosRequestConfig } from 'axios';
// @ts-expect-error -- jsdom declarations are not installed.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { ReplicaItem, AddReplicaRequest } from '@/types/services';
import type { Locale } from '@/types/common';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
  HTMLElement: dom.window.HTMLElement,
  Element: dom.window.Element,
  Node: dom.window.Node,
  MutationObserver: dom.window.MutationObserver,
  Event: dom.window.Event,
  CustomEvent: dom.window.CustomEvent,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
for (const key of Reflect.ownKeys(dom.window)) {
  if (!(key in globalThis))
    Object.defineProperty(globalThis, key, Object.getOwnPropertyDescriptor(dom.window, key)!);
}
require.extensions['.css'] = () => {};
const actualI18n = require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const stubI18n = () => ({ t: (key: string) => key });
const i18nStub = { useI18n: stubI18n };
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = i18nStub;
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { useAddReplica, isSameReplica } =
  require('./use-add-replica') as typeof import('./use-add-replica');
const AddReplicaDialog = require('./add-replica-dialog')
  .default as typeof import('./add-replica-dialog').default;
const originalGet = request.get;
const originalPost = request.post;
let root: Root | undefined;
let host: HTMLDivElement;
let hook: ReturnType<typeof useAddReplica>;
let replicas: ReplicaItem[];
let failReads: boolean;
let post: ReturnType<typeof deferred>;
let posts: { url: string; body: AddReplicaRequest; config?: AxiosRequestConfig }[];
let outcomes: string[];
let refreshWarnings: number[];
let snapshots: { uid: string; replicas: ReplicaItem[] }[];
let reads: string[];

function deferred() {
  let resolve!: (value?: unknown) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function replica(id: number, status = 'READY', created = 1): ReplicaItem {
  return {
    replica_id: id,
    replica_model_uid: `model-rep${id}`,
    created_ts: created,
    status,
    worker_address: 'worker:1234',
    error_message: '',
    gpu_idx: [4, 5, 6, 7],
  };
}
function backendError(status: number, data: unknown) {
  return new AxiosError('request failed', undefined, undefined, undefined, {
    status,
    data,
    statusText: '',
    headers: new AxiosHeaders(),
    config: { headers: new AxiosHeaders() },
  });
}
async function mount(dialog = false) {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  function Harness() {
    hook = useAddReplica({
      onReplicas: (uid, value) => snapshots.push({ uid, replicas: value }),
      onSettled: (op) => outcomes.push(op.phase),
      onSuccessRefreshFailed: (op) => refreshWarnings.push(op.token),
    });
    return dialog ? (
      <AddReplicaDialog
        open
        onOpenChange={() => {}}
        onConfirm={(body) => void hook.submit('model/one', body)}
        loading={hook.operation?.phase === 'pending'}
        operation={hook.operation}
        elapsedSeconds={hook.elapsedSeconds}
        newReplicas={hook.newReplicas}
        onRefresh={() => void hook.refresh()}
        onAcknowledge={hook.reset}
        workerOptions={[]}
        modelUid="model/one"
        modelName="demo"
        modelType="LLM"
        modelEngine="vLLM"
        currentReplicaCount={1}
        defaultDevice="GPU"
      />
    ) : null;
  }
  await act(async () => root!.render(<Harness />));
}
async function submit(body: AddReplicaRequest = { replica: 1 }) {
  await act(async () => {
    void hook.submit('model/one', body);
  });
}
beforeEach(() => {
  replicas = [replica(0)];
  failReads = false;
  post = deferred();
  posts = [];
  outcomes = [];
  refreshWarnings = [];
  snapshots = [];
  reads = [];
  request.get = async <T,>(url: string) => {
    reads.push(url);
    if (failReads) throw new Error('offline');
    if (url.endsWith('/replicas')) return structuredClone(replicas) as T;
    if (url.endsWith('/progress'))
      return {
        replicas: replicas.map((r) => ({
          ...r,
          progress: 1,
          stage: 'loading',
          download_files: [],
        })),
      } as T;
    return {} as T;
  };
  request.post = <T,>(url: string, body: AddReplicaRequest, config?: AxiosRequestConfig) => {
    posts.push({ url, body, config });
    return post.promise as Promise<T>;
  };
});
afterEach(async () => {
  if (root) await act(async () => root!.unmount());
  root = undefined;
  document.body.innerHTML = '';
  request.get = originalGet;
  request.post = originalPost;
  i18nStub.useI18n = stubI18n;
  localStorage.clear();
});

it('waits 181 seconds, deduplicates clicks, preserves four GPU arguments and fixed UID', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'] });
  await mount();
  const body: AddReplicaRequest = {
    replica: 1,
    replica_config: [{ devices: [{ worker_ip: 'worker:1234', n_gpu: 4, gpu_idx: [4, 5, 6, 7] }] }],
  };
  await act(async () => {
    void hook.submit('model/one', body);
    void hook.submit('another', { replica: 2 });
  });
  assert.equal(posts.length, 1);
  assert.equal(posts[0].url, '/v1/models/model%2Fone/replicas');
  assert.deepEqual(posts[0].body, body);
  assert.equal(posts[0].config?.noTimeout, true);
  assert.equal(posts[0].config?.suppressGlobalError, true);
  await act(async () => context.mock.timers.tick(181000));
  assert.equal(hook.elapsedSeconds, 181);
  assert.equal(hook.operation?.phase, 'pending');
  assert.deepEqual(outcomes, []);
  replicas.push(replica(1));
  await act(async () => post.resolve({}));
  assert.equal(hook.operation?.phase, 'succeeded');
  assert.deepEqual(outcomes, ['succeeded']);
  assert.deepEqual(
    hook.newReplicas.map((r) => r.replica_id),
    [1]
  );
  assert.ok(snapshots.every((s) => s.uid === 'model/one'));
  const count = reads.length;
  await act(async () => context.mock.timers.tick(60000));
  assert.equal(reads.length, count);
});

it('shows only new replicas, clamps loading progress, and keeps unknown despite all READY', async () => {
  await mount(true);
  await submit({ replica: 2 });
  replicas.push(replica(1, 'LOADING'));
  await act(async () => {
    await hook.refresh();
  });
  assert.equal(document.querySelectorAll('article').length, 1);
  assert.ok(document.body.textContent?.includes('99%'));
  assert.ok(document.body.textContent?.includes('runningModels.addReplicaWaitingCount'));
  await act(async () => post.reject(backendError(504, { detail: 'gateway timeout' })));
  replicas.push(replica(2));
  replicas[1].status = 'READY';
  await act(async () => {
    await hook.refresh();
  });
  assert.equal(hook.operation?.phase, 'unknown');
  await submit();
  assert.equal(posts.length, 1);
  assert.ok(document.body.textContent?.includes('runningModels.addReplicaState_unknown'));
  const acknowledge = [...document.querySelectorAll('button')].find(
    (b) => b.textContent === 'runningModels.addReplicaAcknowledge'
  );
  assert.ok(acknowledge);
  await act(async () => acknowledge.click());
  assert.equal(hook.operation, null);
});

it('waits for the batch response even when every new replica is READY', async () => {
  await mount();
  await submit({ replica: 2 });
  replicas.push(replica(1), replica(2));
  await act(async () => {
    await hook.refresh();
  });
  assert.equal(hook.newReplicas.length, 2);
  assert.equal(hook.operation?.phase, 'pending');
  assert.deepEqual(outcomes, []);
  await act(async () => post.resolve({}));
  assert.deepEqual(outcomes, ['succeeded']);
});

it('retains backend batch failure after partial READY replicas are rolled back', async () => {
  await mount();
  await submit({ replica: 2 });
  replicas.push(replica(1));
  await act(async () => {
    await hook.refresh();
  });
  assert.equal(hook.operation?.phase, 'pending');
  replicas = [replica(0)];
  await act(async () =>
    post.reject(backendError(500, { detail: 'CUDA out of memory; rolled back' }))
  );
  assert.equal(hook.operation?.phase, 'failed');
  assert.match(hook.operation?.error ?? '', /CUDA out of memory/);
  assert.deepEqual(hook.newReplicas, []);
  assert.deepEqual(outcomes, ['failed']);
});

it('retains successful snapshots on read failure, recovers, and preserves HTTP success on final refresh failure', async () => {
  await mount();
  await submit();
  replicas.push(replica(1, 'LOADING'));
  await act(async () => {
    await hook.refresh();
  });
  failReads = true;
  await act(async () => {
    await hook.refresh();
  });
  assert.equal(hook.newReplicas.length, 1);
  assert.equal(hook.operation?.refreshFailed, true);
  assert.deepEqual(refreshWarnings, []);
  failReads = false;
  await act(async () => {
    await hook.refresh();
  });
  assert.equal(hook.operation?.refreshFailed, false);
  failReads = true;
  await act(async () => post.resolve({}));
  assert.equal(hook.operation?.phase, 'succeeded');
  assert.equal(hook.operation?.refreshFailed, true);
  assert.deepEqual(outcomes, ['succeeded']);
  assert.deepEqual(refreshWarnings, [hook.operation?.token]);
  await act(async () => {
    await hook.refresh();
  });
  assert.equal(refreshWarnings.length, 1);
});

it('ignores a previous final refresh failure after a new operation starts', async () => {
  await mount();
  await submit();
  const slow = deferred();
  const get = request.get;
  request.get = (<T,>(url: string) =>
    url.endsWith('/replicas') ? slow.promise : get<T>(url)) as typeof request.get;
  await act(async () => post.resolve({}));
  assert.equal(hook.operation?.phase, 'succeeded');
  await act(async () => hook.reset());
  request.get = get;
  post = deferred();
  await submit();
  assert.equal(hook.operation?.phase, 'pending');
  await act(async () => slow.reject(new Error('late refresh failure')));
  assert.equal(hook.operation?.phase, 'pending');
  assert.deepEqual(refreshWarnings, []);
  assert.deepEqual(outcomes, ['succeeded']);
});

it('ignores final refresh failure after unmount', async () => {
  await mount();
  await submit();
  const slow = deferred();
  const get = request.get;
  request.get = (<T,>(url: string) =>
    url.endsWith('/replicas') ? slow.promise : get<T>(url)) as typeof request.get;
  await act(async () => post.resolve({}));
  await act(async () => root!.unmount());
  root = undefined;
  await act(async () => slow.reject(new Error('late refresh failure')));
  assert.deepEqual(refreshWarnings, []);
});

for (const [locale, closeText] of Object.entries({
  en: 'Close',
  zh: '关闭',
  'zh-TW': '關閉',
  ja: '閉じる',
  ko: '닫기',
})) {
  it(`renders the close button using the real ${locale} translation`, async () => {
    const { translations } = require('@/i18n/translations') as typeof import('@/i18n/translations');
    assert.equal(translations[locale as Locale].common.close, closeText);
    localStorage.setItem('app_locale', locale);
    i18nStub.useI18n = actualI18n.useI18n;
    await mount();
    const { I18nProvider } = actualI18n;
    await act(async () =>
      root!.render(
        <I18nProvider initialLocale={locale as Locale}>
          <AddReplicaDialog
            open
            onOpenChange={() => {}}
            onConfirm={() => {}}
            loading={false}
            operation={{
              token: 1,
              modelUid: 'demo',
              count: 1,
              startedAt: 1,
              phase: 'failed',
              baseline: [],
              baselineReady: true,
              replicas: [],
              progress: null,
              refreshFailed: false,
            }}
            workerOptions={[]}
            modelUid="demo"
            modelName="demo"
            modelType="LLM"
            currentReplicaCount={1}
            defaultDevice="GPU"
          />
        </I18nProvider>
      )
    );
    assert.ok(
      [...document.querySelectorAll('button')].some((button) => button.textContent === closeText)
    );
    assert.ok(!document.body.textContent?.includes('common.close'));
  });
}

it('does not POST without a valid baseline snapshot', async () => {
  await mount();
  failReads = true;
  await submit();
  assert.equal(posts.length, 0);
  assert.equal(hook.operation?.phase, 'failed');
});

it('backs off failed polling and automatically recovers without another POST', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  await mount();
  await submit();
  failReads = true;
  await act(async () => context.mock.timers.tick(2000));
  assert.equal(hook.operation?.refreshFailed, true);
  const count = reads.length;
  await act(async () => context.mock.timers.tick(2000));
  assert.equal(reads.length, count);
  failReads = false;
  replicas.push(replica(1, 'LOADING'));
  await act(async () => context.mock.timers.tick(2000));
  assert.equal(hook.operation?.refreshFailed, false);
  assert.equal(hook.newReplicas.length, 1);
  assert.equal(posts.length, 1);
});

for (const [status, data, expected] of [
  [500, '<html>proxy error</html>', 'unknown'],
  [502, { detail: 'gateway failure' }, 'unknown'],
  [422, { detail: [{ msg: 'invalid GPU indexes' }] }, 'failed'],
] as const) {
  it(`classifies HTTP ${status} as ${expected}`, async () => {
    await mount();
    await submit();
    await act(async () => post.reject(backendError(status, data)));
    assert.equal(hook.operation?.phase, expected);
  });
}

it('aborts browser requests on unmount and ignores late outcomes', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  await mount();
  await submit();
  const signal = posts[0].config?.signal;
  await act(async () => root!.unmount());
  root = undefined;
  assert.equal(signal?.aborted, true);
  const count = reads.length;
  await act(async () => {
    post.resolve({});
    context.mock.timers.tick(60000);
  });
  assert.deepEqual(outcomes, []);
  assert.equal(reads.length, count);
});

it('serializes slow polling rounds and ignores results after manual acknowledgement', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  await mount();
  await submit();
  await act(async () => post.reject(new AxiosError('Network Error')));
  const slow = deferred();
  request.get = <T,>(url: string) => {
    reads.push(url);
    return slow.promise as Promise<T>;
  };
  await act(async () => {
    void hook.refresh();
  });
  const count = reads.length;
  await act(async () => context.mock.timers.tick(60000));
  assert.equal(reads.length, count);
  const snapshotCount = snapshots.length;
  await act(async () => hook.reset());
  await act(async () => slow.resolve([]));
  assert.equal(hook.operation, null);
  assert.equal(snapshots.length, snapshotCount);
});

it('distinguishes recreated numeric IDs from the original runtime', () => {
  assert.equal(isSameReplica(replica(0), replica(0, 'READY', 2)), false);
  assert.equal(
    isSameReplica({ ...replica(0), model_address: 'a' }, { ...replica(0), model_address: 'b' }),
    false
  );
  assert.equal(isSameReplica(replica(0), replica(0, 'LOADING')), true);
});

it('uses timeout zero in the real Axios adapter for add POST, keeping ordinary GET at 60000', async () => {
  const configs: AxiosRequestConfig[] = [];
  const adapter: NonNullable<AxiosRequestConfig['adapter']> = async (config) => {
    configs.push(config);
    return { data: {}, status: 200, statusText: 'OK', headers: new AxiosHeaders(), config };
  };
  await originalPost(
    '/v1/models/demo/replicas',
    { replica: 1 },
    { noTimeout: true, suppressGlobalError: true, adapter }
  );
  await originalGet('/v1/models/demo/replicas', { adapter });
  assert.equal(configs[0].timeout, 0);
  assert.equal(configs[1].timeout, 60000);
});
