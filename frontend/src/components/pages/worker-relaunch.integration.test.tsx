/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { WorkerRelaunchOperation } from './worker-relaunch';

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
  IS_REACT_ACT_ENVIRONMENT: true,
  requestAnimationFrame: (callback: FrameRequestCallback) => setTimeout(callback, 0),
  cancelAnimationFrame: (id: number) => clearTimeout(id),
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
const { useWorkerRelaunches, WorkerRelaunchDialog, WorkerRelaunchHistory } =
  require('./worker-relaunch') as typeof import('./worker-relaunch');
const originalGet = request.get,
  originalPost = request.post;
let root: Root, host: HTMLDivElement;
let posts: { url: string; body: Record<string, string> }[];
let jobs: WorkerRelaunchOperation[];
let calls: number, closed: number, failOnce: boolean;

beforeEach(() => {
  posts = [];
  jobs = [];
  calls = 0;
  closed = 0;
  failOnce = false;
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  request.get = (async () => {
    calls++;
    return jobs;
  }) as typeof request.get;
  request.post = (async (url: string, body: Record<string, string>) => {
    posts.push({ url, body });
    if (failOnce) {
      failOnce = false;
      throw new Error('transport timeout');
    }
    return { operation_id: 'one', source_worker: '[::1]:9999', status: 'starting' };
  }) as typeof request.post;
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  request.get = originalGet;
  request.post = originalPost;
});
function Probe({ enabled = true }: { enabled?: boolean }) {
  const state = useWorkerRelaunches(enabled);
  return (
    <>
      <output>{JSON.stringify([...state.busyAddresses])}</output>
      <button
        onClick={() =>
          state.addOperation({ operation_id: 'new', source_worker: 'old:1', status: 'starting' })
        }
      >
        Start
      </button>
      <WorkerRelaunchHistory operations={state.operations} error={state.error} />
    </>
  );
}
function submit(action = 'restart') {
  return [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === `clusterInfo.relaunch.${action}`
  )!;
}
it('restores a pending transfer from the server after remounting', async () => {
  jobs = [
    { operation_id: 'one', source_worker: 'old:1', target_worker: 'new:2', status: 'pending' },
  ];
  await act(async () => root.render(<Probe />));
  assert.deepEqual(JSON.parse(document.querySelector('output')!.textContent!), ['old:1', 'new:2']);
  assert.match(document.body.textContent!, /clusterInfo.relaunch.pendingHint/);
  await act(async () => root.unmount());
  root = createRoot(host);
  await act(async () => root.render(<Probe />));
  assert.equal(calls, 2);
  assert.match(document.body.textContent!, /old:1.*new:2/);
});
it('does not fetch administrative operations when disabled', async () => {
  await act(async () => root.render(<Probe enabled={false} />));
  assert.equal(calls, 0);
});
it('stops polling when idle and resumes for a newly started operation', async () => {
  await act(async () => root.render(<Probe />));
  await act(async () => new Promise((resolve) => setTimeout(resolve, 2100)));
  assert.equal(calls, 1);
  jobs = [{ operation_id: 'new', source_worker: 'old:1', status: 'starting' }];
  await act(async () => document.querySelector<HTMLButtonElement>('button')!.click());
  assert.equal(calls, 2);
  jobs = [{ operation_id: 'new', source_worker: 'old:1', status: 'succeeded' }];
  await act(async () => new Promise((resolve) => setTimeout(resolve, 2100)));
  assert.equal(calls, 3);
  await act(async () => new Promise((resolve) => setTimeout(resolve, 2100)));
  assert.equal(calls, 3);
});
it('retries an uncertain submission with the same request ID and typed environment path', async () => {
  failOnce = true;
  await act(async () =>
    root.render(
      <WorkerRelaunchDialog address="[::1]:9999" onClose={() => closed++} onStarted={() => {}} />
    )
  );
  const environment = document.querySelector('details')!;
  assert.equal(environment.open, false);
  await act(async () => environment.querySelector('summary')!.click());
  assert.equal(environment.open, true);
  const input = document.querySelector<HTMLInputElement>('input')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      ' /opt/new/bin/python '
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await act(async () => submit('upgrade').click());
  assert.equal(closed, 0);
  assert.match(document.body.textContent!, /transport timeout/);
  await act(async () => submit('upgrade').click());
  assert.equal(posts.length, 2);
  assert.equal(posts[0].url, '/v1/workers/%5B%3A%3A1%5D%3A9999/relaunch');
  assert.deepEqual(posts[0].body, posts[1].body);
  assert.equal(posts[0].body.python_executable, '/opt/new/bin/python');
  assert.ok(posts[0].body.request_id);
  assert.equal(closed, 1);
});

it('restarts in the current environment without sending a Python override', async () => {
  await act(async () =>
    root.render(
      <WorkerRelaunchDialog address="[::1]:9999" onClose={() => closed++} onStarted={() => {}} />
    )
  );
  assert.equal(document.querySelector('details')!.open, false);
  assert.equal(submit().hasAttribute('aria-label'), false);
  await act(async () => submit().click());
  assert.equal(posts.length, 1);
  assert.ok(posts[0].body.request_id);
  assert.equal('python_executable' in posts[0].body, false);
  assert.equal(closed, 1);
});
