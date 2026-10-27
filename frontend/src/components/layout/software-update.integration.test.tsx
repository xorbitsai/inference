/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
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
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = {
  usePathname: () => '/cluster-information',
  useRouter: () => ({ push() {}, replace() {} }),
};
require('next/link');
require.cache[require.resolve('next/link')]!.exports = {
  __esModule: true,
  default: ({ children, href }: React.PropsWithChildren<{ href: string }>) => (
    <a href={href}>{children}</a>
  ),
};
require('next/image');
require.cache[require.resolve('next/image')]!.exports = {
  __esModule: true,
  default: (props: React.ImgHTMLAttributes<HTMLImageElement>) => (
    <span role="img" aria-label={props.alt ?? ''} />
  ),
};
require('@/contexts/i18n-context');
const t = (key: string) => key;
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({ t, locale: 'en' }),
};
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => ({
    globalReady,
    clusterAuth: { auth: false },
    clusterUIConfig: { auth_advanced: false },
    clusterVersion: { version: displayedVersion },
    setClusterVersion: (value: { version: string }) => {
      displayedVersion = value.version;
      root.render(<Sidebar />);
    },
  }),
};
require('@/hooks/use-menu-auth');
require.cache[require.resolve('@/hooks/use-menu-auth')]!.exports = {
  useMenuAuth: () => ({ isAdmin, hasPageAccess: () => true }),
};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { Sidebar } = require('./sidebar') as typeof import('./sidebar');
const originalGet = request.get,
  originalPost = request.post;
const A = '9999.0.1',
  B = '9999.0.2';
interface Job {
  operation_id: string;
  request_id: string;
  requested_version: string;
  status: string;
  error?: string;
}
let root: Root, host: HTMLDivElement;
let isAdmin: boolean,
  globalReady: boolean,
  failWorker: boolean,
  loseApiResponse: boolean,
  pauseWorker: boolean;
let displayedVersion: string;
let versions: Record<string, string>, jobs: Record<string, Job>;
let posts: { url: string; body: Record<string, string> }[], gets: string[];
beforeEach(() => {
  isAdmin = globalReady = true;
  failWorker = loseApiResponse = pauseWorker = false;
  displayedVersion = A;
  versions = { api: A, supervisor: A, worker: A };
  jobs = {};
  posts = [];
  gets = [];
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  request.get = (async (url: string) => {
    gets.push(url);
    if (url === '/v1/workers') return [{ 'work-ip': '127.0.0.1:9978' }];
    if (url.endsWith('/updates')) {
      const component = url.includes('/workers/')
        ? 'worker'
        : url.includes('/supervisor/')
          ? 'supervisor'
          : 'api';
      return {
        current_version: versions[component],
        latest_version: B,
        update_available: versions[component] !== B,
      };
    }
    if (url === '/v1/cluster/version') return { version: versions.api, 'full-revisionid': null };
    if (url === '/status') return { ready: true };
    const history = url.match(/^\/v1\/(api|supervisor|worker)-relaunches(?:\/(.*))?$/);
    if (history) {
      const job = jobs[history[1]];
      if (job?.status === 'succeeded') versions[history[1]] = job.requested_version;
      return history[2] ? job : job ? [job] : [];
    }
    throw new Error(`unexpected GET ${url}`);
  }) as typeof request.get;
  request.post = (async (url: string, body: Record<string, string>) => {
    posts.push({ url, body });
    const component = url.includes('/workers/')
      ? 'worker'
      : url.includes('/supervisor/')
        ? 'supervisor'
        : 'api';
    const job = (jobs[component] = {
      operation_id: component,
      request_id: body.request_id,
      requested_version: body.version,
      status:
        component === 'worker' && failWorker
          ? 'failed'
          : component === 'worker' && pauseWorker
            ? 'installing'
            : 'succeeded',
      ...(component === 'worker' && failWorker ? { error: 'engine build unavailable' } : {}),
    });
    if (job.status === 'succeeded') versions[component] = body.version;
    if (component === 'api' && loseApiResponse) throw new Error('response lost');
    return job;
  }) as typeof request.post;
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  request.get = originalGet;
  request.post = originalPost;
});
const trigger = () => host.querySelector<HTMLButtonElement>('[aria-label="softwareUpdate.check"]')!;
function button(key: string) {
  return [...document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')].find(
    (item) => item.textContent === `softwareUpdate.${key}`
  )!;
}
async function render() {
  await act(async () => root.render(<Sidebar />));
}
async function open() {
  await act(async () => trigger().click());
}

it('puts checking on the sidebar version and offers the mocked newer version', async () => {
  await render();
  assert.equal(gets.length, 0);
  assert.equal(trigger().textContent, `v:${A}`);
  await open();
  assert.match(document.body.textContent!, /9999.0.1.*9999.0.2/s);
  assert.equal(gets.filter((url) => url.endsWith('/updates')).length, 3);
  assert.equal(button('upgrade').disabled, false);
});
it('installs Worker, Supervisor and API in order, then shows B and a working service', async () => {
  await render();
  await open();
  await act(async () => button('upgrade').click());
  assert.deepEqual(
    posts.map((item) => item.url),
    ['/v1/workers/127.0.0.1%3A9978/relaunch', '/v1/supervisor/relaunch', '/v1/api/relaunch']
  );
  assert.ok(posts.every((item) => item.body.version === B && !('python_executable' in item.body)));
  assert.equal(displayedVersion, B);
  assert.equal(trigger().textContent, `v:${B}`);
  assert.match(document.body.textContent!, /softwareUpdate.complete/);
  assert.deepEqual(await request.get('/status'), { ready: true });
});
it('continues after recovering the accepted API job from a lost response', async () => {
  loseApiResponse = true;
  await render();
  await open();
  await act(async () => button('upgrade').click());
  assert.equal(posts.length, 3);
  assert.ok(gets.includes('/v1/api-relaunches'));
  assert.equal(displayedVersion, B);
});
it('stops before Supervisor and API if the Worker cannot preserve its engine', async () => {
  failWorker = true;
  await render();
  await open();
  await act(async () => button('upgrade').click());
  assert.equal(posts.length, 1);
  assert.equal(displayedVersion, A);
  assert.match(document.body.textContent!, /engine build unavailable/);
  assert.equal(button('upgrade').disabled, true);
});
it('keeps the current job when the dialog closes and the sidebar collapses', async () => {
  pauseWorker = true;
  await render();
  await open();
  await act(async () => button('upgrade').click());
  assert.equal(posts.length, 1);
  await act(async () => button('close').click());
  await act(async () =>
    host.querySelector<HTMLButtonElement>('[aria-label="common.packUp"]')!.click()
  );
  await open();
  assert.equal(gets.filter((url) => url.endsWith('/updates')).length, 3);
  assert.match(document.body.textContent!, /softwareUpdate.installing/);
  jobs.worker.status = 'succeeded';
  await act(async () => new Promise((resolve) => setTimeout(resolve, 1700)));
  assert.equal(posts.length, 3);
  assert.equal(displayedVersion, B);
});
it('does not offer a downgrade or post when every component is current', async () => {
  versions = { api: B, supervisor: B, worker: B };
  displayedVersion = B;
  await render();
  await open();
  assert.equal(button('upgrade').disabled, true);
  assert.equal(posts.length, 0);
});
it('keeps the version read-only for non-administrators', async () => {
  isAdmin = false;
  await render();
  assert.equal(trigger(), null);
  assert.match(host.textContent!, /9999.0.1/);
  assert.equal(gets.length, 0);
});
