/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import { AxiosError } from 'axios';
import type { ServiceRelaunchOperation } from './service-relaunch';

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
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => ({ globalReady }),
};
require('@/hooks/use-menu-auth');
require.cache[require.resolve('@/hooks/use-menu-auth')]!.exports = {
  useMenuAuth: () => ({ isAdmin }),
};
const { ServiceRelaunchPanel } =
  require('./service-relaunch') as typeof import('./service-relaunch');
const originalGet = request.get,
  originalPost = request.post;
const prefix = 'clusterInfo.serviceRelaunch';
const paths = ['/v1/api-relaunches', '/v1/supervisor-relaunches'];
let root: Root, host: HTMLDivElement;
let isAdmin: boolean, globalReady: boolean;
let jobs: Record<string, ServiceRelaunchOperation[]>;
let failures: Record<string, Error[]>;
let gets: string[];
let posts: { url: string; body: Record<string, string> }[];
let loseResponse: boolean, rejectOnce: boolean;
let updateError: boolean;

beforeEach(() => {
  isAdmin = globalReady = true;
  loseResponse = rejectOnce = false;
  updateError = false;
  gets = [];
  posts = [];
  jobs = Object.fromEntries(paths.map((path) => [path, []]));
  failures = Object.fromEntries(paths.map((path) => [path, []]));
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  request.get = (async (url: string, config) => {
    assert.equal(config?.suppressGlobalError, true);
    gets.push(url);
    if (url.endsWith('/updates')) {
      if (updateError) throw new Error('package source unavailable');
      return { current_version: '4.1.0', latest_version: '4.2.0', update_available: true };
    }
    const error = failures[url]?.shift();
    if (error) throw error;
    return jobs[url];
  }) as typeof request.get;
  request.post = (async (url: string, body: Record<string, string>, config) => {
    assert.equal(config?.suppressGlobalError, true);
    posts.push({ url, body });
    if (rejectOnce) {
      rejectOnce = false;
      throw new Error('transport timeout');
    }
    const path = url.includes('/supervisor/') ? paths[1] : paths[0];
    const job: ServiceRelaunchOperation = {
      operation_id: `${posts.length}`,
      request_id: body.request_id,
      created_at: Date.now() / 1000,
      status: 'starting',
    };
    jobs[path] = [job];
    if (loseResponse) throw new Error('response lost');
    return job;
  }) as typeof request.post;
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  request.get = originalGet;
  request.post = originalPost;
});

function trigger(service: 'api' | 'supervisor') {
  return [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === `${prefix}.${service}.title`
  )!;
}
async function submit() {
  await act(async () => {
    document
      .querySelector('form')!
      .dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
}
async function tick() {
  await act(async () => new Promise((resolve) => setTimeout(resolve, 2100)));
}

it('hides both controls and sends no administrative queries until ready and authorized', async () => {
  globalReady = false;
  await act(async () => root.render(<ServiceRelaunchPanel />));
  assert.equal(gets.length, 0);
  assert.equal(host.textContent, '');
  globalReady = true;
  isAdmin = false;
  await act(async () => root.render(<ServiceRelaunchPanel />));
  assert.equal(gets.length, 0);
  assert.equal(host.textContent, '');
});

it('restores independent API and Supervisor progress after remounting', async () => {
  jobs[paths[0]] = [{ operation_id: 'same-id', status: 'draining', created_at: 1 }];
  jobs[paths[1]] = [{ operation_id: 'same-id', status: 'preparing', created_at: 2 }];
  await act(async () => root.render(<ServiceRelaunchPanel />));
  assert.equal(trigger('api').disabled, true);
  assert.equal(trigger('supervisor').disabled, true);
  assert.match(host.textContent!, /api.drainingHint/);
  assert.match(host.textContent!, /preparing/);
  await act(async () => root.unmount());
  root = createRoot(host);
  await act(async () => root.render(<ServiceRelaunchPanel />));
  for (const path of paths) assert.equal(gets.filter((value) => value === path).length, 2);
  assert.match(host.textContent!, /api.drainingHint/);
  assert.match(host.textContent!, /preparing/);
});

it('retains progress through a connection failure, recovers, then stops polling', async () => {
  jobs[paths[0]] = [{ operation_id: 'one', status: 'draining', created_at: 1 }];
  await act(async () => root.render(<ServiceRelaunchPanel />));
  failures[paths[0]].push(new Error('connection reset'));
  await tick();
  assert.match(host.textContent!, /reconnecting/);
  assert.match(host.textContent!, /draining/);
  assert.doesNotMatch(host.textContent!, /serviceRelaunch.failed/);
  assert.equal(trigger('api').disabled, true);
  jobs[paths[0]] = [{ operation_id: 'one', status: 'succeeded', created_at: 1 }];
  await tick();
  assert.doesNotMatch(host.textContent!, /reconnecting/);
  assert.match(host.textContent!, /succeeded/);
  assert.equal(trigger('api').disabled, false);
  const count = gets.length;
  await tick();
  assert.equal(gets.length, count);
});

it('disables an unsupported deployment without retrying forever and offers an explicit refresh', async () => {
  const error = new AxiosError('unsupported');
  error.response = {
    status: 409,
    data: { detail: 'Requires hosted Supervisor' },
  } as typeof error.response;
  failures[paths[0]].push(error);
  await act(async () => root.render(<ServiceRelaunchPanel />));
  assert.equal(trigger('api').disabled, true);
  assert.equal(trigger('supervisor').disabled, false);
  assert.match(host.textContent!, /Requires hosted Supervisor/);
  await tick();
  assert.equal(gets.filter((path) => path === paths[0]).length, 1);
  const retry = [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === `${prefix}.retry`
  )!;
  await act(async () => retry.click());
  assert.equal(trigger('api').disabled, false);
  assert.doesNotMatch(host.textContent!, /Requires hosted Supervisor/);
});

it('uses the existing endpoints and trims the optional prepared environment path', async () => {
  await act(async () => root.render(<ServiceRelaunchPanel />));
  await act(async () => trigger('api').click());
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
  await submit();
  assert.equal(posts[0].url, '/v1/api/relaunch');
  assert.equal(posts[0].body.python_executable, '/opt/new/bin/python');
  assert.ok(posts[0].body.request_id);
  await act(async () => trigger('supervisor').click());
  await submit();
  assert.equal(posts[1].url, '/v1/supervisor/relaunch');
  assert.equal('python_executable' in posts[1].body, false);
  assert.notEqual(posts[0].body.request_id, posts[1].body.request_id);
  assert.equal(trigger('api').disabled, true);
  assert.equal(trigger('supervisor').disabled, true);
});

for (const service of ['api', 'supervisor'] as const) {
  for (const dismissal of ['cancel', 'Escape', 'close'] as const) {
    it(`returns focus to the ${service} launcher after keyboard ${dismissal} dismissal`, async () => {
      await act(async () => root.render(<ServiceRelaunchPanel />));
      const launcher = trigger(service);
      await act(async () => {
        launcher.focus();
        launcher.click();
      });
      const dialog = document.querySelector<HTMLElement>('[role="dialog"]')!;
      assert.ok(dialog.contains(document.activeElement));
      await act(async () => {
        if (dismissal === 'Escape') {
          dialog.dispatchEvent(
            new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })
          );
        } else {
          const button =
            dismissal === 'cancel'
              ? [...dialog.querySelectorAll<HTMLButtonElement>('button')].find(
                  (item) => item.textContent === 'common.cancel'
                )!
              : dialog.querySelector<HTMLButtonElement>('[data-slot="dialog-close"]')!;
          button.focus();
          button.click();
        }
      });
      assert.equal(document.querySelector('[role="dialog"]'), null);
      const deadline = Date.now() + 1000;
      while (document.activeElement !== launcher && Date.now() < deadline) {
        await act(async () => new Promise((resolve) => setTimeout(resolve, 10)));
      }
      assert.ok(document.activeElement === launcher, 'focus must return to the launcher');
      assert.equal(posts.length, 0);
    });
  }
}

it('keeps the same idempotency key when retrying an uncertain submission', async () => {
  rejectOnce = true;
  await act(async () => root.render(<ServiceRelaunchPanel />));
  await act(async () => trigger('supervisor').click());
  await submit();
  assert.match(document.body.textContent!, /transport timeout/);
  assert.ok(document.querySelector('[role="dialog"]'));
  await submit();
  assert.deepEqual(posts[0], posts[1]);
  assert.equal(document.querySelector('[role="dialog"]'), null);
});

it('recovers an accepted operation after losing the POST response without resubmitting', async () => {
  loseResponse = true;
  await act(async () => root.render(<ServiceRelaunchPanel />));
  await act(async () => trigger('api').click());
  await submit();
  assert.equal(posts.length, 1);
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.match(host.textContent!, /starting/);
  assert.doesNotMatch(host.textContent!, /serviceRelaunch.failed/);
  assert.equal(trigger('api').disabled, true);
});

it('shows server-confirmed failures and allows a new operation', async () => {
  jobs[paths[1]] = [
    { operation_id: 'one', status: 'failed', error: 'Python executable not found', created_at: 1 },
  ];
  await act(async () => root.render(<ServiceRelaunchPanel />));
  assert.match(host.textContent!, /serviceRelaunch.failed/);
  assert.match(host.textContent!, /Python executable not found/);
  assert.equal(trigger('supervisor').disabled, false);
});

it('checks the selected component and submits the detected version to the existing upgrade endpoint', async () => {
  await act(async () => root.render(<ServiceRelaunchPanel />));
  await act(async () => trigger('supervisor').click());
  const button = (key: string) =>
    [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (item) => item.textContent === `clusterInfo.relaunch.${key}`
    )!;
  await act(async () => button('checkUpdates').click());
  assert.equal(gets.at(-1), '/v1/supervisor/updates');
  assert.match(document.body.textContent!, /4.1.0.*4.2.0/);
  await act(async () => button('selectUpdate').click());
  assert.ok(button('installUpgrade'));
  await submit();
  assert.equal(posts[0].url, '/v1/supervisor/relaunch');
  assert.equal(posts[0].body.version, '4.2.0');
  assert.equal('python_executable' in posts[0].body, false);
});

it('shows update-check failure while still allowing a current-environment restart', async () => {
  updateError = true;
  await act(async () => root.render(<ServiceRelaunchPanel />));
  await act(async () => trigger('api').click());
  const check = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (item) => item.textContent === 'clusterInfo.relaunch.checkUpdates'
  )!;
  await act(async () => check.click());
  assert.match(document.body.textContent!, /package source unavailable/);
  await submit();
  assert.equal(posts[0].url, '/v1/api/relaunch');
  assert.equal('version' in posts[0].body, false);
});

it('stops administrative polling when the administrator loses access', async () => {
  jobs[paths[0]] = [{ operation_id: 'one', status: 'draining', created_at: 1 }];
  await act(async () => root.render(<ServiceRelaunchPanel />));
  isAdmin = false;
  await act(async () => root.render(<ServiceRelaunchPanel />));
  const count = gets.length;
  await tick();
  assert.equal(gets.length, count);
  assert.equal(host.textContent, '');
});
