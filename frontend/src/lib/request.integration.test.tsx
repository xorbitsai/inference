/* eslint-disable @typescript-eslint/no-require-imports -- Initialize browser globals before UI modules. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it, mock } from 'node:test';
import React, { act } from 'react';
import axios, { AxiosError } from 'axios';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { RequestErrorExtension, RequestErrorEvent } from '@/lib/request-extension-types';
import type { Locale } from '@/types/common';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  Event: dom.window.Event,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
  sessionStorage: dom.window.sessionStorage,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });

let responseHeaders: Record<string, string>;
let detail: string;
let responseStatus = 429;
axios.defaults.adapter = async (config) => {
  throw new AxiosError(
    'Request failed with status code 429',
    'ERR_BAD_RESPONSE',
    config,
    undefined,
    {
      config,
      data: { detail },
      status: responseStatus,
      statusText: 'Too Many Requests',
      headers: responseHeaders,
    }
  );
};
let navigations: string[];
const router = { replace: (url: string) => navigations.push(url) };
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = { useRouter: () => router };
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const { toast } = require('sonner') as typeof import('sonner');
const { I18nProvider } =
  require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const RequestProvider = require('@/contexts/request-context')
  .default as typeof import('@/contexts/request-context').default;
const { RequestExtensionHandlers } =
  require('@/contexts/request-context') as typeof import('@/contexts/request-context');
const { requestManager } = require('./request-manager') as typeof import('./request-manager');
const { validateRequestErrorExtensions } =
  require('./request-extension-types') as typeof import('./request-extension-types');
const extensions = (require('./request-extensions') as typeof import('./request-extensions'))
  .requestErrorExtensions as RequestErrorExtension[];
let received: RequestErrorEvent | null = null;
function Handler({ event }: { event: RequestErrorEvent | null }) {
  received = event;
  return event ? <button onClick={event.release}>release extension</button> : null;
}
function register(statuses = [503], event = 'extension:test:unavailable', handler = Handler) {
  extensions.push({ statuses, event, Handler: handler });
}
const request = require('./request').default as typeof import('./request').default;
let notifications: { message: unknown; options: unknown }[];
let root: Root;
let host: HTMLDivElement;

beforeEach((context) => {
  (context as import('node:test').TestContext).mock.timers.enable({ apis: ['setTimeout'] });
  extensions.length = 0;
  received = null;
  responseStatus = 429;
  requestManager.reset401();
  requestManager.reset403();
  responseHeaders = { 'x-xinference-error-code': 'ip_banned', 'retry-after': '120' };
  detail = 'Too many failed attempts. IP temporarily banned.';
  notifications = [];
  navigations = [];
  localStorage.clear();
  sessionStorage.clear();
  mock.method(toast, 'error', (message: unknown, options: unknown) => {
    notifications.push({ message, options });
    return 'test-toast';
  });
  host = document.body.appendChild(document.createElement('div'));
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  mock.restoreAll();
});

async function render(locale: Locale = 'zh') {
  localStorage.setItem('app_locale', locale);
  await act(async () => {
    root.render(
      <I18nProvider initialLocale={locale}>
        <RequestProvider>
          <RequestExtensionHandlers />
          app
        </RequestProvider>
      </I18nProvider>
    );
  });
}

it('explains an IP ban in Chinese with recovery steps and remaining time', async () => {
  await render();
  await assert.rejects(request.get('/v1/models'));
  assert.equal(notifications.length, 1);
  assert.match(String(notifications[0].message), /IP.*临时封禁/);
  assert.match(String(notifications[0].message), /系统管理 → 安全设置/);
  assert.match(String(notifications[0].message), /120 秒/);
  assert.deepEqual(notifications[0].options, { id: 'auth-ban-ip' });
  assert.deepEqual(navigations, []);
});

it('distinguishes a key ban from an IP ban', async () => {
  await render();
  responseHeaders['x-xinference-error-code'] = 'key_banned';
  detail = 'Too many failed attempts for this key.';
  await assert.rejects(request.get('/v1/models'));
  assert.match(String(notifications[0].message), /API Key 在此 IP 下被临时封禁/);
  assert.deepEqual(notifications[0].options, { id: 'auth-ban-key' });
});

it('keeps an unrelated model rate limit as a regular 429 error', async () => {
  await render();
  responseHeaders = {};
  detail = 'Rate limit reached for the model';
  await assert.rejects(request.get('/v1/models'));
  assert.equal(notifications[0].message, 'Server error: 429 - Rate limit reached for the model');
});

it('recognizes legacy ban messages without inventing a remaining time', async () => {
  await render();
  responseHeaders = {};
  await assert.rejects(request.get('/v1/models'));
  assert.match(String(notifications[0].message), /临时封禁/);
  assert.doesNotMatch(String(notifications[0].message), /秒后/);
});

it('honors suppressed and request-specific error handling', async () => {
  await render();
  await assert.rejects(request.get('/v1/models', { suppressGlobalError: true }));
  assert.equal(notifications.length, 0);
  await assert.rejects(request.get('/v1/models', { errorMessages: { 429: 'Custom message' } }));
  assert.equal(notifications[0].message, 'Server error: 429 - Custom message');
});

for (const [locale, expected] of [
  ['en', /temporarily blocked/],
  ['zh-TW', /暫時封鎖/],
  ['ja', /一時的にブロック/],
  ['ko', /일시적으로 차단/],
] as const) {
  it(`uses the selected ${locale} locale`, async () => {
    await render(locale);
    await assert.rejects(request.get('/v1/models'));
    assert.match(String(notifications[0].message), expected);
    assert.match(String(notifications[0].message), /120/);
  });
}

it('dispatches one extension event, honors suppression and releases on close/unmount', async () => {
  register();
  responseStatus = 503;
  await render();
  await act(async () => {
    await assert.rejects(request.get('/models', { suppressGlobalError: true }));
  });
  assert.equal(received, null);
  await act(async () => {
    await Promise.all([
      assert.rejects(request.get('/models')),
      assert.rejects(request.get('/models')),
    ]);
  });
  const first = received!;
  assert.equal(first.status, 503);
  assert.equal(notifications.length, 0);
  await act(async () => {
    await assert.rejects(request.get('/models'));
  });
  assert.equal(received, first);
  await act(async () => first.release());
  assert.equal(received, null);
  await act(async () => {
    await assert.rejects(request.get('/models'));
  });
  assert.notEqual(received, first);
  await act(async () => root.render(<div>app</div>));
  const release = requestManager.acquireErrorLock('extension:test:unavailable');
  assert.ok(release);
  release();
});
it('preserves ban notifications and authentication handling over extensions', async () => {
  register([401, 403, 429]);
  await render();
  await assert.rejects(request.get('/models'));
  assert.equal(received, null);
  assert.deepEqual(notifications[0].options, { id: 'auth-ban-ip' });
  responseStatus = 401;
  sessionStorage.setItem('token', 'expired');
  sessionStorage.setItem('refresh_token', 'expired-refresh');
  await assert.rejects(request.get('/models', { skipAuthRefresh: true }));
  assert.deepEqual(navigations, ['/login']);
  assert.equal(sessionStorage.getItem('token'), null);
  assert.equal(sessionStorage.getItem('refresh_token'), null);
  assert.equal(received, null);
  responseStatus = 403;
  await assert.rejects(request.get('/models'));
  assert.equal(notifications.at(-1)?.message, 'Server error: 403 No auth');
  assert.equal(received, null);
});
it('allows ordinary 429 extensions after ban detection', async () => {
  register([429]);
  responseHeaders = {};
  detail = 'Model rate limit';
  await render();
  await act(async () => {
    await assert.rejects(request.get('/models'));
  });
  assert.equal(received!.message, 'Model rate limit');
  assert.equal(notifications.length, 0);
});
it('expires abandoned locks without allowing stale releases to unlock a new owner', (context) => {
  const first = requestManager.acquireErrorLock('extension:test:lock')!;
  assert.equal(requestManager.acquireErrorLock('extension:test:lock'), null);
  context.mock.timers.tick(30_000);
  const second = requestManager.acquireErrorLock('extension:test:lock');
  assert.ok(second);
  first();
  assert.equal(requestManager.acquireErrorLock('extension:test:lock'), null);
  second();
});
it('recovers from errors emitted before subscription and toast-only handlers', async (context) => {
  register();
  responseStatus = 503;
  await assert.rejects(request.get('/models'));
  context.mock.timers.tick(30_000);
  await render();
  await act(async () => {
    await assert.rejects(request.get('/models'));
  });
  const first = received;
  await act(async () => context.mock.timers.tick(30_000));
  await act(async () => {
    await assert.rejects(request.get('/models'));
  });
  assert.notEqual(received, first);
});
it('rejects duplicate and non-namespaced events including built-in names', () => {
  register();
  assert.equal(validateRequestErrorExtensions(extensions), extensions);
  register();
  assert.throws(() => validateRequestErrorExtensions(extensions), /Duplicate/);
  extensions.pop();
  extensions[0].event = 'SERVER_ERROR';
  assert.throws(() => validateRequestErrorExtensions(extensions), /Invalid/);
});
it('isolates a throwing extension and releases its active lock', async () => {
  register([503], 'extension:test:broken', ({ event }) => {
    if (event) throw new Error('broken handler');
    return null;
  });
  mock.method(console, 'error', () => {});
  responseStatus = 503;
  await render();
  await act(async () => {
    await assert.rejects(request.get('/models'));
  });
  assert.match(host.textContent!, /app/);
  const release = requestManager.acquireErrorLock('extension:test:broken');
  assert.ok(release);
  release();
});
