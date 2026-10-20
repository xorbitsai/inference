import assert from 'node:assert/strict';
import test from 'node:test';
import React, { act } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';

import { I18nProvider } from '@/contexts/i18n-context';
import { ProgressTiming } from './progress-timing';

test('Chinese generation timing renders unknown ETA as 估算中', () => {
  const html = renderToStaticMarkup(
    <I18nProvider initialLocale="zh">
      <ProgressTiming timing={{ elapsed_seconds: 750, eta_seconds: null }} />
    </I18nProvider>
  );
  assert.match(html, /已用时间/);
  assert.match(html, /00:12:30/);
  assert.match(html, /预计剩余/);
  assert.match(html, /估算中/);
});

test('known ETA uses the elapsed time format', () => {
  const html = renderToStaticMarkup(
    <I18nProvider initialLocale="zh">
      <ProgressTiming timing={{ elapsed_seconds: 750, eta_seconds: 140 }} />
    </I18nProvider>
  );
  assert.match(html, /00:12:30/);
  assert.match(html, /00:02:20/);
  assert.doesNotMatch(html, /估算中/);
});

test('elapsed time advances and the timer is cleaned up after unmount', async (t) => {
  const dom = new JSDOM('<div id="root"></div>', { url: 'http://localhost' });
  dom.window.localStorage.setItem('app_locale', 'zh');
  const previous = new Map<string, PropertyDescriptor | undefined>();
  for (const [key, value] of Object.entries({
    window: dom.window,
    document: dom.window.document,
    navigator: dom.window.navigator,
    localStorage: dom.window.localStorage,
    IS_REACT_ACT_ENVIRONMENT: true,
  })) {
    previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  let now = 1000;
  let tick: (() => void) | undefined;
  t.mock.method(performance, 'now', () => now);
  t.mock.method(globalThis, 'setInterval', (callback: () => void) => {
    tick = callback;
    return 1 as unknown as ReturnType<typeof setInterval>;
  });
  const clear = t.mock.method(globalThis, 'clearInterval', () => {});
  const { createRoot } = await import('react-dom/client');
  const root = createRoot(document.getElementById('root')!);
  t.after(async () => {
    await act(async () => root.unmount());
    dom.window.close();
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
  });
  await act(async () =>
    root.render(
      <I18nProvider initialLocale="zh">
        <div id="live">
          <ProgressTiming timing={{ elapsed_seconds: 750, eta_seconds: null }} />
        </div>
      </I18nProvider>
    )
  );
  now += 2000;
  await act(async () => tick!());
  assert.match(document.getElementById('live')!.textContent!, /00:12:32/);
  await act(async () => root.render(null));
  assert.equal(clear.mock.calls.length, 1);
});
