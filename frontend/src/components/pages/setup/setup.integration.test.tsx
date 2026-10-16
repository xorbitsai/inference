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

let navigations: string[];
const router = { replace: (path: string) => navigations.push(path) };
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = { useRouter: () => router };
require('next/image');
require.cache[require.resolve('next/image')]!.exports = { __esModule: true, default: () => null };
require('@/contexts/i18n-context');
const t = (key: string) => key;
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = { useI18n: () => ({ t }) };
const { toast } = require('sonner') as typeof import('sonner');
const originalError = toast.error;
const originalSuccess = toast.success;
const Setup = require('./index').default as typeof import('./index').default;
const originalFetch = globalThis.fetch;
let root: Root;
let host: HTMLDivElement;
let posts: Record<string, unknown>[];
let postStatus: number;
let errors: unknown[];

beforeEach(() => {
  posts = [];
  navigations = [];
  errors = [];
  postStatus = 201;
  localStorage.clear();
  sessionStorage.clear();
  toast.error = ((message: unknown) => errors.push(message)) as typeof toast.error;
  toast.success = (() => 'success') as typeof toast.success;
  globalThis.fetch = async (_url, options) => {
    if (options?.method === 'POST') {
      posts.push(JSON.parse(String(options.body)));
      return new Response(
        JSON.stringify(postStatus === 201 ? { id: 1 } : { detail: 'Invalid setup token.' }),
        { status: postStatus }
      );
    }
    return new Response(JSON.stringify({ needs_setup: true, password_min_length: 8 }));
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  globalThis.fetch = originalFetch;
  toast.error = originalError;
  toast.success = originalSuccess;
});
async function fill(placeholder: string, value: string) {
  const input = document.querySelector<HTMLInputElement>(`input[placeholder="${placeholder}"]`)!;
  assert.ok(input, placeholder);
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      value
    );
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
}
async function renderAndFillPassword() {
  await act(async () => root.render(<Setup />));
  await fill('userManagement.passwordPlaceholder', 'operator-password');
  await fill('userManagement.confirmPasswordPlaceholder', 'operator-password');
}
async function submit() {
  await act(async () => {
    document
      .querySelector('form')!
      .dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  });
}
it('requires the operator setup token before submitting', async () => {
  await renderAndFillPassword();
  assert.ok(document.body.textContent?.includes('setup.tokenHelp'));
  await submit();
  assert.equal(posts.length, 0);
  assert.equal(navigations.length, 0);
  assert.ok(document.body.textContent?.includes('common.valueEmpty'));
});
it('sends the setup token in the POST body without persisting it in browser storage', async () => {
  await renderAndFillPassword();
  await fill('setup.token', ' \toperator-setup-token\n');
  await submit();
  assert.deepEqual(posts, [
    { setup_token: 'operator-setup-token', username: 'admin', password: 'operator-password' },
  ]);
  assert.deepEqual(navigations, ['/login']);
  for (const storage of [localStorage, sessionStorage]) {
    for (let index = 0; index < storage.length; index++) {
      assert.ok(!storage.getItem(storage.key(index)!)?.includes('operator-setup-token'));
    }
  }
});
it('keeps the setup form available when the server rejects the token', async () => {
  postStatus = 403;
  await renderAndFillPassword();
  await fill('setup.token', 'incorrect-token');
  await submit();
  assert.equal(posts.length, 1);
  assert.deepEqual(errors, ['Invalid setup token.']);
  assert.equal(navigations.length, 0);
});
