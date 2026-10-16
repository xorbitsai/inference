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

require('@/contexts/i18n-context');
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({ t: (key: string) => key }),
};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { ReloadDialog } = require('./reload-dialog') as typeof import('./reload-dialog');
let root: Root;
let host: HTMLDivElement;
let enabled = true,
  polls = 0,
  closed = 0,
  refreshed = 0;
let posts: { url: string; body: unknown }[] = [];
let initial: { status: string; stage?: string; error?: string; restored?: boolean } = {
  status: 'idle',
  stage: 'queued',
};
let final: { status: string; error?: string; restored?: boolean } = { status: 'ready' };
const originalGet = request.get,
  originalPost = request.post;
beforeEach(() => {
  enabled = true;
  polls = 0;
  closed = 0;
  refreshed = 0;
  posts = [];
  initial = { status: 'idle', stage: 'queued' };
  final = { status: 'ready' };
  request.get = async <T,>(url: string) => {
    if (url.endsWith('/config'))
      return {
        enabled,
        model_config: { max_num_seqs: 16, enforce_eager: false },
        parameters: {
          max_num_seqs: 'positive_int',
          enforce_eager: 'bool',
          max_model_len: 'positive_int',
        },
      } as T;
    return (polls++ === 0 ? initial : final) as T;
  };
  request.post = async <T,>(url: string, body: unknown) => {
    posts.push({ url, body });
    return { operation_id: 'one', status: 'reloading', stage: 'queued' } as T;
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
async function render() {
  await act(async () =>
    root.render(
      <ReloadDialog
        open
        modelUid="model/name"
        onReady={() => {
          refreshed++;
        }}
        onOpenChange={() => {
          closed++;
        }}
      />
    )
  );
}
function applyButton() {
  return [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'runningModels.reload.apply'
  )!;
}

it('submits typed changed fields and follows the asynchronous reload', async () => {
  await render();
  assert.ok(applyButton().disabled);
  const input = document.querySelector<HTMLInputElement>('input[aria-label="max_num_seqs"]')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      '32'
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector<HTMLButtonElement>('button[aria-label="enforce_eager"]')!.click();
  });
  await act(async () => applyButton().click());
  assert.deepEqual(posts, [
    {
      url: '/v1/models/model%2Fname/reload',
      body: { model_config: { max_num_seqs: 32, enforce_eager: true } },
    },
  ]);
  assert.ok(applyButton().disabled);
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1100));
  });
  assert.equal(refreshed, 1);
  assert.equal(closed, 1);
});
it('explains that launch-time weight caching is required', async () => {
  enabled = false;
  await render();
  assert.ok(applyButton().disabled);
  assert.match(document.body.textContent || '', /runningModels.reload.disabled/);
  assert.equal(document.querySelectorAll('input').length, 0);
});
it('resumes an existing reload and reports restoration', async () => {
  initial = { status: 'reloading', stage: 'draining' };
  final = { status: 'error', error: 'out of memory', restored: true };
  await render();
  assert.ok(applyButton().disabled);
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1100));
  });
  assert.match(
    document.querySelector('[role="alert"]')?.textContent || '',
    /out of memory.*runningModels.reload.restored/
  );
  assert.equal(refreshed, 0);
  assert.equal(posts.length, 0);
});

it('does not present an earlier failed operation as a new dialog error', async () => {
  initial = { status: 'error', error: 'previous out of memory', restored: true };
  await render();
  assert.equal(document.querySelector('[role="alert"]'), null);
  assert.equal(document.querySelector('[role="status"]'), null);
  assert.equal(posts.length, 0);
});
