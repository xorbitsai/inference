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
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
localStorage.setItem('app_locale', 'en');
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const Module = require('node:module');
function mockModule(name: string, exports: unknown) {
  const filename = require.resolve(name);
  require.cache[filename] = Object.assign(new Module(filename), { exports, loaded: true });
}
mockModule('@/components/ui/markdown-renderer', {
  default: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
});
mockModule('@/components/ui/media-preview', { MediaPreview: () => null });
mockModule('@/components/ui/file-upload', {
  FileUpload: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  FileUploadResult: () => null,
});
const requests: Record<string, unknown>[] = [];
mockModule('@/lib/eventStream', {
  EventStreamController: class {
    terminate() {}
  },
  postEventStreamFetcher: async ({
    data,
    options,
  }: {
    data: Record<string, unknown>;
    options: { onEnd: () => void };
  }) => {
    requests.push(data);
    options.onEnd();
  },
});
const { I18nProvider } =
  require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const { ChatPanel } = require('./chat-panel') as typeof import('./chat-panel');
let root: Root, host: HTMLDivElement, model: RunningModelDetail;
let runtimeContextLength: number | null | undefined;
beforeEach(() => {
  requests.length = 0;
  runtimeContextLength = undefined;
  model = {
    model_type: 'LLM',
    model_name: 'test',
    model_ability: ['chat'],
    context_length: 8192,
  } as RunningModelDetail;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});
async function render() {
  await act(async () =>
    root.render(
      <I18nProvider initialLocale="en">
        <ChatPanel
          model={model}
          modelUid="test-model"
          runtimeContextLength={runtimeContextLength}
        />
      </I18nProvider>
    )
  );
}
function button(text: string) {
  const result = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (b) => b.textContent === text || b.getAttribute('aria-label') === text
  );
  assert.ok(result, `Missing button ${text}`);
  return result;
}
async function click(text: string) {
  await act(async () => button(text).click());
}
async function input(label: string, value: string) {
  const node = document.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`);
  assert.ok(node, `Missing input ${label}`);
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      node,
      value
    );
    node.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
}
async function effort(level: string) {
  const label = [...document.querySelectorAll('label')].find(
    (node) => node.textContent === 'Reasoning effort'
  );
  assert.ok(label);
  const control = label.parentElement!.querySelector('div.relative > div');
  assert.ok(control);
  await act(async () => (control as HTMLElement).click());
  await click(level);
}
async function send() {
  const textarea = document.querySelector('textarea')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')!.set!.call(
      textarea,
      'Hello'
    );
    textarea.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
  await act(async () => {
    button('Send message').click();
    await new Promise((resolve) => setTimeout(resolve, 35));
  });
  assert.ok(requests.length);
  return requests.at(-1)!;
}
const maxTokens = 'Max tokens (0 uses the maximum available)';
it('hides undeclared effort and unknown runtime budgets and sends neither parameter', async () => {
  await render();
  await click('Additional inputs');
  assert.ok(!document.body.textContent!.includes('Reasoning effort'));
  assert.ok(!document.body.textContent!.includes('Thinking token budget'));
  await click('Save');
  const request = await send();
  assert.equal(Object.hasOwn(request, 'reasoning_effort'), false);
  assert.equal(Object.hasOwn(request, 'thinking_budget'), false);
});
it('offers only native GLM levels, forwards selection, and clears back to the model default', async () => {
  model.reasoning_effort = { values: ['low', 'high', 'max'] };
  await render();
  await click('Additional inputs');
  await effort('High');
  await click('Save');
  assert.equal((await send()).reasoning_effort, 'high');
  await click('Additional inputs');
  const label = [...document.querySelectorAll('label')].find(
    (node) => node.textContent === 'Reasoning effort'
  )!;
  await act(async () =>
    (label.parentElement!.querySelector('div.relative > div') as HTMLElement).click()
  );
  const options = [...document.querySelectorAll('[data-slot="select-option"]')].map(
    (node) => node.textContent
  );
  assert.deepEqual(options, ['Model default', 'Low', 'High', 'Maximum']);
  await click('Model default');
  await click('Save');
  assert.equal(Object.hasOwn(await send(), 'reasoning_effort'), false);
});
it('supports numeric native effort without converting it to a token count', async () => {
  model.reasoning_effort = { values: ['low', 'high', 'max'], integer_range: [1, 100] };
  await render();
  await click('Additional inputs');
  await effort('Custom value');
  await input('Custom value', '37');
  await click('Save');
  assert.equal((await send()).reasoning_effort, 37);
});
it('validates custom effort range and integer values before saving', async () => {
  model.reasoning_effort = { values: ['low', 'high', 'max'], integer_range: [1, 100] };
  await render();
  await click('Additional inputs');
  await effort('Custom value');
  for (const value of ['101', '37.5', '0']) {
    await input('Custom value', value);
    await click('Save');
    assert.ok(document.querySelector('[role="dialog"]'));
    assert.ok(document.body.textContent!.includes('Enter an integer from 1 to 100.'));
  }
  await input('Custom value', '75');
  await click('Save');
  assert.equal((await send()).reasoning_effort, 75);
});
it('shows the independently supported token budget and sends a positive numeric value', async () => {
  model.supports_thinking_budget = true;
  await render();
  await click('Additional inputs');
  await input(maxTokens, '2048');
  await input('Thinking token budget', '1024');
  await click('Save');
  const request = await send();
  assert.equal(request.thinking_budget, 1024);
  assert.equal(request.max_tokens, 2048);
  assert.equal(Object.hasOwn(request, 'reasoning_effort'), false);
});
it('requires an explicit larger output limit and lets the budget return to default', async () => {
  model.supports_thinking_budget = true;
  await render();
  await click('Additional inputs');
  await input('Thinking token budget', '1024');
  await click('Save');
  assert.ok(document.querySelector('[role="dialog"]'));
  await input(maxTokens, '1024');
  await click('Save');
  assert.ok(document.querySelector('[role="dialog"]'));
  await input(maxTokens, '2048');
  await click('Save');
  assert.equal((await send()).thinking_budget, 1024);
  await click('Additional inputs');
  await input('Thinking token budget', '');
  await click('Save');
  assert.equal(Object.hasOwn(await send(), 'thinking_budget'), false);
});
it('does not leak saved reasoning settings after the model loses those capabilities', async () => {
  model.reasoning_effort = { values: ['low', 'high', 'max'] };
  model.supports_thinking_budget = true;
  await render();
  await click('Additional inputs');
  await effort('High');
  await input(maxTokens, '2048');
  await input('Thinking token budget', '1024');
  await click('Save');
  model = { ...model, reasoning_effort: null, supports_thinking_budget: false };
  await render();
  await click('Additional inputs');
  assert.ok(!document.body.textContent!.includes('Reasoning effort'));
  assert.ok(!document.body.textContent!.includes('Thinking token budget'));
  await click('Save');
  const request = await send();
  assert.equal(Object.hasOwn(request, 'reasoning_effort'), false);
  assert.equal(Object.hasOwn(request, 'thinking_budget'), false);
});

it('uses the loaded context length for the output token limit', async () => {
  model.context_length = 40960;
  runtimeContextLength = 33604;
  await render();
  await click('Additional inputs');
  const control = document.querySelector<HTMLInputElement>(`input[aria-label="${maxTokens}"]`)!;
  assert.equal(control.max, '33604');
  await input(maxTokens, '40960');
  assert.equal(control.validity.rangeOverflow, true);
  await click('Save');
  assert.ok(document.querySelector('[role="dialog"]'));
  await input(maxTokens, '33604');
  await click('Save');
  assert.equal((await send()).max_tokens, 33604);
});
it('does not use the registered context length when the runtime limit is unknown', async () => {
  model.context_length = 40960;
  runtimeContextLength = null;
  await render();
  await click('Additional inputs');
  assert.equal(
    document.querySelector<HTMLInputElement>(`input[aria-label="${maxTokens}"]`)!.max,
    ''
  );
});
