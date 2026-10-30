/* eslint-disable @typescript-eslint/no-require-imports -- Load UI modules after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import { ModelType } from '@/constants';
import type { Root } from 'react-dom/client';
import type { FormValues } from '@/types/form';
const dom = new JSDOM('<!doctype html><html><body></body></html>', {
  url: 'http://localhost',
});
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
  if (!(key in globalThis)) {
    Object.defineProperty(globalThis, key, Object.getOwnPropertyDescriptor(dom.window, key)!);
  }
}

const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const { createForm } = require('@/hooks/use-form') as typeof import('@/hooks/use-form');
const { Form } = require('@/components/ui/form') as typeof import('@/components/ui/form');
const { FormField } =
  require('@/components/ui/form-field') as typeof import('@/components/ui/form-field');
const { I18nProvider } =
  require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const AdvancedConfig = require('./advanced-config')
  .default as typeof import('./advanced-config').default;
const { transformFormToFetch, generateCommandLineStatement } =
  require('../utils') as typeof import('../utils');
let root: Root | undefined;
afterEach(async () => {
  await act(async () => root?.unmount());
  document.body.replaceChildren();
});

for (const initial of [undefined, null, true, false]) {
  it(`preserves cache mode ${String(initial)} and submits all three choices`, async () => {
    const container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    const form = createForm();
    const values: FormValues = {
      model_engine: 'vLLM',
      model_type: 'LLM',
      enable_virtual_env: false,
    };
    if (initial !== undefined) values.enable_weight_cache = initial;
    await act(async () => {
      root!.render(
        <I18nProvider initialLocale="en">
          <Form form={form} initialValues={values}>
            <FormField hidden name="model_engine" />
            <FormField hidden name="model_type" />
            <AdvancedConfig form={form} modelType={ModelType.LLM} />
          </Form>
        </I18nProvider>
      );
    });
    // Other runtime settings also use radios; find this group by its Auto choice.
    const auto = document.querySelector<HTMLButtonElement>('[role="radio"][value="auto"]')!;
    assert.ok(auto);
    const group = auto.closest('[role="radiogroup"]')!;
    const expected = initial == null ? 'auto' : String(initial);
    assert.equal(transformFormToFetch(form.getFieldsValue()).enable_weight_cache, initial);
    assert.equal(group.querySelector('[data-state="checked"]')?.getAttribute('value'), expected);
    for (const mode of ['false', 'true', 'auto']) {
      await act(async () => {
        group.querySelector<HTMLButtonElement>(`[value="${mode}"]`)!.click();
      });
      const payload = transformFormToFetch(form.getFieldsValue());
      assert.equal(payload.enable_weight_cache, mode === 'auto' ? null : mode === 'true');
      const command = generateCommandLineStatement(payload);
      if (mode === 'auto') assert.ok(!command.includes('--enable_weight_cache'));
      else assert.ok(command.includes(`--enable_weight_cache ${mode}`));
    }
    for (const mode of ['true', 'false']) {
      const label = group.querySelector<HTMLButtonElement>(`[value="${mode}"]`)!.closest('label')!;
      await act(async () => label.click());
      assert.equal(form.getFieldValue('enable_weight_cache'), mode === 'true');
      assert.equal(form.getFieldValue('enable_virtual_env'), false);
    }
  });
}
