/* eslint-disable @typescript-eslint/no-require-imports -- Load UI modules after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import { ModelType } from '@/constants';
import type { Root } from 'react-dom/client';
import type { ModelEngine } from '@/types/services';
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
const RuntimePrecisionField = require('./runtime-precision-field')
  .default as typeof import('./runtime-precision-field').default;
const {
  transformFormToFetch,
  transformFetchToForm,
  generateCommandLineStatement,
  parseXinferenceCommand,
} = require('../utils') as typeof import('../utils');
let root: Root | undefined;
afterEach(async () => {
  await act(async () => root?.unmount());
  document.body.replaceChildren();
});

const engines: ModelEngine = Object.fromEntries(
  ['diffusers', 'vLLM', 'SGLang'].map((engine) => [
    engine,
    [
      {
        model_name: 'MiniMax-H3',
        model_format: 'diffusers',
        model_size_in_billions: '',
        runtime_precisions: ['bf16', 'fp8', ...(engine === 'diffusers' ? ['int8', 'int4'] : [])],
        default_runtime_precision: engine === 'diffusers' ? 'int4' : 'bf16',
      },
    ],
  ])
);
async function mount(values: FormValues = {}, onFinish?: (values: FormValues) => void) {
  localStorage.setItem('app_locale', 'en');
  const container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  const form = createForm();
  await act(async () => {
    root!.render(
      <I18nProvider initialLocale="en">
        <Form
          form={form}
          onFinish={onFinish}
          initialValues={{
            model_name: 'MiniMax-H3',
            model_type: ModelType.Video,
            model_engine: 'diffusers',
            model_format: 'diffusers',
            ...values,
          }}
        >
          <FormField hidden name="model_engine" />
          <FormField hidden name="model_format" />
          <RuntimePrecisionField form={form} engines={engines} />
        </Form>
      </I18nProvider>
    );
  });
  return form;
}
async function choose(label: string) {
  await act(async () => {
    document.querySelector<HTMLElement>('form .h-9')!.click();
  });
  const option = [
    ...document.querySelectorAll<HTMLButtonElement>('[data-slot="select-option"]'),
  ].find((item) => item.textContent === label);
  assert.ok(option, `missing precision option ${label}`);
  assert.equal(option.disabled, false);
  await act(async () => option.click());
}

it('shows the actual engine default and preserves legacy advanced settings', async () => {
  const form = await mount();
  assert.ok(document.body.textContent?.includes('Engine default (INT4)'));
  assert.equal(transformFormToFetch(form.getFieldsValue()).runtime_precision, undefined);
  await act(async () => form.setFieldValue('model_engine', 'SGLang'));
  assert.ok(document.body.textContent?.includes('Engine default (BF16)'));
  await act(async () =>
    form.setFieldValue('kwargs', [
      { key: 'component_quantizations', value: '{"transformer":"fp8"}' },
    ])
  );
  assert.ok(document.body.textContent?.includes('Advanced configuration'));
  assert.equal(
    transformFormToFetch(form.getFieldsValue()).component_quantizations,
    '{"transformer":"fp8"}'
  );
});

for (const precision of ['bf16', 'fp8']) {
  it(`keeps ${precision} when switching engines and round-trips launch history/CLI`, async () => {
    const form = await mount();
    await choose(precision === 'bf16' ? 'BF16 (unquantized)' : 'FP8');
    for (const engine of ['vLLM', 'SGLang', 'diffusers']) {
      await act(async () => form.setFieldValue('model_engine', engine));
      const payload = transformFormToFetch(form.getFieldsValue());
      assert.equal(payload.runtime_precision, precision);
      const restored = transformFetchToForm(payload);
      assert.equal(restored.runtime_precision, precision);
      assert.ok(!restored.kwargs?.some((row: { key: string }) => row.key === 'runtime_precision'));
      const command = generateCommandLineStatement(payload);
      assert.ok(command.includes(`--runtime_precision ${precision}`));
      const parsed = parseXinferenceCommand(command);
      assert.equal(transformFormToFetch(transformFetchToForm(parsed)).runtime_precision, precision);
    }
  });
}

it('retains an unsupported choice, blocks submit, and allows correction', async () => {
  const submitted: FormValues[] = [];
  const form = await mount({}, (values) => {
    submitted.push(values);
  });
  await choose('INT4');
  await act(async () => form.setFieldValue('model_engine', 'vLLM'));
  assert.equal(form.getFieldValue('runtime_precision'), 'int4');
  assert.ok(document.body.textContent?.includes('vLLM does not support INT4'));
  async function submit() {
    await act(async () => {
      document
        .querySelector('form')!
        .dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
    });
  }
  await submit();
  assert.equal(submitted.length, 0);
  await choose('FP8');
  await submit();
  assert.equal(submitted.length, 1);
  assert.equal(submitted[0].runtime_precision, 'fp8');
});

it('restores an explicit saved precision without moving it into advanced parameters', async () => {
  const saved = { model_engine: 'SGLang', runtime_precision: 'fp8', quantization: 'none' };
  const form = await mount(transformFetchToForm(saved));
  assert.ok(document.body.textContent?.includes('FP8'));
  assert.equal(transformFormToFetch(form.getFieldsValue()).runtime_precision, 'fp8');
  await choose('Engine default (BF16)');
  assert.equal(transformFormToFetch(form.getFieldsValue()).runtime_precision, 'auto');
});
