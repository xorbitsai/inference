/* eslint-disable @typescript-eslint/no-require-imports -- Load UI modules after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import { ModelType } from '@/constants';
import type { Root } from 'react-dom/client';
import type { ModelEngine, ModelEngineItem } from '@/types/services';
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
const { createForm, useFormValues } =
  require('@/hooks/use-form') as typeof import('@/hooks/use-form');
const { Form } = require('@/components/ui/form') as typeof import('@/components/ui/form');
const { FormField } =
  require('@/components/ui/form-field') as typeof import('@/components/ui/form-field');
const { I18nProvider } =
  require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const { createModelDefaultsController, selectEngineDefaults } =
  require('./model-defaults') as typeof import('./model-defaults');
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
const { useModelDefaults } = require('./model-defaults') as typeof import('./model-defaults');
const AdvancedConfig = require('./advanced-config')
  .default as typeof import('./advanced-config').default;
const RuntimePrecisionField = require('./runtime-precision-field')
  .default as typeof import('./runtime-precision-field').default;
const { Switch } = require('@/components/ui/switch') as typeof import('@/components/ui/switch');
const fields = new Set(['runtime_precision', 'cpu_offload']);
const diffusers: ModelEngineItem = {
  model_name: 'example',
  model_format: 'diffusers',
  model_size_in_billions: '',
  runtime_precisions: ['bf16', 'fp8', 'int4'],
  default_runtime_precision: 'int4',
  default_model_config: { quantization: 'int4', group_offload: true, torch_dtype: 'bfloat16' },
};
const native: ModelEngineItem = {
  ...diffusers,
  default_runtime_precision: 'bf16',
  runtime_precisions: ['bf16', 'fp8'],
  default_model_config: { task_type: 'video', trust_remote_code: true },
};
function setup(values: FormValues = {}) {
  const form = createForm();
  form.initialValues.current = { cpu_offload: false };
  form.setFieldsValue({
    model_engine: 'diffusers',
    model_format: 'diffusers',
    quantization: 'none',
    ...values,
  });
  return { form, controller: createModelDefaultsController(form) };
}
it('fills precision and extra parameters without replacing the checkpoint selector', () => {
  const { form, controller } = setup();
  controller.sync('diffusers', diffusers, fields);
  const payload = transformFormToFetch(form.getFieldsValue());
  assert.equal(payload.runtime_precision, 'int4');
  assert.equal(payload.quantization, 'none');
  assert.equal(payload.group_offload, true);
  assert.equal(payload.torch_dtype, 'bfloat16');
  assert.ok(!('model_default_key' in payload));
});
it('replaces automatic defaults when switching engines', () => {
  const { form, controller } = setup();
  controller.sync('diffusers', diffusers, fields);
  controller.sync('native', native, fields);
  const payload = transformFormToFetch(form.getFieldsValue());
  assert.equal(payload.runtime_precision, 'bf16');
  assert.equal(payload.task_type, 'video');
  assert.equal(payload.trust_remote_code, true);
  assert.ok(!('group_offload' in payload));
  assert.ok(!('torch_dtype' in payload));
});
it('keeps user edits, including explicitly choosing the same value as a default', () => {
  const { form, controller } = setup();
  controller.sync('diffusers', diffusers, fields);
  controller.userChanged('runtime_precision');
  const index = form
    .getFieldValue('kwargs')
    .findIndex((row: { key: string }) => row.key === 'group_offload');
  form.setFieldValue(['kwargs', index, 'value'], 'false');
  controller.userChanged(['kwargs', index, 'value']);
  controller.sync('native', native, fields);
  const payload = transformFormToFetch(form.getFieldsValue());
  assert.equal(payload.runtime_precision, 'int4');
  assert.equal(payload.group_offload, false);
});
it('keeps a deliberately removed default row removed', () => {
  const { form, controller } = setup();
  controller.sync('diffusers', diffusers, fields);
  form.setFieldValue(
    'kwargs',
    form.getFieldValue('kwargs').filter((row: { key: string }) => row.key !== 'group_offload')
  );
  controller.userChanged('kwargs');
  controller.sync('diffusers', diffusers, fields);
  assert.ok(!('group_offload' in transformFormToFetch(form.getFieldsValue())));
});
it('preserves history values equal to generic form defaults and refills after reset', () => {
  const { form, controller } = setup();
  const spec = { ...diffusers, default_model_config: { cpu_offload: true } };
  controller.sync('engine', spec, fields);
  assert.equal(form.getFieldValue('cpu_offload'), true);
  form.resetFields();
  controller.reset();
  const saved = transformFetchToForm({ cpu_offload: false, runtime_precision: 'fp8' });
  controller.snapshotApplied(saved);
  form.setFieldsValue(saved);
  controller.sync('engine', spec, fields);
  assert.equal(form.getFieldValue('cpu_offload'), false);
  assert.equal(form.getFieldValue('runtime_precision'), 'fp8');
  form.resetFields();
  controller.reset();
  controller.sync('engine', spec, fields);
  assert.equal(form.getFieldValue('cpu_offload'), true);
});
it('moves advanced overrides into their visible controls', () => {
  const { form, controller } = setup({ kwargs: [{ key: 'cpu_offload', value: 'false' }] });
  controller.sync('engine', { ...diffusers, default_model_config: { cpu_offload: true } }, fields);
  assert.equal(form.getFieldValue('cpu_offload'), false);
  assert.ok(
    !form.getFieldValue('kwargs')?.some((row: { key: string }) => row.key === 'cpu_offload')
  );
});
it('recognizes an existing legacy precision override', () => {
  const { form, controller } = setup({ quantization_config: 'fp8' });
  controller.sync('engine', diffusers, fields);
  assert.equal(transformFormToFetch(form.getFieldsValue()).runtime_precision, 'fp8');
});
it('preserves JSON/default types through requests, history and CLI import, and rejects malformed JSON', () => {
  const defaults = {
    vad_kwargs: { max_single_segment_time: 30000 },
    layers: [1, 2],
    enabled: false,
    optional: null,
    literal: 'false',
  };
  const spec = { ...diffusers, default_model_config: defaults };
  const { form, controller } = setup();
  controller.sync('engine', spec, fields);
  const payload = transformFormToFetch(form.getFieldsValue());
  for (const [key, value] of Object.entries(defaults)) assert.deepEqual(payload[key], value);
  assert.deepEqual(transformFormToFetch(transformFetchToForm(payload)), payload);
  const imported = transformFetchToForm(
    parseXinferenceCommand(generateCommandLineStatement(payload))
  );
  const next = setup(imported);
  next.controller.snapshotApplied(imported);
  next.controller.sync('engine', spec, fields);
  assert.deepEqual(
    transformFormToFetch(next.form.getFieldsValue()).vad_kwargs,
    defaults.vad_kwargs
  );
  const index = form
    .getFieldValue('kwargs')
    .findIndex((row: { key: string }) => row.key === 'vad_kwargs');
  form.setFieldValue(['kwargs', index, 'value'], '{');
  assert.throws(() => transformFormToFetch(form.getFieldsValue()), /valid JSON/);
});
it('does not guess between ambiguous variants', () => {
  const engines: ModelEngine = {
    diffusers: [diffusers],
    Native: [native],
    ambiguous: [diffusers, native],
  };
  assert.equal(
    selectEngineDefaults(engines, { model_engine: 'diffusers', model_format: 'diffusers' }),
    diffusers
  );
  assert.equal(selectEngineDefaults(engines, { model_engine: 'missing' }), undefined);
  assert.equal(selectEngineDefaults(engines, { model_engine: 'ambiguous' }), undefined);
});
it('fills actual controls and preserves switch edits', async () => {
  const form = createForm();
  function Harness() {
    const values = useFormValues(form),
      controller = useModelDefaults(form);
    React.useEffect(
      () =>
        controller.sync(
          'engine',
          { ...diffusers, default_model_config: { cpu_offload: true, group_offload: true } },
          fields
        ),
      [controller, values]
    );
    return (
      <Form
        form={form}
        initialValues={{ model_engine: 'diffusers', model_format: 'diffusers', cpu_offload: false }}
        onUserChange={controller.userChanged}
      >
        <FormField hidden name="model_engine" />
        <FormField hidden name="model_format" />
        <FormField name="cpu_offload" valuePropName="checked">
          <Switch />
        </FormField>
        <RuntimePrecisionField form={form} engines={{ diffusers: [diffusers] }} />
        <AdvancedConfig form={form} modelType={ModelType.Video} />
      </Form>
    );
  }
  localStorage.setItem('app_locale', 'en');
  const container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root!.render(
      <I18nProvider initialLocale="en">
        <Harness />
      </I18nProvider>
    )
  );
  assert.equal(form.getFieldValue('runtime_precision'), 'int4');
  assert.equal(document.querySelector('[role="switch"]')?.getAttribute('aria-checked'), 'true');
  assert.ok(!document.body.textContent?.includes('Model defaults'));
  await act(async () => document.querySelector<HTMLButtonElement>('[role="switch"]')!.click());
  assert.equal(form.getFieldValue('cpu_offload'), false);
  assert.equal(transformFormToFetch(form.getFieldsValue()).group_offload, true);
});
