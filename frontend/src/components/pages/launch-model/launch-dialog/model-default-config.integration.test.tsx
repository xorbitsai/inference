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
const { default: ModelDefaultConfig, selectEngineDefaults } =
  require('./model-default-config') as typeof import('./model-default-config');
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

const variants: ModelEngine = {
  diffusers: [
    {
      model_name: 'example-video',
      model_format: 'diffusers',
      model_size_in_billions: '',
      default_model_config: { torch_dtype: 'bfloat16', group_offload: true },
    },
  ],
  Native: [
    {
      model_name: 'example-video',
      model_format: 'diffusers',
      model_size_in_billions: '',
      default_model_config: { dtype: 'bfloat16', task_type: 'video' },
    },
  ],
};
function Harness({ form, engines }: { form: ReturnType<typeof createForm>; engines: ModelEngine }) {
  const values = useFormValues(form);
  const spec = selectEngineDefaults(engines, values);
  return spec ? <ModelDefaultConfig form={form} spec={spec} /> : null;
}
async function mount(values: FormValues, engines = variants) {
  localStorage.setItem('app_locale', 'en');
  const container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  const form = createForm();
  await act(async () => {
    root!.render(
      <I18nProvider initialLocale="en">
        <Form form={form} initialValues={values}>
          {Object.keys(values).map((key) => (
            <FormField key={key} hidden name={key} />
          ))}
          <Harness form={form} engines={engines} />
        </Form>
      </I18nProvider>
    );
  });
  return form;
}

it('shows declared video defaults and updates them when switching engines without changing parameters', async () => {
  const values = {
    model_type: ModelType.Video,
    model_engine: 'diffusers',
    model_format: 'diffusers',
  };
  const form = await mount(values);
  assert.ok(document.body.textContent?.includes('bfloat16'));
  assert.ok(document.body.textContent?.includes('group_offload'));
  assert.deepEqual(form.getFieldsValue(), values);
  await act(async () => form.setFieldValue('model_engine', 'Native'));
  assert.ok(document.body.textContent?.includes('task_type'));
  assert.ok(!document.body.textContent?.includes('group_offload'));
  assert.equal('default_model_config' in transformFormToFetch(form.getFieldsValue()), false);
  assert.equal('kwargs' in form.getFieldsValue(), false);
});

for (const [kind, defaults] of [
  [ModelType.Image, { quantize: true, torch_dtype: 'bfloat16' }],
  [
    ModelType.Audio,
    { vad_model: 'fsmn-vad', vad_kwargs: { max_single_segment_time: 30000 }, optional: null },
  ],
] as const) {
  it(`shows the same defaults component for ${kind}, including booleans, null and objects`, async () => {
    const engine: ModelEngine = {
      PyTorch: [
        {
          model_name: 'example',
          model_format: 'pytorch',
          model_size_in_billions: '',
          default_model_config: defaults,
        },
      ],
    };
    const form = await mount(
      { model_type: kind, model_engine: 'PyTorch', model_format: 'pytorch' },
      engine
    );
    for (const key of Object.keys(defaults)) assert.ok(document.body.textContent?.includes(key));
    assert.deepEqual(transformFormToFetch(form.getFieldsValue()), {
      model_type: kind,
      model_engine: 'PyTorch',
      model_format: 'pytorch',
    });
  });
}

it('labels actual overrides from top-level settings and additional parameters, including false', async () => {
  const form = await mount({
    model_engine: 'diffusers',
    model_format: 'diffusers',
    kwargs: [
      { key: 'group_offload', value: 'false' },
      { key: 'torch_dtype', value: 'float32' },
    ],
  });
  assert.equal(document.querySelectorAll('dd .text-primary').length, 2);
  assert.ok(document.body.textContent?.includes('false'));
  assert.ok(document.body.textContent?.includes('float32'));
  const payload = transformFormToFetch(form.getFieldsValue());
  assert.equal(payload.group_offload, false);
  const restored = transformFetchToForm(payload);
  assert.deepEqual(transformFormToFetch(restored), payload);
  const imported = parseXinferenceCommand(generateCommandLineStatement(payload));
  assert.equal(transformFormToFetch(transformFetchToForm(imported)).group_offload, false);
  await act(async () => form.setFieldValue('kwargs', [{ key: 'realtime_options', value: '{' }]));
  assert.ok(document.body.textContent?.includes('Model defaults'));
});

it('keeps checkpoint none separate from runtime defaults and marks explicit precision overrides', async () => {
  const engines: ModelEngine = {
    diffusers: [
      {
        model_name: 'example',
        model_format: 'diffusers',
        model_size_in_billions: '',
        runtime_precisions: ['bf16', 'fp8'],
        default_model_config: { quantization: 'int4', torch_dtype: 'bfloat16' },
      },
    ],
  };
  const form = await mount(
    { model_engine: 'diffusers', model_format: 'diffusers', quantization: 'none' },
    engines
  );
  assert.ok(document.body.textContent?.includes('int4'));
  assert.ok(!document.body.textContent?.includes('Current setting:'));
  await act(async () => form.setFieldValue('runtime_precision', 'bf16'));
  assert.equal(document.querySelectorAll('dd .text-primary').length, 2);
  assert.ok(document.body.textContent?.includes('Overridden by the runtime precision selection.'));
});

it('does not guess defaults for missing engines, empty metadata or ambiguous variants', async () => {
  const form = await mount({ model_engine: 'Missing', model_format: 'diffusers' });
  assert.ok(!document.body.textContent?.includes('Model defaults'));
  assert.equal(
    selectEngineDefaults(
      { empty: [{ model_name: '', model_format: '', model_size_in_billions: '' }] },
      { model_engine: 'empty' }
    )?.default_model_config,
    undefined
  );
  assert.equal(
    selectEngineDefaults(
      {
        engine: [
          ...(variants.diffusers as ModelEngineItem[]),
          ...(variants.Native as ModelEngineItem[]),
        ],
      },
      { model_engine: 'engine' }
    ),
    undefined
  );
  await act(async () => form.setFieldValue('model_engine', 'diffusers'));
  assert.ok(document.body.textContent?.includes('Model defaults'));
});
