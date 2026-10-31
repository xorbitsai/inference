/* eslint-disable @typescript-eslint/no-require-imports -- Load UI modules after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, describe, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { FormInstance, FormValues } from '@/types/form';
import type { CatalogModel } from '../types';
import { ModelType } from '@/constants';

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
const formModule = require('@/hooks/use-form') as typeof import('@/hooks/use-form');
let form: FormInstance;
require.cache[require.resolve('@/hooks/use-form')]!.exports = {
  ...formModule,
  useForm: () => [form],
};
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = { useRouter: () => ({ push() {} }) };
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { toast } = require('sonner') as typeof import('sonner');
const { I18nProvider } =
  require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const { GlobalProvider } =
  require('@/contexts/global-context') as typeof import('@/contexts/global-context');
const { translations } = require('@/i18n/translations') as typeof import('@/i18n/translations');
const {
  buildLaunchHistoryValues,
  transformFormToFetch,
  transformFetchToForm,
  generateCommandLineStatement,
  parseXinferenceCommand,
} = require('../utils') as typeof import('../utils');
const { StructuredLaunchParameterError } =
  require('../structured-launch-params') as typeof import('../structured-launch-params');
const { saveLaunchConfigHistory, readLaunchConfigHistory } =
  require('./launch-history') as typeof import('./launch-history');
const LaunchDialog = require('./launch-dialog').default as typeof import('./launch-dialog').default;

const OPTIONS = {
  optimize: true,
  first_partial_ms: 960,
  partial_interval_ms: 1500,
  max_segment_ms: 6000,
};
const JSON_OPTIONS = JSON.stringify(OPTIONS);
const model: CatalogModel = {
  model_name: 'Fun-ASR-Nano-2512',
  model_description: '',
  abilities: ['speech2text'],
  languages: ['zh'],
  modelSpecs: [],
};
const BASE_VALUES = {
  model_name: model.model_name,
  model_type: ModelType.Audio,
  model_engine: 'PyTorch',
  quantization: 'none',
  replica: 2,
  n_gpu: 'auto',
  download_hub: 'modelscope',
};

function withOptions(value: unknown): FormValues {
  return {
    ...BASE_VALUES,
    kwargs: [
      { key: 'realtime', value: 'true' },
      { key: 'realtime_options', value },
    ],
  };
}

const INVALID_INPUTS = [
  '',
  '{',
  '{"PRIVATE_PAYLOAD":',
  "{'optimize': True}",
  `'${JSON_OPTIONS}'`,
  JSON.stringify(JSON_OPTIONS),
  'null',
  '[]',
  '960',
  'true',
  '"text"',
  '[object Object]',
  '{"optimize":NaN}',
  '{"max_segment_ms":Infinity}',
];

describe('structured launch parameter conversion', () => {
  const originalPost = request.post;
  beforeEach(() => {
    localStorage.clear();
    request.post = (async () => ({})) as typeof request.post;
  });
  afterEach(() => {
    request.post = originalPost;
    localStorage.clear();
  });

  for (const value of [JSON_OPTIONS, ` \n${JSON_OPTIONS}\t `, OPTIONS, {}, '{}']) {
    it(`submits a JSON object from ${JSON.stringify(value)}`, () => {
      const input = withOptions(value);
      const original = JSON.stringify(input);
      const result = transformFormToFetch(input);
      const expected = typeof value === 'string' ? JSON.parse(value) : value;
      assert.deepEqual(result.realtime_options, expected);
      assert.equal(typeof result.realtime_options, 'object');
      assert.notEqual(result.realtime_options, value);
      assert.equal(result.realtime, true);
      assert.equal('kwargs' in result, false);
      assert.equal(JSON.stringify(input), original);
      assert.deepEqual(JSON.parse(JSON.stringify(result)).realtime_options, expected);
    });
  }

  it('preserves already structured top-level input without mutating it', () => {
    const options = Object.freeze({ ...OPTIONS });
    const input = Object.freeze({ ...BASE_VALUES, realtime: true, realtime_options: options });
    const result = transformFormToFetch(input);
    assert.deepEqual(result.realtime_options, OPTIONS);
    assert.notEqual(result.realtime_options, options);
    assert.equal(input.realtime_options, options);
  });

  for (const value of INVALID_INPUTS) {
    it(`rejects invalid structured input ${JSON.stringify(value)} without echoing it`, () => {
      assert.throws(
        () => transformFormToFetch(withOptions(value)),
        (error: unknown) => {
          assert.ok(error instanceof StructuredLaunchParameterError);
          assert.equal(error.parameter, 'realtime_options');
          assert.ok(error.message.includes('realtime_options'));
          assert.ok(!error.message.includes('PRIVATE_PAYLOAD'));
          return true;
        }
      );
    });
  }

  it('does not widen conversion rules for other parameters or environment variables', () => {
    const result = transformFormToFetch({
      ...withOptions(JSON_OPTIONS),
      kwargs: [
        ...withOptions(JSON_OPTIONS).kwargs,
        { key: 'other_json', value: JSON_OPTIONS },
        { key: 'number', value: '1500' },
        { key: 'flag', value: 'false' },
        { key: 'none_value', value: 'None' },
        { key: 'path', value: '/models/true/960' },
      ],
      envs: [{ key: 'JSON_CONFIG', value: JSON_OPTIONS }],
    });
    assert.deepEqual(result.realtime_options, OPTIONS);
    assert.equal(result.other_json, JSON_OPTIONS);
    assert.equal(result.number, 1500);
    assert.equal(result.flag, false);
    assert.equal(result.none_value, null);
    assert.equal(result.path, '/models/true/960');
    assert.deepEqual(result.envs, { JSON_CONFIG: JSON_OPTIONS });
  });

  for (const legacy of [false, true]) {
    for (const nested of [false, true]) {
      it(`restores history losslessly (legacy=${legacy}, nested=${nested})`, () => {
        const input = { realtime: true, realtime_options: legacy ? JSON_OPTIONS : OPTIONS };
        const saved = { ...BASE_VALUES, ...(nested ? { kwargs: input } : input) };
        const original = JSON.stringify(saved);
        const restored = transformFetchToForm(saved);
        const field = restored.kwargs.find(
          (row: { key: string }) => row.key === 'realtime_options'
        );
        assert.deepEqual(JSON.parse(field.value), OPTIONS);
        const payload = transformFormToFetch(restored);
        assert.deepEqual(payload.realtime_options, OPTIONS);
        assert.equal(payload.realtime, true);
        assert.equal(JSON.stringify(saved), original);
      });
    }
  }

  it('round-trips command generation, import, history saving and restoration', async () => {
    localStorage.clear();
    const formValues = withOptions(JSON_OPTIONS);
    const payload = { ...transformFormToFetch(formValues), model_uid: 'nano-options-test' };
    const command = generateCommandLineStatement(payload);
    assert.ok(command.includes(`--realtime_options '${JSON_OPTIONS}'`));
    const parsed = parseXinferenceCommand(command);
    assert.deepEqual(parsed.realtime_options, OPTIONS);
    const fromCommand = transformFormToFetch(transformFetchToForm(parsed));
    assert.deepEqual(fromCommand.realtime_options, OPTIONS);
    assert.equal(fromCommand.realtime, true);
    const history = buildLaunchHistoryValues(formValues, payload);
    assert.equal(await saveLaunchConfigHistory(history.data, false, history.uiData), true);
    const stored = readLaunchConfigHistory(false)[0];
    assert.deepEqual(stored.data.realtime_options, OPTIONS);
    const restored = transformFormToFetch(transformFetchToForm(stored.data));
    assert.deepEqual(restored.realtime_options, OPTIONS);
    assert.equal(restored.realtime, true);
    localStorage.clear();
  });

  it('rejects invalid CLI JSON without recursively decoding or removing shell quotes', () => {
    assert.throws(
      () => parseXinferenceCommand("xinference launch --realtime_options '[]'"),
      StructuredLaunchParameterError
    );
  });

  it('provides both validation messages in every maintained UI language', () => {
    for (const locale of Object.values(translations)) {
      assert.ok(locale.launchModel.structuredParameterInvalidJson.includes('{{parameter}}'));
      assert.ok(locale.launchModel.structuredParameterObjectRequired.includes('{{parameter}}'));
    }
  });
});

describe('Web UI structured parameter submission', () => {
  const originalGet = request.get;
  const originalPost = request.post;
  const originalError = toast.error;
  const originalSetInterval = globalThis.setInterval;
  const originalClearInterval = globalThis.clearInterval;
  let root: Root;
  let posts: Array<{ url: string; body: FormValues }>;
  let errors: string[];
  let resolveLaunch: (value: { model_uid: string }) => void;
  let intervals: number;
  let serverHistory: FormValues[];

  function dialog(locale: 'en' | 'zh' = 'en') {
    return (
      <GlobalProvider initClusterAuth={{ auth: false }}>
        <I18nProvider key={locale} initialLocale={locale}>
          <LaunchDialog
            model={model}
            modelType={ModelType.Audio}
            gpuAvailable={2}
            onOpenChange={() => {}}
          />
        </I18nProvider>
      </GlobalProvider>
    );
  }

  beforeEach(async () => {
    localStorage.clear();
    localStorage.setItem('app_locale', 'en');
    form = formModule.createForm();
    posts = [];
    errors = [];
    intervals = 0;
    serverHistory = [];
    globalThis.setInterval = (() => {
      intervals += 1;
      return 1 as unknown as ReturnType<typeof setInterval>;
    }) as typeof setInterval;
    globalThis.clearInterval = (() => {}) as typeof clearInterval;
    toast.error = ((message: unknown) => {
      errors.push(String(message));
      return 'test-toast';
    }) as typeof toast.error;
    request.get = (async (url: string) => {
      if (url.startsWith('/v1/engines/')) {
        return {
          PyTorch: [
            {
              model_format: 'pytorch',
              quantization: 'none',
              default_model_config: {
                vad_model: 'fsmn-vad',
                vad_kwargs: { max_single_segment_time: 30000 },
              },
            },
          ],
        };
      }
      if (url === '/v1/cluster/system_settings') return { download_source: 'auto' };
      if (url === '/v1/launch_history') return serverHistory;
      return [];
    }) as typeof request.get;
    request.post = ((url: string, body: FormValues) => {
      // Inspect exactly the JSON-serialized payload passed by the real dialog.
      posts.push({ url, body: JSON.parse(JSON.stringify(body)) });
      if (url === '/v1/models') {
        return new Promise<{ model_uid: string }>((resolve) => {
          resolveLaunch = resolve;
        });
      }
      if (url === '/v1/launch_history') {
        serverHistory = [
          { ...JSON.parse(JSON.stringify(body)), updated_at: Date.now(), is_owner: true },
        ];
      }
      return Promise.resolve({});
    }) as typeof request.post;
    const container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root.render(dialog());
    });
    await act(async () => form.setFieldsValue(BASE_VALUES));
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    document.body.innerHTML = '';
    request.get = originalGet;
    request.post = originalPost;
    toast.error = originalError;
    globalThis.setInterval = originalSetInterval;
    globalThis.clearInterval = originalClearInterval;
    localStorage.clear();
  });

  async function setOptions(value: unknown) {
    await act(async () => form.setFieldsValue(withOptions(value)));
  }

  async function submit() {
    await act(async () => {
      document
        .querySelector('form')!
        .dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
    });
  }

  it('posts an object to /v1/models and saves object-valued history', async () => {
    await act(async () => form.setFieldValue('replica', 1));
    const recommend = [...document.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
      button.textContent?.includes(translations.en.launchModel.recommendConfiguration)
    );
    assert.ok(recommend);
    assert.equal(recommend.disabled, false);
    assert.ok(
      [...document.querySelectorAll<HTMLInputElement>('input')].some(
        (input) => input.value === 'fsmn-vad'
      )
    );
    assert.ok(!document.body.textContent?.includes('Model defaults'));
    await setOptions(JSON_OPTIONS);
    await submit();
    const launches = posts.filter((row) => row.url === '/v1/models');
    assert.equal(launches.length, 1);
    assert.deepEqual(launches[0].body.realtime_options, OPTIONS);
    assert.equal(launches[0].body.realtime, true);
    assert.equal(launches[0].body.replica, 2);
    assert.equal(launches[0].body.vad_model, 'fsmn-vad');
    assert.deepEqual(launches[0].body.vad_kwargs, { max_single_segment_time: 30000 });
    assert.deepEqual(errors, []);
    await act(async () => resolveLaunch({ model_uid: 'nano-ui-options-test' }));
    const saved = readLaunchConfigHistory(false)[0];
    assert.deepEqual(saved.data.realtime_options, OPTIONS);
  });

  for (const value of INVALID_INPUTS.filter((input) => input !== '')) {
    it(`keeps rendering and sends no launch request for ${JSON.stringify(value)}`, async () => {
      await setOptions(value);
      assert.ok(document.querySelector('form'));
      assert.deepEqual(errors, []); // No errors while typing / comparing history.
      await submit();
      assert.equal(posts.length, 0);
      assert.equal(intervals, 0);
      assert.equal(errors.length, 1);
      assert.ok(errors[0].includes('realtime_options'));
      assert.ok(!errors[0].includes('PRIVATE_PAYLOAD'));
    });
  }

  it('keeps the existing required-field guard for empty parameter values', async () => {
    await setOptions('');
    await submit();
    assert.equal(posts.length, 0);
    assert.equal(intervals, 0);
  });

  it('can correct incomplete JSON and submit without reopening the dialog', async () => {
    await setOptions('{');
    await submit();
    assert.equal(posts.length, 0);
    await setOptions(JSON_OPTIONS);
    await submit();
    assert.deepEqual(posts[0].body.realtime_options, OPTIONS);
  });

  it('shows the localized validation message for Chinese users', async () => {
    await act(async () => {
      localStorage.setItem('app_locale', 'zh');
      root.render(dialog('zh'));
    });
    await setOptions('{');
    await submit();
    assert.deepEqual(errors, ['realtime_options 必须填写合法的 JSON。']);
    assert.equal(posts.length, 0);
  });

  for (const legacy of [false, true]) {
    it(`loads cached options through the real history dialog (legacy=${legacy})`, async () => {
      await saveLaunchConfigHistory(
        {
          ...BASE_VALUES,
          model_uid: 'nano-history',
          realtime: true,
          realtime_options: legacy ? JSON_OPTIONS : OPTIONS,
        },
        false
      );
      await act(async () => {
        const button = [...document.querySelectorAll('button')].find((item) =>
          item.textContent?.includes(translations.en.launchModel.configCache)
        );
        assert.ok(button);
        button.click();
      });
      await act(async () => {
        const button = [...document.querySelectorAll('button')].find(
          (item) => item.textContent === translations.en.launchModel.loadCache
        );
        assert.ok(button);
        button.click();
      });
      const row = form
        .getFieldsValue()
        .kwargs.find((item: { key: string }) => item.key === 'realtime_options');
      assert.deepEqual(JSON.parse(row.value), OPTIONS);
      await submit();
      const launch = posts.find((item) => item.url === '/v1/models');
      assert.deepEqual(launch?.body.realtime_options, OPTIONS);
    });
  }

  it('reports malformed JSON on command copy without sending a request', async () => {
    await setOptions('{');
    await act(async () => {
      const button = document.querySelector<HTMLButtonElement>(
        `button[aria-label="${translations.en.launchModel.copyToCommandLine}"]`
      );
      assert.ok(button);
      button.click();
    });
    assert.deepEqual(errors, ['realtime_options must contain valid JSON.']);
    assert.equal(posts.length, 0);
  });

  it('blocks download-only requests safely when JSON is incomplete', async () => {
    await setOptions('{');
    await act(async () => {
      const button = [...document.querySelectorAll('button')].find(
        (item) => item.textContent === translations.en.launchModel.downloadOnly
      );
      assert.ok(button);
      assert.equal(button.disabled, false);
      button.click();
    });
    assert.deepEqual(errors, ['realtime_options must contain valid JSON.']);
    assert.equal(posts.length, 0);
    assert.equal(intervals, 0);
  });
});
