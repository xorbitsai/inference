import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import { runInThisContext } from 'node:vm';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { JSDOM } from 'jsdom';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const source = readFileSync(new URL('./capability-task-panel.tsx', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    jsx: ts.JsxEmit.ReactJSX,
    esModuleInterop: true,
  },
});

test('image and video requests apply the selected Lightning mode and its step defaults', () => {
  const previousCssLoader = require.extensions['.css'];
  require.extensions['.css'] = () => {};
  try {
    const { CAPABILITY_CONFIGS } = require('../capability-config.tsx');
    for (const ability of [
      'text2image',
      'image2image',
      'inpainting',
      'text2video',
      'image2video',
      'firstlastframe2video',
    ]) {
      const { transformValues } = CAPABILITY_CONFIGS[ability];
      for (const [model, enabled, expected] of [
        [{ lightning_inference_steps: 4 }, true, { use_lightning: true }],
        [
          { lightning_inference_steps: null },
          true,
          { use_lightning: true, num_inference_steps: 25 },
        ],
        [
          { lightning_inference_steps: 4 },
          false,
          { use_lightning: false, num_inference_steps: 25 },
        ],
        [{}, true, { num_inference_steps: 25 }],
      ]) {
        const request = transformValues({
          model,
          modelUid: 'video-model',
          values: { prompt: 'test', use_lightning: enabled, num_inference_steps: 25 },
        });
        const kwargs = request instanceof FormData ? request.get('kwargs') : request.kwargs;
        assert.deepEqual(JSON.parse(kwargs), expected);
      }
    }
  } finally {
    if (previousCssLoader) require.extensions['.css'] = previousCssLoader;
    else delete require.extensions['.css'];
  }
});

function mountTestDom(t) {
  const dom = new JSDOM('<div id="root"></div>');
  const previous = {};
  for (const [key, value] of Object.entries({
    window: dom.window,
    document: dom.window.document,
    Event: dom.window.Event,
    IS_REACT_ACT_ENVIRONMENT: true,
  })) {
    previous[key] = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  const root = createRoot(document.getElementById('root'));
  t.after(async () => {
    await act(async () => root.unmount());
    dom.window.close();
    for (const [key, descriptor] of Object.entries(previous)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  return { dom, root };
}

test('a custom Lightning checkpoint requires steps while Lightning is enabled', async (t) => {
  const { dom, root } = mountTestDom(t);
  const { Form } = require('@/components/ui/form');
  const { I18nProvider } = require('@/contexts/i18n-context');
  const { createForm } = require('@/hooks/use-form');
  const { TextToVideoPanel } = require('./form-panels.tsx');
  const form = createForm();
  const model = { lightning_inference_steps: null };
  const submissions = [];
  const initialValues = { prompt: 'test', use_lightning: true, num_inference_steps: '' };
  await act(async () => {
    root.render(
      React.createElement(
        I18nProvider,
        null,
        React.createElement(
          Form,
          { form, initialValues, onFinish: (values) => submissions.push(values) },
          React.createElement(TextToVideoPanel, { form, model })
        )
      )
    );
  });
  const submit = () =>
    act(async () => {
      document
        .querySelector('form')
        .dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
    });

  await submit();
  assert.equal(submissions.length, 0);
  await act(async () => form.setFieldValue('num_inference_steps', 1.5));
  await submit();
  assert.equal(submissions.length, 0);
  await act(async () => form.setFieldValue('num_inference_steps', 8));
  await submit();
  assert.equal(submissions[0].num_inference_steps, 8);
  await act(async () => {
    form.setFieldValue('num_inference_steps', '');
    form.setFieldValue('use_lightning', false);
  });
  await submit();
  assert.equal(submissions[1].use_lightning, false);
});

async function mountPanel(
  t,
  {
    stream = true,
    transformValues = () => ({ stream: true }),
    showProgress = false,
    getProgress = async () => ({ progress: 0.5 }),
  } = {}
) {
  const { dom, root } = mountTestDom(t);
  const streams = [];
  const errors = [];
  const sleeps = [];
  let resultProps;
  const mocks = {
    'lucide-react': { Copy: () => null, RotateCcw: () => null, Sparkles: () => null },
    '@/components/ui/button': {
      Button: ({ loading, disabled, type, onClick, children }) =>
        React.createElement('button', { type, onClick, disabled: disabled || loading }, children),
    },
    '@/components/ui/form': {
      Form: ({ onFinish, children }) =>
        React.createElement(
          'form',
          {
            onSubmit: (event) => {
              event.preventDefault();
              onFinish();
            },
          },
          children
        ),
    },
    '@/constants': {
      ModelAbility: { Generate: 'generate' },
      ModelType: { World: 'world' },
      RequestEvents: {},
    },
    '@/contexts/i18n-context': {
      useI18n: () => ({
        t: (key) => (key === 'runningModels.detail.latency' ? 'Latency' : key),
      }),
    },
    '@/hooks/use-form': { createForm: () => ({ getFieldsValue: () => ({}), resetFields() {} }) },
    '@/lib/request': {
      __esModule: true,
      default: { post: () => new Promise(() => {}), get: getProgress },
    },
    '@/lib/eventStream': {
      EventStreamController: class {
        signal = new AbortController();
        terminate() {
          this.signal.abort();
        }
        getSignal() {
          return this.signal.signal;
        }
      },
      postEventStreamFetcher: ({ options }, controller) => {
        streams.push({ options, controller });
      },
    },
    '@/lib/event-bus': { eventBus: { emit: (...args) => errors.push(args) } },
    '@/lib/utils': {
      cn: () => '',
      copyToClipboard() {},
      sleep: () => new Promise((resolve) => sleeps.push(resolve)),
    },
    '@/lib/is': { isNumber: (value) => typeof value === 'number' },
    '../audio-stream': {},
    '../utils': {
      createId: () => 'test',
      stringValue: (value) => value ?? '',
      booleanValue: Boolean,
    },
  };
  const compiledModule = { exports: {} };
  runInThisContext(`(function(require, module, exports) { ${outputText}\n})`)(
    (name) => mocks[name] ?? require(name),
    compiledModule,
    compiledModule.exports
  );
  await act(async () =>
    root.render(
      React.createElement(compiledModule.exports.default, {
        model: { model_name: 'test' },
        modelUid: 'test',
        config: {
          ability: 'generate',
          stream,
          showProgress,
          transformValues,
          icon: () => null,
          formPanel: ({ actions }) => actions,
          resultPanel: (props) => {
            resultProps = props;
            return null;
          },
        },
      })
    )
  );
  return {
    streams,
    errors,
    result: () => resultProps,
    poll: () => act(async () => sleeps.shift()()),
    reset: () => document.querySelector('button[type="button"]'),
    submit: () =>
      act(async () => {
        document
          .querySelector('form')
          .dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
      }),
    unmount: () => act(async () => root.unmount()),
  };
}

test('generation polls pass backend elapsed and ETA to the result panel', async (t) => {
  let fail = false;
  const panel = await mountPanel(t, {
    stream: false,
    showProgress: true,
    getProgress: async () => {
      if (fail) throw new Error('temporarily unavailable');
      return { progress: 0.5, elapsed_seconds: 12, eta_seconds: 6 };
    },
  });
  await panel.submit();
  assert.equal(panel.result().progressTiming.eta_seconds, null);
  await panel.poll();
  assert.equal(panel.result().progress, 50);
  assert.deepEqual(panel.result().progressTiming, {
    elapsed_seconds: 12,
    eta_seconds: 6,
  });
  fail = true;
  await panel.poll();
  assert.equal(panel.result().progressTiming.eta_seconds, null);
  assert.equal(panel.result().progress, 50);
  await act(async () => panel.reset().click());
  assert.equal(panel.result().progressTiming, undefined);
});

test('a late progress response cannot restore timing after reset', async (t) => {
  let resolve;
  const response = new Promise((done) => {
    resolve = done;
  });
  const panel = await mountPanel(t, {
    stream: false,
    showProgress: true,
    getProgress: () => response,
  });
  await panel.submit();
  await panel.poll();
  await act(async () => panel.reset().click());
  await act(async () => resolve({ progress: 0.8, elapsed_seconds: 12, eta_seconds: 3 }));
  assert.equal(panel.result().progress, undefined);
  assert.equal(panel.result().progressTiming, undefined);
});

test('Reset aborts a completion before its first token and ignores late callbacks', async (t) => {
  const panel = await mountPanel(t);
  await panel.submit();
  assert.equal(panel.streams.length, 1);
  assert.equal(panel.result().loading, true);
  assert.equal(panel.result().result, undefined);
  assert.equal(panel.reset().disabled, false);
  const { controller, options } = panel.streams[0];
  await act(async () => panel.reset().click());
  assert.equal(controller.getSignal().aborted, true);
  await act(async () => {
    options.onData({ choices: [{ text: 'late token' }] });
    options.onError('late error');
    options.onEnd();
  });
  assert.equal(panel.result().loading, false);
  assert.equal(panel.result().result, undefined);
  assert.deepEqual(panel.errors, []);
});

test('a controller created after async input preparation enables Reset before any token', async (t) => {
  let finishTransform;
  const panel = await mountPanel(t, {
    transformValues: () =>
      new Promise((resolve) => {
        finishTransform = resolve;
      }),
  });
  await panel.submit();
  assert.equal(panel.streams.length, 0);
  await act(async () => finishTransform({ stream: true }));
  assert.equal(panel.streams.length, 1);
  assert.equal(panel.reset().disabled, false);
  await act(async () => panel.reset().click());
  assert.equal(panel.streams[0].controller.getSignal().aborted, true);
});

test('unmount while preparing a completion does not start an orphan stream', async (t) => {
  let finishTransform;
  const panel = await mountPanel(t, {
    transformValues: () =>
      new Promise((resolve) => {
        finishTransform = resolve;
      }),
  });
  await panel.submit();
  await panel.unmount();
  await act(async () => finishTransform({ stream: true }));
  assert.equal(panel.streams.length, 0);
});

test('non-streaming requests without progress keep Reset disabled while loading', async (t) => {
  const panel = await mountPanel(t, { stream: false });
  await panel.submit();
  assert.equal(panel.reset().disabled, true);
});

test('ending an old stream does not disable Reset for a new stream', async (t) => {
  const panel = await mountPanel(t);
  await panel.submit();
  const oldStream = panel.streams[0];
  await act(async () => panel.reset().click());
  await panel.submit();
  await act(async () => {
    oldStream.options.onData({ choices: [{ text: 'stale' }] });
    oldStream.options.onEnd();
  });
  assert.equal(panel.result().loading, true);
  assert.equal(panel.result().result, undefined);
  assert.equal(panel.reset().disabled, false);
  await act(async () => {
    panel.streams[1].options.onData({ choices: [{ text: 'new' }] });
  });
  assert.equal(panel.result().result.choices[0].text, 'new');
  await panel.unmount();
  assert.equal(panel.streams[1].controller.getSignal().aborted, true);
});

test('latency is shown only for a successful completion and cleared by Reset', async (t) => {
  const panel = await mountPanel(t);
  await panel.submit();
  await act(async () => {
    panel.streams[0].options.onData({ choices: [{ text: 'complete' }] });
    panel.streams[0].options.onEnd();
  });
  assert.equal(panel.result().loading, false);
  assert.match(document.body.textContent, /Latency/);
  await act(async () => panel.reset().click());
  assert.doesNotMatch(document.body.textContent, /Latency/);
  await panel.submit();
  await act(async () => {
    panel.streams[1].options.onError('failed');
    panel.streams[1].options.onEnd();
  });
  assert.equal(panel.result().loading, false);
  assert.doesNotMatch(document.body.textContent, /Latency/);
});
