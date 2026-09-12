/* eslint-disable @typescript-eslint/no-require-imports -- UI modules must load after jsdom setup. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, describe, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom does not publish declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import { ModelType } from '@/constants';
import type { FormInstance, FormValues } from '@/types/form';
import type { LaunchConfigHistoryItem } from './launch-history';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}

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
  EventTarget: dom.window.EventTarget,
  CustomEvent: dom.window.CustomEvent,
  MouseEvent: dom.window.MouseEvent,
  KeyboardEvent: dom.window.KeyboardEvent,
  FocusEvent: dom.window.FocusEvent,
  getComputedStyle: dom.window.getComputedStyle,
  requestAnimationFrame: (callback: FrameRequestCallback) => setTimeout(callback, 0),
  cancelAnimationFrame: (id: number) => clearTimeout(id),
});
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: dom.window.navigator,
});
for (const key of Reflect.ownKeys(dom.window)) {
  if (!(key in globalThis)) {
    Object.defineProperty(globalThis, key, Object.getOwnPropertyDescriptor(dom.window, key)!);
  }
}

const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const { I18nProvider } =
  require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const { GlobalProvider } =
  require('@/contexts/global-context') as typeof import('@/contexts/global-context');
const { createForm } = require('@/hooks/use-form') as typeof import('@/hooks/use-form');
const { Form } = require('@/components/ui/form') as typeof import('@/components/ui/form');
const { FormField } =
  require('@/components/ui/form-field') as typeof import('@/components/ui/form-field');
const { Input } = require('@/components/ui/input') as typeof import('@/components/ui/input');
const { Select } = require('@/components/ui/select') as typeof import('@/components/ui/select');
const { transformFormToFetch } = require('../utils') as typeof import('../utils');

let readHistory: () => LaunchConfigHistoryItem[] = () => [];
let refreshHistory: () => Promise<{
  history: LaunchConfigHistoryItem[];
  usedLocalFallback: boolean;
  syncFailed: boolean;
}> = async () => ({ history: [], usedLocalFallback: false, syncFailed: false });

const historyModulePath = require.resolve('./launch-history');
const actualHistoryModule = require(historyModulePath) as typeof import('./launch-history');
require.cache[historyModulePath]!.exports = {
  ...actualHistoryModule,
  readLaunchConfigHistory: () => readHistory(),
  refreshLaunchConfigHistory: () => refreshHistory(),
};

const ConfigCache = require('./config-cache').default as typeof import('./config-cache').default;
const { replaceLaunchHistoryFormSnapshot, useLaunchHistoryForm } =
  require('./use-launch-history-form') as typeof import('./use-launch-history-form');

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });

  return { promise, resolve };
}

function historyItem(
  modelUid: string,
  updatedAt: number,
  data: FormValues,
  options: { modelName?: string; modelType?: string } = {}
): LaunchConfigHistoryItem {
  const modelName = options.modelName ?? 'demo';
  const modelType = options.modelType ?? ModelType.LLM;
  return {
    data: {
      ...data,
      model_name: modelName,
      model_type: modelType,
      model_uid: modelUid,
    },
    model_name: modelName,
    model_uid: modelUid,
    created_by: '',
    updated_at: updatedAt,
    autostart_enabled: false,
    source: 'local',
    pending_sync: false,
    is_owner: true,
  };
}

interface HarnessProps {
  form: FormInstance;
  initialItem?: LaunchConfigHistoryItem;
  onSubmit?: (values: FormValues) => void;
}

function Harness({ form, initialItem, onSubmit }: HarnessProps) {
  const { markFormEdited, handleHistoryRefreshed } = useLaunchHistoryForm(form, true, 'demo');

  React.useLayoutEffect(() => {
    replaceLaunchHistoryFormSnapshot(form, initialItem);
  }, [form, initialItem]);

  return (
    <GlobalProvider initClusterAuth={{ auth: false }}>
      <I18nProvider initialLocale="en">
        <ConfigCache
          form={form}
          modelName="demo"
          modelType={ModelType.LLM}
          onHistoryRefreshed={handleHistoryRefreshed}
          onUserChange={markFormEdited}
        />
        <Form
          form={form}
          initialValues={{
            model_name: 'demo',
            model_type: 'LLM',
            model_engine: 'Transformers',
          }}
          onUserChange={markFormEdited}
          onFinish={(values) => onSubmit?.(transformFormToFetch(values))}
        >
          <FormField hidden name="model_name" />
          <FormField hidden name="model_type" />
          <FormField name="model_uid">
            <Input />
          </FormField>
          <FormField name="model_engine">
            <Select
              className="engine-select"
              allowClear={false}
              options={[
                { label: 'Transformers', value: 'Transformers' },
                { label: 'vLLM', value: 'vLLM' },
              ]}
            />
          </FormField>
          <FormField name="model_path">
            <Input />
          </FormField>
          <button type="submit">Launch</button>
        </Form>
      </I18nProvider>
    </GlobalProvider>
  );
}

async function click(element: Element | null) {
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Element was not found: ${document.body.textContent}`);
  }

  try {
    await act(async () => {
      element.click();
    });
  } catch (error) {
    if (error instanceof AggregateError) {
      throw new Error(error.errors.map((item) => String(item?.stack ?? item)).join('\n'));
    }
    throw error;
  }
}

async function flush() {
  await act(async () => {
    await Promise.resolve();
  });
}

describe('launch history form integration', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    document.body.innerHTML = '';
    readHistory = () => [];
    refreshHistory = async () => ({
      history: [],
      usedLocalFallback: false,
      syncFailed: false,
    });
  });

  it('keeps a custom Select choice made before history refresh completes', async () => {
    const local = historyItem('local', 1, {
      model_engine: 'Transformers',
      model_path: '/local/path',
    });
    const server = historyItem('server', 2, {
      model_engine: 'Transformers',
      model_path: '/server/path',
    });
    const refresh = deferred<{
      history: LaunchConfigHistoryItem[];
      usedLocalFallback: boolean;
      syncFailed: boolean;
    }>();
    readHistory = () => [local];
    refreshHistory = () => refresh.promise;
    const form = createForm();

    await act(async () => {
      root.render(<Harness form={form} initialItem={local} />);
    });

    await click(container.querySelector('.engine-select > div'));
    await click(document.querySelector('[data-slot="select-option"]:last-of-type'));
    assert.equal(form.getFieldValue('model_engine'), 'vLLM');

    await act(async () => {
      refresh.resolve({ history: [server], usedLocalFallback: false, syncFailed: false });
      await refresh.promise;
    });

    assert.equal(form.getFieldValue('model_engine'), 'vLLM');
  });

  it('keeps an explicitly selected older history item before refresh completes', async () => {
    const older = historyItem('older', 1, {
      model_engine: 'vLLM',
      model_path: '/older/path',
    });
    const server = historyItem('server', 3, {
      model_engine: 'Transformers',
      model_path: '/server/path',
    });
    const refresh = deferred<{
      history: LaunchConfigHistoryItem[];
      usedLocalFallback: boolean;
      syncFailed: boolean;
    }>();
    readHistory = () => [older];
    refreshHistory = () => refresh.promise;
    const form = createForm();

    await act(async () => {
      root.render(<Harness form={form} />);
    });

    await click(
      Array.from(container.querySelectorAll('button')).find((item) =>
        item.textContent?.includes('Config Cache')
      ) ?? null
    );
    await flush();
    await click(
      Array.from(document.querySelectorAll('button')).find((item) => item.textContent === 'Use') ??
        null
    );

    assert.equal(form.getFieldValue('model_uid'), 'older');
    assert.equal(form.getFieldValue('model_path'), '/older/path');

    await act(async () => {
      refresh.resolve({ history: [server], usedLocalFallback: false, syncFailed: false });
      await refresh.promise;
    });

    assert.equal(form.getFieldValue('model_uid'), 'older');
    assert.equal(form.getFieldValue('model_path'), '/older/path');
  });

  it('keeps a new configuration started before refresh completes', async () => {
    const local = historyItem('local', 1, {
      model_engine: 'vLLM',
      model_path: '/local/path',
    });
    const server = historyItem('server', 2, {
      model_engine: 'vLLM',
      model_path: '/server/path',
    });
    const refresh = deferred<{
      history: LaunchConfigHistoryItem[];
      usedLocalFallback: boolean;
      syncFailed: boolean;
    }>();
    readHistory = () => [local];
    refreshHistory = () => refresh.promise;
    const form = createForm();

    await act(async () => {
      root.render(<Harness form={form} initialItem={local} />);
    });

    await click(
      Array.from(container.querySelectorAll('button')).find((item) =>
        item.textContent?.includes('Config Cache')
      ) ?? null
    );
    await flush();
    await click(
      Array.from(document.querySelectorAll('button')).find(
        (item) => item.textContent === 'New Cache'
      ) ?? null
    );

    assert.equal(form.getFieldValue('model_uid'), undefined);
    assert.equal(form.getFieldValue('model_path'), undefined);

    await act(async () => {
      refresh.resolve({ history: [server], usedLocalFallback: false, syncFailed: false });
      await refresh.promise;
    });

    assert.equal(form.getFieldValue('model_uid'), undefined);
    assert.equal(form.getFieldValue('model_path'), undefined);
  });

  it('keeps a confirmed same-type template when history refresh completes later', async () => {
    const template = historyItem(
      'other-uid',
      1,
      {
        model_engine: 'vLLM',
        model_path: '/other/path',
      },
      { modelName: 'other-model' }
    );
    const server = historyItem('server', 2, {
      model_engine: 'Transformers',
      model_path: '/server/path',
    });
    const refresh = deferred<{
      history: LaunchConfigHistoryItem[];
      usedLocalFallback: boolean;
      syncFailed: boolean;
    }>();
    readHistory = () => [template];
    refreshHistory = () => refresh.promise;
    const form = createForm();
    const submitted: FormValues[] = [];

    await act(async () => {
      root.render(<Harness form={form} onSubmit={(values) => submitted.push(values)} />);
    });

    await click(
      Array.from(container.querySelectorAll('button')).find((item) =>
        item.textContent?.includes('Config Cache')
      ) ?? null
    );
    await flush();
    await click(
      Array.from(document.querySelectorAll('button')).find(
        (item) => item.textContent === 'Use as Template'
      ) ?? null
    );
    await flush();
    await click(
      Array.from(document.querySelectorAll('button'))
        .filter((item) => item.textContent === 'Use as Template')
        .at(-1) ?? null
    );

    assert.equal(form.getFieldValue('model_name'), 'demo');
    assert.equal(form.getFieldValue('model_type'), ModelType.LLM);
    assert.equal(form.getFieldValue('model_uid'), undefined);
    assert.equal(form.getFieldValue('model_path'), '/other/path');

    await act(async () => {
      refresh.resolve({ history: [server], usedLocalFallback: false, syncFailed: false });
      await refresh.promise;
    });

    assert.equal(form.getFieldValue('model_path'), '/other/path');

    await click(
      Array.from(container.querySelectorAll('button')).find(
        (item) => item.textContent === 'Launch'
      ) ?? null
    );
    assert.equal(submitted.length, 1);
    assert.equal(submitted[0].model_name, 'demo');
    assert.equal(submitted[0].model_type, ModelType.LLM);
    assert.equal(submitted[0].model_path, '/other/path');
  });

  it('does not offer a different-type history as a launch template', async () => {
    const audioHistory = historyItem(
      'audio-uid',
      1,
      { model_path: '/audio/path' },
      { modelName: 'whisper', modelType: ModelType.Audio }
    );
    const refresh = deferred<{
      history: LaunchConfigHistoryItem[];
      usedLocalFallback: boolean;
      syncFailed: boolean;
    }>();
    readHistory = () => [audioHistory];
    refreshHistory = () => refresh.promise;
    const form = createForm();
    const submitted: FormValues[] = [];

    await act(async () => {
      root.render(<Harness form={form} onSubmit={(values) => submitted.push(values)} />);
    });

    await click(
      Array.from(container.querySelectorAll('button')).find((item) =>
        item.textContent?.includes('Config Cache')
      ) ?? null
    );
    await flush();

    assert.equal(
      Array.from(document.querySelectorAll('button')).some(
        (item) => item.textContent === 'Use as Template'
      ),
      false
    );
    assert.equal(form.getFieldValue('model_type'), ModelType.LLM);
    assert.equal(form.getFieldValue('model_path'), undefined);

    await click(
      Array.from(container.querySelectorAll('button')).find(
        (item) => item.textContent === 'Launch'
      ) ?? null
    );
    assert.equal(submitted.length, 1);
    assert.equal(submitted[0].model_type, ModelType.LLM);
    assert.equal(Object.hasOwn(submitted[0], 'model_path'), false);
  });

  it('replaces a cached snapshot and excludes stale optional fields from launch payload', async () => {
    const local = historyItem('same', 1, {
      model_engine: 'Transformers',
      model_path: '/stale/path',
    });
    const server = historyItem('same', 2, {
      model_engine: 'vLLM',
    });
    const refresh = deferred<{
      history: LaunchConfigHistoryItem[];
      usedLocalFallback: boolean;
      syncFailed: boolean;
    }>();
    readHistory = () => [local];
    refreshHistory = () => refresh.promise;
    const form = createForm();
    const submitted: FormValues[] = [];
    const onSubmit = (values: FormValues) => submitted.push(values);

    await act(async () => {
      root.render(<Harness form={form} initialItem={local} onSubmit={onSubmit} />);
    });

    assert.equal(form.getFieldValue('model_path'), '/stale/path');

    await act(async () => {
      refresh.resolve({ history: [server], usedLocalFallback: false, syncFailed: false });
      await refresh.promise;
    });

    assert.equal(form.getFieldValue('model_engine'), 'vLLM');
    assert.equal(form.getFieldValue('model_path'), undefined);

    await click(
      Array.from(container.querySelectorAll('button')).find(
        (item) => item.textContent === 'Launch'
      ) ?? null
    );

    assert.equal(submitted.length, 1);
    assert.equal(Object.hasOwn(submitted[0], 'model_path'), false);
  });
});
