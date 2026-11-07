/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom declarations are not installed.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { AddReplicaRequest } from '@/types/services';
import type { Locale } from '@/types/common';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
  HTMLElement: dom.window.HTMLElement,
  Element: dom.window.Element,
  Node: dom.window.Node,
  MutationObserver: dom.window.MutationObserver,
  Event: dom.window.Event,
  CustomEvent: dom.window.CustomEvent,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
for (const key of Reflect.ownKeys(dom.window)) {
  if (!(key in globalThis))
    Object.defineProperty(globalThis, key, Object.getOwnPropertyDescriptor(dom.window, key)!);
}
require.extensions['.css'] = () => {};
const actualI18n = require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const stubI18n = () => ({ t: (key: string) => key });
const i18nStub = { useI18n: stubI18n };
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = i18nStub;
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const AddReplicaDialog = require('./add-replica-dialog')
  .default as typeof import('./add-replica-dialog').default;
const originalGet = request.get;
let root: Root | undefined;
let host: HTMLDivElement;
let posts: { body: AddReplicaRequest }[];
let updateDialog: (props: Partial<React.ComponentProps<typeof AddReplicaDialog>>) => Promise<void>;
let finishPost: () => void;
async function finishSubmission() {
  await act(async () => finishPost());
}

async function mount(
  dialog = false,
  initialProps: Partial<React.ComponentProps<typeof AddReplicaDialog>> = {}
) {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  let props = initialProps;
  function Harness() {
    const [loading, setLoading] = React.useState(false);
    finishPost = () => setLoading(false);
    return dialog ? (
      <AddReplicaDialog
        open
        onOpenChange={() => {}}
        onConfirm={(body) => {
          posts.push({ body });
          setLoading(true);
        }}
        loading={loading}
        workerOptions={[]}
        modelUid="model/one"
        modelName="demo"
        modelType="LLM"
        modelEngine="vLLM"
        currentReplicaCount={1}
        defaultDevice="GPU"
        {...props}
      />
    ) : null;
  }
  await act(async () => root!.render(<Harness />));
  updateDialog = async (next) => {
    props = { ...props, ...next };
    await act(async () => root!.render(<Harness />));
  };
}
beforeEach(() => {
  posts = [];
  request.get = async <T,>() => ({}) as T;
});
afterEach(async () => {
  if (root) await act(async () => root!.unmount());
  root = undefined;
  document.body.innerHTML = '';
  request.get = originalGet;
  i18nStub.useI18n = stubI18n;
  localStorage.clear();
});

const placementWorkers = [
  { label: 'Worker A', value: 'a:9978', gpuCount: 8 },
  { label: 'Worker B', value: 'b:9978', gpuCount: 8 },
  { label: 'CPU Worker', value: 'cpu:9978', gpuCount: 0 },
];
function placementRow(index: number) {
  const row = document.querySelector<HTMLElement>(`[data-replica-index="${index}"]`);
  assert.ok(row);
  return row;
}
async function inputValue(input: HTMLInputElement | null, value: string) {
  assert.ok(input);
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      value
    );
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
    input.dispatchEvent(new window.Event('change', { bubbles: true }));
  });
}
async function selectValue(container: Element | null, label: string) {
  assert.ok(container);
  const trigger = container.querySelector<HTMLElement>('.relative > div');
  assert.ok(trigger);
  await act(async () => trigger.click());
  const option = [
    ...document.querySelectorAll<HTMLButtonElement>('[data-slot="select-option"]'),
  ].find((element) => element.textContent?.trim() === label);
  assert.ok(option, `Missing option ${label}`);
  await act(async () => option.click());
}
async function placementMode(mode: 'simple' | 'custom') {
  const button = document.querySelector<HTMLButtonElement>(`[role="radio"][value="${mode}"]`);
  assert.ok(button);
  await act(async () => button.click());
}
async function placementCount(count: number) {
  await inputValue(document.querySelector('input[type="number"]'), String(count));
}
async function configureRow(index: number, worker: string, gpu: string, alias = '') {
  await selectValue(placementRow(index).querySelector('[data-field="worker_ip"]'), worker);
  await inputValue(placementRow(index).querySelector('input[data-field="gpu_idx"]'), gpu);
  await inputValue(placementRow(index).querySelector('input[data-field="replica_uid"]'), alias);
}
async function clickAdd() {
  const button = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (element) => element.textContent === 'runningModels.addReplica'
  );
  assert.ok(button);
  await act(async () => button.click());
}

it('submits independent Worker-local GPU lists from the real placement editor', async () => {
  await mount(true, { workerOptions: placementWorkers });
  await placementMode('custom');
  await placementCount(3);
  await configureRow(0, 'Worker A', '0,1', ' new-a ');
  await configureRow(1, 'Worker B', '0,1');
  await configureRow(2, 'Worker B', '4,5,6,7');
  assert.equal(document.querySelector('[data-field="role"]'), null);
  await clickAdd();
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].body, {
    replica: 3,
    model_engine: 'vLLM',
    replica_config: [
      { replica_uid: 'new-a', devices: [{ worker_ip: 'a:9978', n_gpu: 2, gpu_idx: [0, 1] }] },
      { devices: [{ worker_ip: 'b:9978', n_gpu: 2, gpu_idx: [0, 1] }] },
      { devices: [{ worker_ip: 'b:9978', n_gpu: 4, gpu_idx: [4, 5, 6, 7] }] },
    ],
  });
  assert.ok(
    [...document.querySelectorAll<HTMLButtonElement>('[role="radio"]')].every(
      (button) => button.disabled
    )
  );
  assert.ok(
    [...document.querySelectorAll<HTMLInputElement>('[data-replica-index] input')].every(
      (input) => input.disabled
    )
  );
  const trigger = placementRow(0).querySelector<HTMLElement>(
    '[data-field="worker_ip"] .relative > div'
  )!;
  await act(async () => trigger.click());
  assert.equal(document.querySelector('[data-slot="select-dropdown"]'), null);
  await finishSubmission();
});

it('retains row drafts across modes, preserves prefixes and does not resurrect removed rows', async () => {
  await mount(true, { workerOptions: placementWorkers });
  await placementMode('custom');
  await placementCount(2);
  await configureRow(0, 'Worker A', '0,1', 'keep');
  await configureRow(1, 'Worker B', '4,5', 'remove');
  await placementMode('simple');
  assert.equal(document.querySelector('[data-replica-index]'), null);
  await placementMode('custom');
  assert.equal(
    placementRow(1).querySelector<HTMLInputElement>('[data-field="gpu_idx"]')!.value,
    '4,5'
  );
  await placementCount(1);
  await placementCount(2);
  assert.equal(
    placementRow(0).querySelector<HTMLInputElement>('[data-field="replica_uid"]')!.value,
    'keep'
  );
  assert.equal(
    placementRow(1).querySelector<HTMLInputElement>('[data-field="replica_uid"]')!.value,
    ''
  );
  await clickAdd();
  assert.equal(posts.length, 0);
  assert.ok(document.querySelector('[role="alert"]')?.textContent?.includes('RowError_worker'));
  await updateDialog({ open: false });
  await updateDialog({ open: true });
  assert.equal(
    document.querySelector('[role="radio"][value="simple"]')?.getAttribute('data-state'),
    'checked'
  );
  await placementMode('custom');
  assert.equal(document.querySelectorAll('[data-replica-index]').length, 1);
  assert.equal(
    placementRow(0).querySelector<HTMLInputElement>('[data-field="replica_uid"]')!.value,
    ''
  );
});

it('clears GPU indexes when a Worker changes or disappears without changing the alias', async () => {
  await mount(true, { workerOptions: placementWorkers });
  await placementMode('custom');
  await configureRow(0, 'Worker A', '0,1', 'keep');
  await selectValue(placementRow(0).querySelector('[data-field="worker_ip"]'), 'Worker B');
  assert.equal(
    placementRow(0).querySelector<HTMLInputElement>('[data-field="gpu_idx"]')!.value,
    ''
  );
  await inputValue(placementRow(0).querySelector('[data-field="gpu_idx"]'), '4');
  await updateDialog({
    workerOptions: placementWorkers.filter((worker) => worker.value !== 'b:9978'),
  });
  assert.equal(
    placementRow(0).querySelector<HTMLInputElement>('[data-field="gpu_idx"]')!.value,
    ''
  );
  assert.equal(
    placementRow(0).querySelector<HTMLInputElement>('[data-field="replica_uid"]')!.value,
    'keep'
  );
  await clickAdd();
  assert.equal(posts.length, 0);
  assert.ok(document.querySelector('[role="alert"]')?.textContent?.includes('RowError_worker'));
});

it('clears stale GPU fields for CPU and submits CPU allocation to a CPU Worker', async () => {
  await mount(true, { workerOptions: placementWorkers });
  await placementMode('custom');
  await configureRow(0, 'Worker A', '0,1');
  await selectValue(document.querySelector('[data-field="device"]'), 'CPU');
  assert.equal(placementRow(0).querySelector('[data-field="gpu_idx"]'), null);
  await selectValue(document.querySelector('[data-field="device"]'), 'GPU');
  assert.equal(
    placementRow(0).querySelector<HTMLInputElement>('[data-field="gpu_idx"]')!.value,
    ''
  );
  await selectValue(document.querySelector('[data-field="device"]'), 'CPU');
  await selectValue(placementRow(0).querySelector('[data-field="worker_ip"]'), 'CPU Worker');
  await clickAdd();
  assert.deepEqual(posts[0].body.replica_config, [
    { devices: [{ worker_ip: 'cpu:9978', n_gpu: 0 }] },
  ]);
  await finishSubmission();
});

it('sends Worker automatic allocation for blank GPU and locks while submitting', async () => {
  await mount(true, { workerOptions: placementWorkers });
  await placementMode('custom');
  await configureRow(0, 'Worker B', '');
  await clickAdd();
  assert.deepEqual(posts[0].body.replica_config, [
    { devices: [{ worker_ip: 'b:9978', n_gpu: 'auto' }] },
  ]);
  assert.ok(
    [...document.querySelectorAll<HTMLButtonElement>('[role="radio"]')].every(
      (button) => button.disabled
    )
  );
  assert.ok(
    [...document.querySelectorAll<HTMLInputElement>('[data-replica-index] input')].every(
      (input) => input.disabled
    )
  );
  assert.equal(posts.length, 1);
});

for (const phase of ['unknown', 'succeeded'] as const) {
  it(`locks custom placement and preserves its draft while the batch is ${phase}`, async () => {
    await mount(true, { workerOptions: placementWorkers });
    await placementMode('custom');
    await configureRow(0, 'Worker B', '4,5', 'keep');
    await updateDialog({
      operation: {
        token: 1,
        modelUid: 'model/one',
        count: 1,
        startedAt: Date.now(),
        phase,
        baseline: [],
        baselineReady: true,
        replicas: [],
        progress: null,
        refreshFailed: false,
      },
      workerOptions: placementWorkers.filter((worker) => worker.value !== 'b:9978'),
    });
    assert.ok(
      [...document.querySelectorAll<HTMLButtonElement>('[role="radio"]')].every(
        (button) => button.disabled
      )
    );
    assert.ok(
      [...placementRow(0).querySelectorAll<HTMLInputElement>('input')].every(
        (input) => input.disabled
      )
    );
    assert.equal(
      placementRow(0).querySelector<HTMLInputElement>('[data-field="gpu_idx"]')!.value,
      '4,5'
    );
    assert.equal(
      placementRow(0).querySelector<HTMLInputElement>('[data-field="replica_uid"]')!.value,
      'keep'
    );
    const trigger = placementRow(0).querySelector<HTMLElement>(
      '[data-field="worker_ip"] .relative > div'
    )!;
    await act(async () => trigger.click());
    assert.equal(document.querySelector('[data-slot="select-dropdown"]'), null);
    assert.equal(posts.length, 0);
  });
}

for (const [gpu, alias, error] of [
  ['1,,2', '', 'gpu'],
  ['1,1', '', 'duplicateGpu'],
  ['', 'same', 'duplicateAlias'],
]) {
  it(`blocks ${error} in a later placement row before making any POST`, async () => {
    await mount(true, { workerOptions: placementWorkers });
    await placementMode('custom');
    await placementCount(2);
    await configureRow(0, 'Worker A', '0', 'same');
    await configureRow(1, 'Worker B', gpu, alias);
    await clickAdd();
    assert.equal(posts.length, 0);
    assert.ok(document.querySelector('[role="alert"]')?.textContent?.includes(`RowError_${error}`));
    assert.equal(
      placementRow(1).querySelector<HTMLInputElement>('[data-field="gpu_idx"]')!.value,
      gpu
    );
  });
}

it('does not validate a hidden custom draft when submitting simple automatic placement', async () => {
  await mount(true, { workerOptions: placementWorkers });
  await placementMode('custom');
  await inputValue(placementRow(0).querySelector('[data-field="gpu_idx"]'), 'invalid');
  await placementMode('simple');
  await clickAdd();
  assert.deepEqual(posts[0].body, { replica: 1, model_engine: 'vLLM', n_gpu: 1 });
  await finishSubmission();
});

it('keeps initial deployment roles, hidden GPU counts and form edit notifications', async () => {
  const { createForm } = require('@/hooks/use-form') as typeof import('@/hooks/use-form');
  const { Form } = require('@/components/ui/form') as typeof import('@/components/ui/form');
  const ReplicaPlacementConfig = require('../launch-model/launch-dialog/replica-placement-config')
    .default as typeof import('../launch-model/launch-dialog/replica-placement-config').default;
  const form = createForm();
  form.setFieldsValue({
    model_engine: 'vLLM',
    replica_config: [
      { replica_uid: 'prefill-a', worker_ip: 'a:9978', gpu_idx: '', n_gpu: 4, role: 'prefill' },
    ],
  });
  let edits = 0;
  await mount();
  const workers = placementWorkers.map((worker) => ({ ...worker }));
  await act(async () =>
    root!.render(
      <Form form={form} onUserChange={() => edits++}>
        <ReplicaPlacementConfig form={form} workerOptions={workers} />
      </Form>
    )
  );
  assert.ok(document.querySelector('[data-field="role"]'));
  await inputValue(placementRow(0).querySelector('[data-field="replica_uid"]'), 'prefill-new');
  assert.equal(form.getFieldValue('replica_config')[0].n_gpu, 4);
  await selectValue(placementRow(0).querySelector('[data-field="role"]'), 'Decode');
  assert.equal(form.getFieldValue('replica_config')[0].role, 'decode');
  await inputValue(placementRow(0).querySelector('[data-field="gpu_idx"]'), '0,1');
  assert.equal(form.getFieldValue('replica_config')[0].n_gpu, undefined);
  assert.equal(form.getFieldValue('replica_config')[0].gpu_idx, '0,1');
  assert.equal(edits, 3);
  await act(async () => form.setFieldsValue({ model_engine: 'Transformers' }));
  assert.equal(document.querySelector('[data-field="role"]'), null);
  assert.equal(form.getFieldValue('replica_config')[0].role, 'decode');
});

for (const locale of ['en', 'zh', 'zh-TW', 'ja', 'ko'] as Locale[]) {
  it(`renders custom placement labels, hints and row errors with real ${locale} translations`, async () => {
    const { translations } = require('@/i18n/translations') as typeof import('@/i18n/translations');
    localStorage.setItem('app_locale', locale);
    i18nStub.useI18n = actualI18n.useI18n;
    await mount();
    await act(async () =>
      root!.render(
        <actualI18n.I18nProvider initialLocale={locale}>
          <AddReplicaDialog
            open
            onOpenChange={() => {}}
            onConfirm={() => assert.fail('Invalid row submitted')}
            loading={false}
            workerOptions={placementWorkers}
            modelUid="demo"
            modelName="demo"
            modelType="LLM"
            currentReplicaCount={1}
            defaultDevice="GPU"
          />
        </actualI18n.I18nProvider>
      )
    );
    const automaticNames = {
      en: 'Automatic Scheduling',
      zh: '自动调度',
      'zh-TW': '自動調度',
      ja: '自動スケジューリング',
      ko: '자동 스케줄링',
    };
    const automatic = document.querySelector<HTMLButtonElement>('[role="radio"][value="simple"]');
    assert.ok(automatic);
    assert.equal(automatic.closest('label')?.textContent, automaticNames[locale]);
    assert.equal(automatic.getAttribute('data-state'), 'checked');
    assert.equal(
      document.querySelector('[role="radio"][value="custom"]')?.closest('label')?.textContent,
      translations[locale].launchModel.placementCustom
    );
    await placementMode('custom');
    const text = translations[locale].runningModels;
    assert.ok(document.body.textContent?.includes(text.addReplicaPlacementHint));
    const button = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (element) => element.textContent === text.addReplica
    )!;
    await act(async () => button.click());
    const alert = document.querySelector('[role="alert"]')!.textContent!;
    assert.ok(alert.includes('1'));
    assert.ok(!alert.includes('runningModels.'));
    assert.ok(!document.body.textContent?.includes('runningModels.addReplicaRow'));
    for (const code of ['count', 'worker', 'gpu', 'duplicateGpu', 'duplicateAlias'] as const) {
      assert.ok(text[`addReplicaRowError_${code}`]);
    }
  });
}

it('ignores hidden simple Worker minimum and GPU errors when submitting a custom row', async () => {
  await mount(true, { workerOptions: placementWorkers });
  const trigger = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (element) => element.textContent === 'runningModels.addReplicaAutoWorker'
  )!;
  await act(async () => trigger.click());
  for (const label of ['Worker A', 'Worker B']) {
    const option = [
      ...document.querySelectorAll<HTMLButtonElement>('[data-slot="select-dropdown"] button'),
    ].find((element) => element.textContent === label)!;
    await act(async () => option.click());
  }
  await act(async () => trigger.click());
  assert.equal(document.querySelector<HTMLInputElement>('input[type="number"]')!.value, '2');
  await inputValue(
    document.querySelector('input[placeholder="runningModels.addReplicaGpuIdxPlaceholder"]'),
    '1,,2'
  );
  await placementMode('custom');
  await placementCount(1);
  const count = document.querySelector<HTMLInputElement>('input[type="number"]')!;
  assert.equal(count.min, '1');
  await act(async () => count.dispatchEvent(new window.FocusEvent('focusout', { bubbles: true })));
  assert.equal(count.value, '1');
  await configureRow(0, 'Worker B', '3');
  await clickAdd();
  assert.deepEqual(posts[0].body, {
    replica: 1,
    model_engine: 'vLLM',
    replica_config: [{ devices: [{ worker_ip: 'b:9978', n_gpu: 1, gpu_idx: [3] }] }],
  });
  await finishSubmission();
});
