/* eslint-disable @typescript-eslint/no-require-imports -- Load UI modules after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
import { ModelType } from '@/constants';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { FormInstance } from '@/types/form';
import type { CatalogModel } from '../types';

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
const formModule = require('@/hooks/use-form') as typeof import('@/hooks/use-form');
let form: FormInstance;
require.cache[require.resolve('@/hooks/use-form')]!.exports = {
  ...formModule,
  useForm: () => [form],
};
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = { useRouter: () => ({ push() {} }) };
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { I18nProvider } =
  require('@/contexts/i18n-context') as typeof import('@/contexts/i18n-context');
const { GlobalProvider } =
  require('@/contexts/global-context') as typeof import('@/contexts/global-context');
const { translations } = require('@/i18n/translations') as typeof import('@/i18n/translations');
const LaunchDialog = require('./launch-dialog').default as typeof import('./launch-dialog').default;

const originalGet = request.get;
const originalPost = request.post;
const originalSetInterval = globalThis.setInterval;
const originalClearInterval = globalThis.clearInterval;
const model: CatalogModel = {
  model_name: 'demo',
  model_description: '',
  abilities: ['chat'],
  languages: [],
  modelSpecs: [],
};
let root: Root;
let poll: () => Promise<void>;
let polling = false;
type ReplicaProgress = {
  replica_id: number;
  replica_model_uid: string;
  progress: number;
  stage: string;
  info: null;
  updated_at: null;
  download_files: [];
};
let progress: {
  progress: number;
  stage: string;
  download_files?: unknown[];
  replicas?: ReplicaProgress[];
};
let replicaStatuses: unknown[];
let failProgressOnce = false;
let failReplicasOnce = false;
let resolveLaunch!: (value: { model_uid: string }) => void;
let rejectLaunch!: (reason: Error) => void;
let resolveAutostart!: () => void;
let closed = false;

function rethrowWithDetails(error: unknown): never {
  if (error instanceof AggregateError) {
    throw new Error(error.errors.map((item) => item.stack || item).join('\n'));
  }
  throw error;
}

async function tick() {
  await act(async () => {
    await poll();
  });
}

async function deploy() {
  await act(async () => {
    form.setFieldsValue({
      model_engine: 'Transformers',
      model_format: 'pytorch',
      model_size_in_billions: 7,
      quantization: 'none',
    });
  });
  try {
    await act(async () => {
      document
        .querySelector('form')!
        .dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
    });
  } catch (error) {
    rethrowWithDetails(error);
  }
  assert.equal(polling, true);
}

beforeEach(async () => {
  localStorage.clear();
  form = formModule.createForm();
  progress = { progress: 0, stage: 'pending' };
  replicaStatuses = [];
  failProgressOnce = false;
  failReplicasOnce = false;
  polling = false;
  closed = false;
  globalThis.setInterval = ((callback: () => Promise<void>) => {
    poll = callback;
    polling = true;
    return 1 as unknown as ReturnType<typeof setInterval>;
  }) as typeof setInterval;
  globalThis.clearInterval = (() => {
    polling = false;
  }) as typeof clearInterval;
  request.get = (async (url: string) => {
    if (url === '/v1/models/demo/progress') {
      if (failProgressOnce) {
        failProgressOnce = false;
        throw new Error('temporary progress error');
      }
      return progress;
    }
    if (url === '/v1/models/demo/replicas') {
      if (failReplicasOnce) {
        failReplicasOnce = false;
        throw new Error('temporary replica error');
      }
      return replicaStatuses;
    }
    if (url.startsWith('/v1/engines/')) {
      return {
        Transformers: [
          { model_format: 'pytorch', model_size_in_billions: 7, quantizations: ['none'] },
        ],
      };
    }
    if (url === '/v1/cluster/system_settings') return { download_source: 'auto' };
    return [];
  }) as typeof request.get;
  request.post = ((url: string) => {
    if (url === '/v1/models') {
      return new Promise<{ model_uid: string }>((resolve, reject) => {
        resolveLaunch = resolve;
        rejectLaunch = reject;
      });
    }
    if (url === '/v1/autostart/models') {
      return new Promise<void>((resolve) => {
        resolveAutostart = resolve;
      });
    }
    return Promise.resolve({});
  }) as typeof request.post;
  const container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  try {
    await act(async () => {
      root.render(
        <GlobalProvider initClusterAuth={{ auth: false }}>
          <I18nProvider initialLocale="en">
            <LaunchDialog
              model={model}
              modelType={ModelType.LLM}
              gpuAvailable={1}
              onOpenChange={(open) => {
                closed = !open;
              }}
            />
          </I18nProvider>
        </GlobalProvider>
      );
    });
  } catch (error) {
    rethrowWithDetails(error);
  }
});

afterEach(async () => {
  await act(async () => root.unmount());
  document.body.innerHTML = '';
  request.get = originalGet;
  request.post = originalPost;
  globalThis.setInterval = originalSetInterval;
  globalThis.clearInterval = originalClearInterval;
});

it('shows preparing, downloading, and loading stages without claiming readiness early', async () => {
  await act(async () => {
    Array.from(document.querySelectorAll('label'))
      .find((label) => label.textContent?.includes('Autostart after successful launch'))!
      .querySelector<HTMLButtonElement>('[role="switch"]')!
      .click();
  });
  await deploy();
  assert.match(document.body.textContent!, /Preparing deployment/);
  assert.match(document.body.textContent!, /Waiting for replica status/);
  assert.doesNotMatch(document.body.textContent!, /launchModel\.noReplicaStatus/);

  failProgressOnce = true;
  replicaStatuses = [{ replica_id: 0, worker_address: 'worker:1234', status: 'CREATING' }];
  await tick();
  assert.equal(polling, true);
  assert.match(document.body.textContent!, /CREATING/);

  progress = { progress: 0.4, stage: 'downloading', download_files: [] };
  failReplicasOnce = true;
  await tick();
  assert.equal(polling, true);
  assert.match(document.body.textContent!, /Downloading model files/);
  assert.match(document.body.textContent!, /40%/);

  progress = { progress: 0.8, stage: 'loading' };
  await tick();
  assert.match(document.body.textContent!, /Preparing the runtime and loading the model/);
  assert.match(document.body.textContent!, /80%/);

  progress = { progress: 1, stage: 'loading' };
  await tick();
  assert.match(document.body.textContent!, /99%/);
  assert.equal(polling, true);
  assert.equal(closed, false);

  await act(async () => resolveLaunch({ model_uid: 'demo' }));
  assert.equal(polling, false);
  assert.match(document.body.textContent!, /100%/);
  assert.match(document.body.textContent!, /Model is ready/);
  assert.equal(closed, false);
  await act(async () => resolveAutostart());
  assert.equal(closed, true);
});

it('stops polling and clears progress when deployment fails', async () => {
  await deploy();
  progress = { progress: 0.8, stage: 'loading' };
  await tick();
  await act(async () => rejectLaunch(new Error('load failed')));
  assert.equal(polling, false);
  assert.doesNotMatch(document.body.textContent!, /Preparing the runtime and loading the model/);
});

it('shows the stage and progress for each replica independently', async () => {
  await deploy();
  replicaStatuses = [
    { replica_id: 0, worker_address: 'worker-a:1234', status: 'CREATING' },
    { replica_id: 1, worker_address: 'worker-b:1234', status: 'CREATING' },
  ];
  progress = {
    progress: 0.55,
    stage: 'downloading',
    replicas: [
      {
        replica_id: 0,
        replica_model_uid: 'demo-0',
        progress: 0.3,
        stage: 'downloading',
        info: null,
        updated_at: null,
        download_files: [],
      },
      {
        replica_id: 1,
        replica_model_uid: 'demo-1',
        progress: 0.8,
        stage: 'loading',
        info: null,
        updated_at: null,
        download_files: [],
      },
    ],
  };
  await tick();

  assert.match(document.body.textContent!, /Overall progress/);
  const replicaBar = (id: number) =>
    document.querySelector<HTMLElement>(`[role="progressbar"][aria-label^="Replica ${id}:"]`);
  assert.equal(replicaBar(0)?.getAttribute('aria-valuenow'), '30');
  assert.match(replicaBar(0)?.getAttribute('aria-label') ?? '', /Downloading model files/);
  assert.equal(replicaBar(1)?.getAttribute('aria-valuenow'), '80');
  assert.match(replicaBar(1)?.getAttribute('aria-label') ?? '', /loading the model/);

  replicaStatuses = [
    { replica_id: 0, worker_address: 'worker-a:1234', status: 'READY' },
    { replica_id: 1, worker_address: 'worker-b:1234', status: 'ERROR' },
  ];
  progress.replicas![0].progress = 1;
  progress.replicas![1].progress = 1;
  await tick();
  assert.equal(replicaBar(0)?.getAttribute('aria-valuenow'), '100');
  assert.match(replicaBar(0)?.getAttribute('aria-label') ?? '', /Model is ready/);
  assert.equal(replicaBar(1)?.getAttribute('aria-valuenow'), '99');
  assert.match(replicaBar(1)?.getAttribute('aria-label') ?? '', /Replica failed to start/);
  await act(async () => rejectLaunch(new Error('replica failed')));
});

it('stops polling and clears progress when deployment is cancelled', async () => {
  await deploy();
  progress = { progress: 0.8, stage: 'loading' };
  await tick();
  await act(async () => {
    Array.from(document.querySelectorAll('button'))
      .find((button) => button.textContent?.includes('Stop'))!
      .click();
  });
  assert.equal(polling, false);
  assert.doesNotMatch(document.body.textContent!, /Preparing the runtime and loading the model/);
  await act(async () => rejectLaunch(new Error('cancelled')));
});

it('provides the deployment stage and empty replica labels in every locale', () => {
  for (const locale of ['zh', 'zh-TW', 'en', 'ja', 'ko'] as const) {
    const labels = translations[locale].launchModel;
    for (const key of [
      'noReplicaStatus',
      'overallProgress',
      'stagePreparing',
      'stageDownloading',
      'stageLoading',
      'stageReady',
      'stageFailed',
    ] as const) {
      assert.ok(labels[key]);
      assert.doesNotMatch(labels[key], /^launchModel\./);
    }
  }
});
