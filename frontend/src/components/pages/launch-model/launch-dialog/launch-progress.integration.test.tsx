/* eslint-disable @typescript-eslint/no-require-imports -- Load UI modules after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
import { ModelType } from '@/constants';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { FormInstance } from '@/types/form';
import type { DownloadProgressFile } from '@/types/services';
import type { Locale } from '@/types/common';
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
let navigations: string[] = [];
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = {
  useRouter: () => ({ push: (url: string) => navigations.push(url) }),
};
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
const originalDateNow = Date.now;
const model: CatalogModel = {
  model_name: 'demo',
  model_description: '',
  abilities: ['chat'],
  languages: [],
  modelSpecs: [],
};
let root: Root;
let polling = false;
let intervals: Map<number, () => void | Promise<void>>;
let nextIntervalId: number;
let now: number;
type ReplicaProgress = {
  replica_id: number;
  replica_model_uid: string;
  progress: number;
  stage: string;
  info: null;
  updated_at: null;
  download_files: DownloadProgressFile[];
  dependency_install_completed?: number;
  dependency_install_total?: number;
  dependency_install_plan?: string[];
  dependency_install_status?: 'performed' | 'skipped';
};
let progress: {
  progress: number;
  stage: string;
  download_files?: DownloadProgressFile[];
  replicas?: ReplicaProgress[];
};
let replicaStatuses: unknown[];
let failProgressOnce = false;
let failReplicasOnce = false;
let resolveLaunch!: (value: { model_uid: string }) => void;
let rejectLaunch!: (reason: Error) => void;
let resolveAutostart!: () => void;
let resolveDownload!: () => void;
let cancelRequest: Promise<unknown> | undefined;
let closed = false;

function rethrowWithDetails(error: unknown): never {
  if (error instanceof AggregateError) {
    throw new Error(error.errors.map((item) => item.stack || item).join('\n'));
  }
  throw error;
}

async function tick() {
  await act(async () => {
    for (const callback of [...intervals.values()]) await callback();
  });
}

async function renderDialog(visibleModel: CatalogModel | null = model, locale: Locale = 'en') {
  localStorage.setItem('app_locale', locale);
  await act(async () => {
    root.render(
      <GlobalProvider initClusterAuth={{ auth: false }}>
        <I18nProvider key={locale} initialLocale={locale}>
          <LaunchDialog
            model={visibleModel ?? undefined}
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
}

async function stop() {
  await act(async () => {
    Array.from(document.querySelectorAll('button'))
      .find((button) => button.textContent?.includes('Stop'))!
      .click();
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
  navigations = [];
  form = formModule.createForm();
  progress = { progress: 0, stage: 'pending' };
  replicaStatuses = [];
  failProgressOnce = false;
  failReplicasOnce = false;
  polling = false;
  intervals = new Map();
  nextIntervalId = 0;
  now = 1_000_000;
  Date.now = () => now;
  cancelRequest = undefined;
  closed = false;
  globalThis.setInterval = ((callback: () => void | Promise<void>) => {
    const id = ++nextIntervalId;
    intervals.set(id, callback);
    polling = true;
    return id as unknown as ReturnType<typeof setInterval>;
  }) as typeof setInterval;
  globalThis.clearInterval = ((id: number) => {
    intervals.delete(id);
    polling = intervals.size > 0;
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
    if (url.startsWith('/v1/cache/models/') && url.endsWith('/progress')) return progress;
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
    if (url === '/v1/cache/models') {
      return new Promise<void>((resolve) => {
        resolveDownload = resolve;
      });
    }
    if (url.endsWith('/cancel')) return cancelRequest ?? Promise.resolve({});
    return Promise.resolve({});
  }) as typeof request.post;
  const container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  try {
    await renderDialog();
  } catch (error) {
    rethrowWithDetails(error);
  }
});

afterEach(async () => {
  await act(async () => root.unmount());
  assert.equal(intervals.size, 0, 'unmount must clear both polling and elapsed-time intervals');
  document.body.innerHTML = '';
  request.get = originalGet;
  request.post = originalPost;
  globalThis.setInterval = originalSetInterval;
  globalThis.clearInterval = originalClearInterval;
  Date.now = originalDateNow;
});

it('shows replica progress without an overall launch bar', async () => {
  await act(async () => {
    Array.from(document.querySelectorAll('label'))
      .find((label) => label.textContent?.includes('Autostart after successful launch'))!
      .querySelector<HTMLButtonElement>('[role="switch"]')!
      .click();
  });
  await deploy();
  assert.match(document.body.textContent!, /Waiting for replica status/);
  assert.equal(document.querySelector('[aria-label="Overall progress"]'), null);
  assert.doesNotMatch(document.body.textContent!, /launchModel\.noReplicaStatus/);

  failProgressOnce = true;
  replicaStatuses = [{ replica_id: 0, worker_address: 'worker:1234', status: 'CREATING' }];
  await tick();
  assert.equal(polling, true);
  assert.match(document.body.textContent!, /CREATING/);

  const replicaProgress: ReplicaProgress = {
    replica_id: 0,
    replica_model_uid: 'demo-0',
    progress: 0.4,
    stage: 'downloading',
    info: null,
    updated_at: null,
    download_files: [],
  };
  progress = { progress: 0.4, stage: 'downloading', replicas: [replicaProgress] };
  failReplicasOnce = true;
  await tick();
  assert.equal(polling, true);
  assert.match(document.body.textContent!, /Downloading model files/);
  assert.match(document.body.textContent!, /40%/);

  replicaProgress.progress = 0.8;
  replicaProgress.stage = 'loading';
  progress = { progress: 0.8, stage: 'loading', replicas: [replicaProgress] };
  await tick();
  assert.match(document.body.textContent!, /Preparing the runtime and loading the model/);
  assert.match(document.body.textContent!, /80%/);

  replicaProgress.progress = 1;
  progress = { progress: 1, stage: 'loading', replicas: [replicaProgress] };
  await tick();
  assert.match(document.body.textContent!, /99%/);
  assert.equal(polling, true);
  assert.equal(closed, false);

  replicaStatuses = [{ replica_id: 0, worker_address: 'worker:1234', status: 'READY' }];
  await tick();
  assert.match(document.body.textContent!, /Deploying model, waiting for replicas to be ready/);
  assert.equal(closed, false);
  await act(async () => resolveLaunch({ model_uid: 'demo' }));
  assert.equal(polling, false);
  assert.match(document.body.textContent!, /100%/);
  assert.match(document.body.textContent!, /Model is ready/);
  assert.match(document.body.textContent!, /Model is ready, finishing deployment/);
  assert.doesNotMatch(document.body.textContent!, /Waited \d+ seconds/);
  assert.equal(
    Array.from(document.querySelectorAll('button')).find((button) => button.textContent === 'Stop')!
      .disabled,
    true
  );
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
  assert.equal(intervals.size, 0);
  assert.doesNotMatch(document.body.textContent!, /Waited \d+ seconds/);
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
        stage: 'waiting_for_dependencies',
        info: null,
        updated_at: null,
        download_files: [],
      },
    ],
  };
  await tick();

  assert.equal(document.querySelector('[aria-label="Overall progress"]'), null);
  const replicaBar = (id: number) =>
    document.querySelector<HTMLElement>(`[role="progressbar"][aria-label^="Replica ${id}:"]`);
  assert.equal(replicaBar(0)?.getAttribute('aria-valuenow'), '30');
  assert.match(replicaBar(0)?.getAttribute('aria-label') ?? '', /Downloading model files/);
  assert.equal(replicaBar(1)?.getAttribute('aria-valuenow'), '80');
  assert.match(
    replicaBar(1)?.getAttribute('aria-label') ?? '',
    /Waiting to prepare model dependencies/
  );

  const downloadFile = (name: string): DownloadProgressFile => ({
    name,
    downloaded_bytes: 50,
    total_bytes: 100,
    progress: 0.5,
    speed_bytes_per_second: 10,
    elapsed_seconds: 5,
    eta_seconds: 5,
    status: 'downloading',
  });
  progress.replicas![0].download_files = [downloadFile('worker-a-model.bin')];
  progress.replicas![1].stage = 'downloading';
  progress.replicas![1].download_files = [downloadFile('worker-b-model.bin')];
  await tick();
  const replicaCard = (id: number) =>
    document.querySelector<HTMLElement>(`article[aria-label="Replica ${id}"]`);
  assert.match(replicaCard(0)?.textContent ?? '', /worker-a-model\.bin/);
  assert.doesNotMatch(replicaCard(0)?.textContent ?? '', /worker-b-model\.bin/);
  assert.match(replicaCard(1)?.textContent ?? '', /worker-b-model\.bin/);
  assert.doesNotMatch(replicaCard(1)?.textContent ?? '', /worker-a-model\.bin/);
  assert.equal(replicaCard(0)?.querySelectorAll('[data-slot="collapsible-trigger"]').length, 1);
  assert.equal(replicaCard(1)?.querySelectorAll('[data-slot="collapsible-trigger"]').length, 1);

  progress.replicas![1].download_files = [];
  progress.replicas![1].stage = 'installing_dependencies';
  progress.replicas![1].dependency_install_completed = 2;
  progress.replicas![1].dependency_install_total = 5;
  progress.replicas![1].dependency_install_plan = ['example==2.0', 'child-pkg==3.0'];
  await tick();
  assert.match(replicaBar(1)?.getAttribute('aria-label') ?? '', /Installing model dependencies/);
  assert.match(replicaCard(1)?.textContent ?? '', /Dependencies installed: 2 \/ 5/);
  assert.match(replicaCard(1)?.textContent ?? '', /Installation plan/);
  assert.match(replicaCard(1)?.textContent ?? '', /child-pkg==3\.0/);
  assert.doesNotMatch(replicaCard(0)?.textContent ?? '', /Dependencies installed/);
  progress.replicas![1].stage = 'loading';
  await tick();
  assert.match(replicaBar(1)?.getAttribute('aria-label') ?? '', /loading the model/);
  assert.match(replicaCard(1)?.textContent ?? '', /Dependencies installed: 2 \/ 5/);

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

it('shows the dependency result when installation finishes between polls', async () => {
  await deploy();
  replicaStatuses = [{ replica_id: 0, worker_address: 'worker-a:1234', status: 'CREATING' }];
  progress = {
    progress: 0.82,
    stage: 'loading',
    replicas: [
      {
        replica_id: 0,
        replica_model_uid: 'demo-0',
        progress: 0.82,
        stage: 'loading',
        info: null,
        updated_at: null,
        download_files: [],
        dependency_install_completed: 2,
        dependency_install_total: 2,
        dependency_install_plan: ['example==2.0', 'child-pkg==3.0'],
      },
    ],
  };
  await tick();

  const replicaCard = document.querySelector<HTMLElement>('article[aria-label="Replica 0"]');
  assert.match(replicaCard?.textContent ?? '', /Dependencies installed: 2 \/ 2/);
});

it('explains when a replica reuses prepared dependencies', async () => {
  await deploy();
  replicaStatuses = [{ replica_id: 0, worker_address: 'worker-a:1234', status: 'CREATING' }];
  progress = {
    progress: 0.82,
    stage: 'loading',
    replicas: [
      {
        replica_id: 0,
        replica_model_uid: 'demo-0',
        progress: 0.82,
        stage: 'loading',
        info: null,
        updated_at: null,
        download_files: [],
        dependency_install_status: 'skipped',
      },
    ],
  };
  await tick();

  const replicaCard = document.querySelector<HTMLElement>('article[aria-label="Replica 0"]');
  assert.match(replicaCard?.textContent ?? '', /No dependency installation needed/);
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
  assert.equal(intervals.size, 0);
  assert.doesNotMatch(document.body.textContent!, /Preparing the runtime and loading the model/);
  await act(async () => rejectLaunch(new Error('cancelled')));
});

it('provides the deployment stage and empty replica labels in every locale', () => {
  for (const locale of ['zh', 'zh-TW', 'en', 'ja', 'ko'] as const) {
    const labels = translations[locale].launchModel;
    for (const key of [
      'noReplicaStatus',
      'launchWaiting',
      'launchElapsed',
      'launchWaitHint',
      'launchCanceling',
      'launchFinishing',
      'overallProgress',
      'stagePreparing',
      'stageDownloading',
      'stageWaitingDependencies',
      'stageInstallingDependencies',
      'dependencyInstallCount',
      'dependencyInstallPlan',
      'dependencyInstallSkipped',
      'stageLoading',
      'stageReady',
      'stageFailed',
    ] as const) {
      assert.ok(labels[key]);
      assert.doesNotMatch(labels[key], /^launchModel\./);
    }
  }
});

it('does not navigate or close another page when a launch completes after unmount', async () => {
  await deploy();
  await act(async () => root.unmount());
  await act(async () => resolveLaunch({ model_uid: 'demo' }));
  assert.deepEqual(navigations, []);
  assert.equal(closed, false);
  assert.equal(polling, false);
});

it('does not navigate when a background launch completes after closing its dialog', async () => {
  await deploy();
  const close = document.querySelector<HTMLButtonElement>('[data-slot="dialog-close"]')!;
  assert.ok(close);
  await act(async () => close.click());
  assert.equal(closed, true);
  assert.equal(polling, false);
  await act(async () => resolveLaunch({ model_uid: 'demo' }));
  assert.deepEqual(navigations, []);
});

it('shows waiting immediately and measures elapsed wall time independently of polling', async () => {
  assert.doesNotMatch(document.body.textContent!, /Deploying model, waiting/);
  await deploy();
  assert.match(document.body.textContent!, /Deploying model, waiting for replicas to be ready/);
  assert.match(document.body.textContent!, /Waited 0 seconds/);
  assert.match(document.body.textContent!, /Model preparation, downloading, and loading may take/);
  assert.match(document.body.textContent!, /Waiting for replica status/);

  // One refresh after a long gap must use elapsed time, not the number of callbacks.
  now += 181_900;
  failProgressOnce = true;
  failReplicasOnce = true;
  await tick();
  assert.match(document.body.textContent!, /Waited 181 seconds/);
  const elapsed = Array.from(document.querySelectorAll('span')).find((span) =>
    span.textContent?.includes('Waited 181 seconds')
  )!;
  assert.equal(elapsed.getAttribute('aria-live'), 'off');
  assert.equal(elapsed.closest('[role="status"], [aria-live="polite"]'), null);

  replicaStatuses = [{ replica_id: 0, worker_address: 'worker:1234', status: 'READY' }];
  progress = { progress: 1, stage: 'completed' };
  now += 3000;
  await tick();
  assert.match(document.body.textContent!, /Waited 184 seconds/);
  assert.match(document.body.textContent!, /Deploying model, waiting/);
  assert.equal(closed, false);
  await act(async () => resolveLaunch({ model_uid: 'demo' }));
  assert.equal(intervals.size, 0);
  assert.equal(closed, true);
});

it('resets the elapsed time when retrying after failure', async () => {
  await deploy();
  now += 60_000;
  await tick();
  assert.match(document.body.textContent!, /Waited 60 seconds/);
  await act(async () => rejectLaunch(new Error('failed')));
  assert.equal(intervals.size, 0);
  await deploy();
  assert.match(document.body.textContent!, /Waited 0 seconds/);
  now += 2500;
  await tick();
  assert.match(document.body.textContent!, /Waited 2 seconds/);
});

it('keeps the original elapsed time when cancellation fails', async () => {
  await deploy();
  let rejectCancel!: (error: Error) => void;
  cancelRequest = new Promise((_, reject) => {
    rejectCancel = reject;
  });
  now += 5000;
  await tick();
  await stop();
  assert.match(document.body.textContent!, /Canceling deployment, please wait/);
  now += 10_000;
  await tick();
  assert.match(document.body.textContent!, /Waited 15 seconds/);
  await act(async () => rejectCancel(new Error('cancel failed')));
  assert.match(document.body.textContent!, /Deploying model, waiting/);
  assert.match(document.body.textContent!, /Waited 15 seconds/);
  assert.equal(closed, false);
  assert.equal(intervals.size, 2);
});

it('ignores a canceled launch completing during the next deployment', async () => {
  await deploy();
  const oldResolve = resolveLaunch;
  await stop();
  assert.equal(intervals.size, 0);
  await deploy();
  now += 7000;
  await tick();
  await act(async () => oldResolve({ model_uid: 'demo' }));
  assert.match(document.body.textContent!, /Deploying model, waiting/);
  assert.match(document.body.textContent!, /Waited 7 seconds/);
  assert.equal(intervals.size, 2);
  assert.equal(closed, false);
  assert.deepEqual(navigations, []);
});

it('cleans up on external close and isolates old requests and timers after reopening', async () => {
  await deploy();
  const oldResolve = resolveLaunch;
  const oldTimer = [...intervals.values()][1];
  const get = request.get;
  let resolveOldProgress!: (value: unknown) => void;
  request.get = ((url: string) =>
    url === '/v1/models/demo/progress'
      ? new Promise<unknown>((resolve) => {
          resolveOldProgress = resolve;
        })
      : get(url)) as typeof request.get;
  let inFlight!: Promise<void>;
  await act(async () => {
    inFlight = Promise.resolve([...intervals.values()][0]());
  });
  await renderDialog(null);
  assert.equal(intervals.size, 0);
  request.get = get;
  now += 100_000;
  await renderDialog();
  await deploy();
  now += 4000;
  await tick();
  await act(async () => {
    await oldTimer();
    resolveOldProgress({ progress: 0.9, stage: 'failed' });
    await inFlight;
    oldResolve({ model_uid: 'demo' });
  });
  assert.match(document.body.textContent!, /Waited 4 seconds/);
  assert.match(document.body.textContent!, /Deploying model, waiting/);
  assert.match(document.body.textContent!, /Waiting for replica status/);
  assert.equal(intervals.size, 2);
  assert.equal(closed, false);
  assert.deepEqual(navigations, []);
});

it('shows only download progress for download-only operations', async () => {
  await act(async () => {
    form.setFieldsValue({
      model_engine: 'Transformers',
      model_format: 'pytorch',
      model_size_in_billions: 7,
      quantization: 'none',
    });
  });
  await act(async () => {
    Array.from(document.querySelectorAll('button'))
      .find((button) => button.textContent === translations.en.launchModel.downloadOnly)!
      .click();
  });
  progress = { progress: 0.4, stage: 'downloading' };
  now += 30_000;
  await tick();
  assert.equal(intervals.size, 1);
  assert.ok(document.querySelector('[aria-label="Overall progress"]'));
  assert.match(document.body.textContent!, /40%/);
  assert.doesNotMatch(document.body.textContent!, /Deploying model, waiting|Waited \d+ seconds/);
  await act(async () => resolveDownload());
  assert.equal(intervals.size, 0);
  assert.equal(closed, true);
});

for (const locale of ['en', 'zh', 'zh-TW', 'ja', 'ko'] as const) {
  it(`renders the deployment waiting notice and elapsed time in ${locale}`, async () => {
    await renderDialog(model, locale);
    await deploy();
    const labels = translations[locale].launchModel;
    assert.ok(document.body.textContent!.includes(labels.launchWaiting));
    assert.ok(document.body.textContent!.includes(labels.launchWaitHint));
    assert.ok(
      document.body.textContent!.includes(labels.launchElapsed.replace('{{seconds}}', '0'))
    );
    assert.doesNotMatch(document.body.textContent!, /launchModel\.launch/);
  });
}
