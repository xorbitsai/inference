/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { beforeEach, afterEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { FormInstance } from '@/types/form';
import type { RecommendationResponse } from './recommendation';
import type { CatalogModel, RequestModelType } from '../types';
import type { LaunchConfigHistoryItem } from './launch-history';

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
const historyModule = require('./launch-history') as typeof import('./launch-history');
let refreshHistory: typeof historyModule.refreshLaunchConfigHistory = async () => ({
  history: [],
  usedLocalFallback: false,
  syncFailed: false,
});
require.cache[require.resolve('./launch-history')]!.exports = {
  ...historyModule,
  refreshLaunchConfigHistory: (...args: Parameters<typeof refreshHistory>) =>
    refreshHistory(...args),
};
const LaunchDialog = require('./launch-dialog').default as typeof import('./launch-dialog').default;
const { recommendationRequest, recommendationUnavailable, recommendationPatch } =
  require('./recommendation') as typeof import('./recommendation');

const engines = {
  'llama.cpp': [{ model_format: 'ggufv2', model_size_in_billions: 7, quantizations: ['Q4_K_M'] }],
  Transformers: [
    { model_format: 'pytorch', model_size_in_billions: 7, quantizations: ['none', '4-bit'] },
  ],
  vLLM: [{ model_format: 'pytorch', model_size_in_billions: 7, quantizations: ['none', '8-bit'] }],
};
const recommended: RecommendationResponse = {
  status: 'recommended',
  config: {
    model_engine: 'vLLM',
    model_format: 'pytorch',
    model_size_in_billions: '7',
    quantization: '8-bit',
    worker_ip: 'pinned:9999',
    enable_virtual_env: false,
  },
  reasons: [{ code: 'fit', message: 'Fits available memory' }],
  warnings: [{ code: 'estimate', message: 'Memory is estimated' }],
};
const model = {
  model_name: 'demo',
  abilities: ['chat'],
  modelSpecs: [],
} as unknown as CatalogModel;
let root: Root;
let resolveResponse: (value: RecommendationResponse) => void;
let posts: { url: string; data: unknown }[];
let engineQueries: unknown[];
const originalGet = request.get;
const originalPost = request.post;

async function render(
  currentModel: CatalogModel | undefined = model,
  modelType = 'LLM',
  gpuAvailable = 4
) {
  await act(async () => {
    root.render(
      <GlobalProvider initClusterAuth={{ auth: false }}>
        <I18nProvider initialLocale="en">
          <LaunchDialog
            model={currentModel}
            modelType={modelType as RequestModelType}
            gpuAvailable={gpuAvailable}
            onOpenChange={() => {}}
          />
        </I18nProvider>
      </GlobalProvider>
    );
  });
}
function button() {
  return Array.from(document.querySelectorAll('button')).find((b) =>
    b.textContent?.includes('Recommend configuration')
  )!;
}
async function start() {
  await act(async () => {
    button().click();
  });
}
async function finish(response = recommended) {
  await act(async () => {
    resolveResponse(response);
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 5));
  });
}
beforeEach(async () => {
  localStorage.clear();
  form = formModule.createForm();
  posts = [];
  engineQueries = [];
  refreshHistory = async () => ({ history: [], usedLocalFallback: false, syncFailed: false });
  request.get = (async (url: string, config?: { params?: unknown }) => {
    if (url.startsWith('/v1/engines/')) {
      engineQueries.push(config?.params);
      return engines;
    }
    if (url === '/v1/cluster/system_settings') return { download_source: 'auto' };
    if (url === '/v1/cluster/info') return [];
    return [];
  }) as typeof request.get;
  request.post = ((url: string, data: unknown) => {
    posts.push({ url, data });
    return new Promise<RecommendationResponse>((resolve) => {
      resolveResponse = resolve;
    });
  }) as typeof request.post;
  const container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  try {
    await render();
  } catch (error) {
    if (error instanceof AggregateError)
      throw new Error(error.errors.map((item) => item.stack || item).join('\n'));
    throw error;
  }
  await act(async () => {
    form.setFieldsValue({
      model_engine: 'Transformers',
      model_format: 'pytorch',
      model_size_in_billions: 7,
      quantization: 'none',
    });
  });
});
afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  document.body.innerHTML = '';
  request.get = originalGet;
  request.post = originalPost;
});

it('converts only supported constraints and rejects malformed GPU indexes', () => {
  assert.deepEqual(
    recommendationRequest('demo', {
      n_gpu: 'CPU',
      enable_virtual_env: false,
      worker_ip: ['host:1234'],
      gpu_idx: '0, 2',
      model_engine: 'ignored',
      quantization: 'ignored',
      model_format: 'ignored',
    }),
    {
      model_name: 'demo',
      model_type: 'LLM',
      constraints: {
        n_gpu: null,
        enable_virtual_env: false,
        worker_ip: 'host:1234',
        gpu_idx: [0, 2],
      },
    }
  );
  assert.deepEqual(recommendationRequest('demo', { enable_virtual_env: 'unset' }).constraints, {});
  for (const gpu_idx of ['1,,2', '1x', '-1', '1,', ' '])
    assert.throws(() => recommendationRequest('demo', { gpu_idx }));
  assert.throws(() => recommendationRequest('demo', { n_gpu: 0 }));
});

it('applies atomically in the actual dialog, retaining linked selections and unrelated values', async () => {
  await act(async () => {
    form.setFieldsValue({
      worker_ip: ['pinned'],
      n_gpu: 2,
      gpu_idx: '0,1',
      enable_virtual_env: false,
      download_hub: 'none',
      model_uid: 'mine',
      request_limits: 8,
      envs: [{ key: 'TEST', value: 'yes' }],
    });
  });
  await start();
  await finish();
  assert.equal(posts[0].url, '/v1/models/recommend');
  assert.deepEqual((posts[0].data as { constraints: unknown }).constraints, {
    model_size_in_billions: 7,
    worker_ip: 'pinned',
    enable_virtual_env: false,
    n_gpu: 2,
    gpu_idx: [0, 1],
  });
  assert.deepEqual(engineQueries.at(-1), { enable_virtual_env: false });
  assert.equal(form.getFieldValue('model_engine'), 'vLLM');
  assert.equal(form.getFieldValue('model_size_in_billions'), 7);
  assert.equal(form.getFieldValue('quantization'), '8-bit');
  assert.deepEqual(form.getFieldValue('worker_ip'), ['pinned:9999']);
  assert.equal(form.getFieldValue('download_hub'), 'none');
  assert.equal(form.getFieldValue('model_uid'), 'mine');
  assert.equal(form.getFieldValue('gpu_idx'), '0,1');
  assert.equal(form.getFieldValue('n_gpu'), 2);
  assert.equal(form.getFieldValue('enable_virtual_env'), false);
  assert.deepEqual(form.getFieldValue('envs'), [{ key: 'TEST', value: 'yes' }]);
});

it('does not apply a stale response after an edit, even when reverted', async () => {
  await start();
  await act(async () => {
    form.setFieldValue('model_uid', 'changed');
    form.setFieldValue('model_uid', undefined);
  });
  await finish();
  assert.equal(form.getFieldValue('model_engine'), 'Transformers');
});

it('uses the effective environment when unset and renders only localized concise feedback', async () => {
  assert.equal(form.getFieldValue('enable_virtual_env'), 'unset');
  const engineQueryCount = engineQueries.length;
  await start();
  await finish({
    ...recommended,
    config: { ...recommended.config!, enable_virtual_env: true },
    reasons: [{ code: 'selection_policy', message: 'Do not display this English policy' }],
    warnings: [
      { code: 'virtual_env_not_verified', message: 'Raw setup detail' },
      { code: 'memory_not_verified', message: 'Raw memory detail' },
    ],
  });
  assert.equal(form.getFieldValue('enable_virtual_env'), true);
  assert.deepEqual(engineQueries.at(-1), { enable_virtual_env: true });
  assert.equal(engineQueries.length, engineQueryCount + 1);
  assert.equal(form.getFieldValue('model_engine'), 'vLLM');
  assert.equal(form.getFieldValue('quantization'), '8-bit');
  assert.match(document.body.textContent!, /Environment setup may be required/);
  assert.match(document.body.textContent!, /Available memory has not been verified/);
  assert.match(document.body.textContent!, /Recommended configuration applied/);
  assert.doesNotMatch(document.body.textContent!, /English policy|Raw setup|Raw memory/);
});

it('marks recommendation as user edited so delayed history cannot overwrite it', async () => {
  let resolveHistory!: (value: Awaited<ReturnType<typeof refreshHistory>>) => void;
  refreshHistory = () =>
    new Promise((resolve) => {
      resolveHistory = resolve;
    });
  await render({ ...model, model_name: 'history-demo' });
  await start();
  await finish();
  const history = {
    data: {
      model_name: 'history-demo',
      model_type: 'LLM',
      model_engine: 'Transformers',
      model_uid: 'old',
    },
    model_name: 'history-demo',
    model_uid: 'old',
    updated_at: Date.now(),
    created_by: '',
    autostart_enabled: false,
    source: 'local',
    pending_sync: false,
    is_owner: true,
  } as LaunchConfigHistoryItem;
  await act(async () => {
    resolveHistory({ history: [history], usedLocalFallback: false, syncFailed: false });
  });
  assert.equal(form.getFieldValue('model_engine'), 'vLLM');
  assert.notEqual(form.getFieldValue('model_uid'), 'old');
});

it('keeps explicit environment selection when a response violates it', async () => {
  await act(async () => {
    form.setFieldValue('enable_virtual_env', false);
  });
  await start();
  await finish({ ...recommended, config: { ...recommended.config!, enable_virtual_env: true } });
  assert.equal(form.getFieldValue('enable_virtual_env'), false);
  assert.equal(form.getFieldValue('model_engine'), 'Transformers');
});

it('handles request failure and allows retry', async () => {
  request.post = async () => {
    throw new Error('offline');
  };
  await start();
  assert.match(document.body.textContent!, /Could not apply/);
  assert.equal(button().disabled, false);
  assert.equal(form.getFieldValue('model_engine'), 'Transformers');
});

it('does not apply a stale response after a model switch or closing the dialog', async () => {
  await start();
  await render({ ...model, model_name: 'other' });
  await finish();
  assert.notEqual(form.getFieldValue('model_engine'), 'vLLM');
  await start();
  await act(async () => {
    Array.from(document.querySelectorAll('button'))
      .find((b) => b.textContent === 'Cancel')!
      .click();
  });
  await finish();
  assert.notEqual(form.getFieldValue('model_engine'), 'vLLM');
});

it('shows no recommendation and reasons without changing the form', async () => {
  await start();
  await finish({
    status: 'no_recommendation',
    config: null,
    reasons: [{ code: 'memory', message: 'Not enough memory' }],
    warnings: [],
  });
  assert.equal(form.getFieldValue('model_engine'), 'Transformers');
  assert.match(document.body.textContent!, /No suitable configuration/);
  assert.doesNotMatch(document.body.textContent!, /Not enough memory/);
});

it('rejects results absent from refreshed engine options or violating selected size', async () => {
  await start();
  await finish({ ...recommended, config: { ...recommended.config!, model_engine: 'unknown' } });
  assert.equal(form.getFieldValue('model_engine'), 'Transformers');
  assert.match(document.body.textContent!, /Could not apply/);
  await start();
  await finish({ ...recommended, config: { ...recommended.config!, model_size_in_billions: 14 } });
  assert.equal(form.getFieldValue('model_size_in_billions'), 7);
});

it('blocks llama.cpp custom engine parameters without changing the form or requesting a recommendation', async () => {
  await act(async () => {
    form.setFieldsValue({
      model_engine: 'llama.cpp',
      model_format: 'ggufv2',
      model_size_in_billions: 7,
      quantization: 'Q4_K_M',
      kwargs: [{ key: 'n_ctx', value: '4096' }],
    });
  });
  const before = structuredClone(form.getFieldsValue());
  assert.equal(before.model_engine, 'llama.cpp');
  assert.deepEqual(before.kwargs, [{ key: 'n_ctx', value: '4096' }]);
  assert.equal(recommendationUnavailable(before), true);
  assert.equal(button().disabled, true);
  assert.match(document.body.textContent!, /custom engine parameters/);
  assert.throws(() => recommendationRequest('demo', before), /unsupported/);
  await start();
  assert.equal(posts.length, 0);
  assert.deepEqual(form.getFieldsValue(), before);

  // Empty rows do not add engine arguments; named falsy values still do.
  assert.equal(recommendationUnavailable({ kwargs: [{ key: '', value: '' }] }), false);
  for (const key of ['n_ctx', 'model_engine', 'worker_ip', 'n_gpu']) {
    assert.equal(recommendationUnavailable({ kwargs: [{ key, value: '' }] }), true);
    assert.equal(recommendationUnavailable({ kwargs: [{ key, value: 0 }] }), true);
  }
});

it('disables unsafe placement and custom paths, and is absent for unsupported model types', async () => {
  for (const values of [
    { model_path: '/custom' },
    { replica: 2 },
    { replica_placement_mode: 'custom' },
    { n_worker: 2 },
    { worker_ip: ['a', 'b'] },
  ])
    assert.equal(recommendationUnavailable(values), true);
  await act(async () => {
    form.setFieldValue('model_path', '/custom');
  });
  assert.equal(button().disabled, true);
  await render(model, 'image');
  assert.ok(!button());
});

for (const modelType of ['embedding', 'rerank', 'audio']) {
  it(`recommends ${modelType} without an LLM size and preserves unrelated fields`, async () => {
    const nonLLMEngines = {
      [modelType === 'audio' ? 'MLX' : 'sentence_transformers']: [
        { model_format: modelType === 'audio' ? 'mlx' : 'pytorch', quantization: 'none' },
      ],
    };
    const engine = Object.keys(nonLLMEngines)[0];
    const urls: string[] = [];
    request.get = (async (url: string) => {
      urls.push(url);
      return url.startsWith('/v1/engines/') ? nonLLMEngines : [];
    }) as typeof request.get;
    await render(
      {
        ...model,
        modelSpecs: [{ model_format: nonLLMEngines[engine][0].model_format, quantization: 'none' }],
      } as CatalogModel,
      modelType
    );
    await act(async () => {
      form.setFieldsValue({ model_uid: 'keep-me', n_gpu: 'CPU', enable_virtual_env: false });
    });
    assert.ok(button());
    await start();
    assert.equal((posts.at(-1)!.data as { model_type: string }).model_type, modelType);
    const constraints = (posts.at(-1)!.data as { constraints: Record<string, unknown> })
      .constraints;
    assert.equal(constraints.n_gpu, null);
    assert.ok(!('model_size_in_billions' in constraints));
    await finish({
      status: 'recommended',
      config: {
        model_engine: engine,
        ...nonLLMEngines[engine][0],
        worker_ip: 'worker:9999',
        enable_virtual_env: false,
      },
      reasons: [],
      warnings: [],
    });
    assert.equal(form.getFieldValue('model_engine'), engine);
    assert.equal(form.getFieldValue('quantization'), 'none');
    assert.equal(form.getFieldValue('model_uid'), 'keep-me');
    assert.deepEqual(form.getFieldValue('worker_ip'), ['worker:9999']);
    assert.ok(urls.includes(`/v1/engines/${modelType}/demo`));
  });
}

it('accepts audio without a format but rejects unknown audio quantization', () => {
  const response: RecommendationResponse = {
    status: 'recommended',
    config: { model_engine: 'PyTorch', worker_ip: 'host:1', enable_virtual_env: false },
    reasons: [],
    warnings: [],
  };
  const catalog = {
    PyTorch: [{ model_format: null }],
  } as unknown as import('@/types/services').ModelEngine;
  const patch = recommendationPatch(response, {}, catalog, 'audio');
  assert.equal(patch?.model_engine, 'PyTorch');
  assert.equal(patch?.model_format, undefined);
  assert.ok(!('model_size_in_billions' in patch!));
  assert.throws(() =>
    recommendationPatch(
      { ...response, config: { ...response.config!, quantization: 'unknown' } },
      {},
      catalog,
      'audio',
      ['none']
    )
  );
});

it('discards a pending recommendation when the model type changes', async () => {
  await start();
  await render(model, 'embedding');
  await finish();
  assert.notEqual(form.getFieldValue('model_engine'), 'vLLM');
  assert.notDeepEqual(form.getFieldValue('worker_ip'), ['pinned:9999']);
});

for (const modelType of ['embedding', 'rerank', 'audio']) {
  it(`defaults ${modelType} to auto when GPU count is zero and honors explicit CPU`, async () => {
    await act(async () => {
      root.render(null);
    });
    form = formModule.createForm();
    const engine = modelType === 'audio' ? 'MLX' : 'sentence_transformers';
    const format = modelType === 'audio' ? 'mlx' : 'pytorch';
    request.get = (async (url: string) =>
      url.startsWith('/v1/engines/')
        ? { [engine]: [{ model_format: format, quantization: 'none' }] }
        : []) as typeof request.get;
    await render(
      { ...model, modelSpecs: [{ model_format: format, quantization: 'none' }] } as CatalogModel,
      modelType,
      0
    );
    assert.equal(form.getFieldValue('n_gpu'), 'auto');
    await start();
    assert.equal(
      (posts.at(-1)!.data as { constraints: { n_gpu: unknown } }).constraints.n_gpu,
      'auto'
    );
    await finish({
      status: 'recommended',
      config: {
        model_engine: engine,
        model_format: format,
        quantization: 'none',
        worker_ip: 'mac:9999',
        enable_virtual_env: true,
      },
      reasons: [],
      warnings: [],
    });
    assert.equal(form.getFieldValue('model_engine'), engine);
    assert.equal(form.getFieldValue('n_gpu'), 'auto');
    await act(async () => {
      form.setFieldValue('n_gpu', 'CPU');
    });
    await start();
    assert.equal(
      (posts.at(-1)!.data as { constraints: { n_gpu: unknown } }).constraints.n_gpu,
      null
    );
    await finish({ status: 'no_recommendation', config: null, reasons: [], warnings: [] });
    assert.equal(form.getFieldValue('n_gpu'), 'CPU');
  });
}

it('restores auto and legacy GPU history without converting either to CPU', () => {
  const { transformFetchToForm } = require('../utils') as typeof import('../utils');
  for (const model_type of ['embedding', 'rerank', 'audio']) {
    for (const n_gpu of ['auto', 'GPU']) {
      assert.equal(transformFetchToForm({ model_type, n_gpu }).n_gpu, 'auto');
    }
    assert.equal(transformFetchToForm({ model_type, n_gpu: null }).n_gpu, 'CPU');
  }
});
