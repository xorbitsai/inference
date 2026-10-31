/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { beforeEach, afterEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
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

require('@/contexts/i18n-context');
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({
    t: (key: string, params?: Record<string, unknown>) =>
      params && (key.endsWith('.replicaProgress') || key.endsWith('.partial'))
        ? `${key} ${params?.completed}/${params?.total}`
        : key,
  }),
};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { ReloadDialog } = require('./reload-dialog') as typeof import('./reload-dialog');
let root: Root;
let host: HTMLDivElement;
let enabled = true,
  polls = 0,
  closed = 0,
  refreshed = 0;
let posts: { url: string; body: unknown }[] = [];
let imageLimits: Record<string, number> | undefined;
let hotParameters: string[];
let imageField: string;
let canUpgradeEnvironment: boolean;
interface TestReloadStatus {
  status: string;
  stage?: string;
  error?: string;
  restored?: boolean;
  outcome_unknown?: boolean;
  partial?: boolean;
  completed_replicas?: number;
  replica_count?: number;
  replicas?: { replica_uid: string; status: string; stage?: string }[];
}
let initial: TestReloadStatus = { status: 'idle', stage: 'queued' };
let final: TestReloadStatus = { status: 'ready' };
let mixedParameters: string[];
let sequenceLimit = 16;
const originalGet = request.get,
  originalPost = request.post;
beforeEach(() => {
  enabled = true;
  canUpgradeEnvironment = true;
  polls = 0;
  closed = 0;
  refreshed = 0;
  posts = [];
  imageLimits = undefined;
  mixedParameters = [];
  sequenceLimit = 16;
  imageField = 'limit_mm_per_prompt';
  hotParameters = ['reasoning_content'];
  initial = { status: 'idle', stage: 'queued' };
  final = { status: 'ready' };
  request.get = async <T,>(url: string) => {
    if (url.endsWith('/config'))
      return {
        enabled,
        can_upgrade_environment: canUpgradeEnvironment,
        mixed_parameters: mixedParameters,
        model_config: Object.fromEntries(
          Object.entries({
            max_num_seqs: sequenceLimit,
            enforce_eager: false,
            reasoning_content: false,
            ...(imageLimits ? { [imageField]: imageLimits } : {}),
          }).filter(([key]) => !mixedParameters.includes(key))
        ),
        hot_parameters: hotParameters,
        parameters: {
          max_num_seqs: 'positive_int',
          enforce_eager: 'bool',
          max_model_len: 'positive_int',
          ...(hotParameters.includes('reasoning_content') ? { reasoning_content: 'bool' } : {}),
          ...(imageLimits || imageField === 'limit_mm_data_per_request'
            ? { [imageField]: 'image_limits' }
            : {}),
        },
      } as T;
    return (polls++ === 0 ? initial : final) as T;
  };
  request.post = async <T,>(url: string, body: unknown) => {
    posts.push({ url, body });
    return { operation_id: 'one', status: 'reloading', stage: 'queued' } as T;
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  request.get = originalGet;
  request.post = originalPost;
});
async function render() {
  await act(async () =>
    root.render(
      <ReloadDialog
        open
        modelUid="model/name"
        onReady={() => {
          refreshed++;
        }}
        onOpenChange={() => {
          closed++;
        }}
      />
    )
  );
}
function applyButton() {
  return [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'runningModels.reload.apply'
  )!;
}

it('submits typed changed fields and follows the asynchronous reload', async () => {
  await render();
  assert.ok(applyButton().disabled);
  const input = document.querySelector<HTMLInputElement>('input[aria-label="max_num_seqs"]')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      '32'
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector<HTMLButtonElement>('button[aria-label="enforce_eager"]')!.click();
  });
  await act(async () => applyButton().click());
  assert.deepEqual(posts, [
    {
      url: '/v1/models/model%2Fname/reload',
      body: { model_config: { max_num_seqs: 32, enforce_eager: true } },
    },
  ]);
  assert.ok(applyButton().disabled);
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1100));
  });
  assert.equal(refreshed, 1);
  assert.equal(closed, 1);
});
it('can upgrade the model environment without changing engine parameters', async () => {
  await render();
  const environment = document.querySelector('details')!;
  assert.equal(environment.open, false);
  await act(async () => environment.querySelector('summary')!.click());
  assert.equal(environment.open, true);
  const input = document.querySelector<HTMLInputElement>(
    'input[aria-label="runningModels.reload.python"]'
  )!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      '/opt/new/bin/python'
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  assert.ok(!applyButton().disabled);
  await act(async () => applyButton().click());
  assert.deepEqual(posts[0].body, { model_config: {}, python_executable: '/opt/new/bin/python' });
});
it('groups engine and response settings with readable labels and parameter names', async () => {
  await render();
  const engine = document.querySelector(
    'section[aria-label="runningModels.reload.engineSettings"]'
  )!;
  const response = document.querySelector(
    'section[aria-label="runningModels.reload.responseSettings"]'
  )!;
  assert.match(engine.textContent || '', /runningModels.reload.fields.concurrency/);
  assert.ok(engine.querySelector('input[aria-label="max_num_seqs"]'));
  assert.ok(response.querySelector('button[aria-label="reasoning_content"]'));
  assert.equal(engine.querySelector('button[aria-label="reasoning_content"]'), null);
  assert.match(engine.textContent || '', /max_num_seqs/);
});
it('hides environment selection without administrator access', async () => {
  canUpgradeEnvironment = false;
  await render();
  assert.equal(document.querySelector('input[aria-label="runningModels.reload.python"]'), null);
  assert.ok(document.querySelector('input[aria-label="max_num_seqs"]'));
});
it('disables reload when no parameters can be updated', async () => {
  enabled = false;
  hotParameters = [];
  await render();
  assert.ok(applyButton().disabled);
  assert.match(document.body.textContent || '', /runningModels.reload.disabled/);
  assert.equal(document.querySelectorAll('input').length, 0);
  assert.equal(document.querySelector('[role="switch"]'), null);
  assert.doesNotMatch(document.body.textContent || '', /runningModels.reload.reasoningHint/);
});
it('can update reasoning parsing without launch-time weight caching', async () => {
  enabled = false;
  await render();
  assert.equal(document.querySelector('input[aria-label="max_num_seqs"]'), null);
  await act(async () => {
    document.querySelector<HTMLButtonElement>('button[aria-label="reasoning_content"]')!.click();
  });
  assert.ok(!applyButton().disabled);
  await act(async () => applyButton().click());
  assert.deepEqual(posts[0].body, { model_config: { reasoning_content: true } });
  assert.match(document.body.textContent || '', /runningModels.reload.reasoningHint/);
});
for (const field of ['limit_mm_per_prompt', 'limit_mm_data_per_request']) {
  it(`edits ${field} and leaves an unchanged object unsubmitted`, async () => {
    imageField = field;
    imageLimits = { image: 2, video: 0 };
    await render();
    assert.ok(applyButton().disabled);
    const input = document.querySelector<HTMLInputElement>(`input[aria-label="${field}"]`)!;
    assert.equal(input.value, '2');
    assert.equal(input.min, '1');
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
        input,
        '8'
      );
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => applyButton().click());
    assert.deepEqual(posts[0].body, { model_config: { [field]: { image: 8 } } });
  });
}
it('can set a SGLang image limit from the engine default', async () => {
  imageField = 'limit_mm_data_per_request';
  await render();
  const input = document.querySelector<HTMLInputElement>(
    'input[aria-label="limit_mm_data_per_request"]'
  )!;
  assert.equal(input.value, '');
  assert.ok(applyButton().disabled);
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      '2'
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await act(async () => applyButton().click());
  assert.deepEqual(posts[0].body, { model_config: { limit_mm_data_per_request: { image: 2 } } });
});
it('hides the no-restart hint when a cached reload mixes hot and engine parameters', async () => {
  imageLimits = { image: 2, video: 0 };
  await render();
  await act(async () => {
    document.querySelector<HTMLButtonElement>('button[aria-label="reasoning_content"]')!.click();
  });
  assert.match(document.body.textContent || '', /runningModels.reload.reasoningHint/);
  const input = document.querySelector<HTMLInputElement>(
    'input[aria-label="limit_mm_per_prompt"]'
  )!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      '8'
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  assert.doesNotMatch(document.body.textContent || '', /runningModels.reload.reasoningHint/);
  await act(async () => applyButton().click());
  assert.deepEqual(posts[0].body, {
    model_config: { reasoning_content: true, limit_mm_per_prompt: { image: 8 } },
  });
});
it('resumes an existing reload and reports restoration', async () => {
  initial = { status: 'reloading', stage: 'draining' };
  final = { status: 'error', error: 'out of memory', restored: true };
  await render();
  assert.ok(applyButton().disabled);
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1100));
  });
  assert.match(
    document.querySelector('[role="alert"]')?.textContent || '',
    /out of memory.*runningModels.reload.restored/
  );
  assert.equal(refreshed, 0);
  assert.equal(posts.length, 0);
});

it('does not present an earlier failed operation as a new dialog error', async () => {
  initial = { status: 'error', error: 'previous out of memory', restored: true };
  await render();
  assert.equal(document.querySelector('[role="alert"]'), null);
  assert.equal(document.querySelector('[role="status"]'), null);
  assert.equal(posts.length, 0);
});

it('shows the progress of a rolling replica reload', async () => {
  initial = {
    status: 'reloading',
    stage: 'loading',
    completed_replicas: 1,
    replica_count: 2,
    replicas: [
      { replica_uid: 'model-rep0', status: 'ready', stage: 'ready' },
      { replica_uid: 'model-rep1', status: 'reloading', stage: 'loading' },
    ],
  };
  await render();
  const progress = document.querySelector(
    'section[aria-label="runningModels.reload.replicaProgress"]'
  )!;
  assert.match(progress.textContent || '', /1\/2/);
  assert.match(progress.textContent || '', /model-rep0.*runningModels.reload.ready/);
  assert.match(progress.textContent || '', /model-rep1.*runningModels.reload.loading/);
  assert.ok(applyButton().disabled);
});

it('keeps partial rollout results visible when the dialog reopens', async () => {
  initial = {
    status: 'error',
    error: 'candidate failed',
    partial: true,
    completed_replicas: 1,
    replica_count: 2,
    replicas: [
      { replica_uid: 'model-rep0', status: 'ready', stage: 'ready' },
      { replica_uid: 'model-rep1', status: 'error', stage: 'error' },
    ],
  };
  await render();
  assert.match(
    document.querySelector('[role="alert"]')?.textContent || '',
    /candidate failed.*runningModels.reload.partial 1\/2/
  );
  assert.match(document.body.textContent || '', /model-rep0/);
  assert.match(document.body.textContent || '', /model-rep1/);
  assert.equal(closed, 0);
});

it('shows an uncertain recovered outcome even with only one reported participant', async () => {
  initial = {
    status: 'error',
    error: 'Reload result cannot be confirmed after Supervisor restart',
    outcome_unknown: true,
    replicas: [{ replica_uid: 'model-rep0', status: 'ready' }],
  };
  await render();
  assert.match(document.querySelector('[role="alert"]')?.textContent || '', /cannot be confirmed/);
  assert.equal(closed, 0);
});

for (const mixed of [false, true]) {
  it(`explicitly reapplies displayed values after reopening an unknown result${mixed ? ' without guessing mixed values' : ''}`, async () => {
    sequenceLimit = 32;
    if (mixed) mixedParameters = ['max_num_seqs'];
    initial = {
      status: 'error',
      error: 'Reload result cannot be confirmed after Supervisor restart',
      outcome_unknown: true,
      replicas: [{ replica_uid: 'model-rep0', status: 'ready' }],
    };
    await render();
    assert.ok(applyButton().disabled);
    assert.equal(posts.length, 0, 'Reopening must never automatically replay a reload');
    const reapply = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'runningModels.reload.reapply'
    )!;
    assert.ok(reapply && !reapply.disabled);
    await act(async () => reapply.click());
    assert.deepEqual(posts[0].body, {
      model_config: {
        ...(!mixed ? { max_num_seqs: 32 } : {}),
        enforce_eager: false,
        reasoning_content: false,
      },
    });
    assert.ok(!reapply.isConnected && applyButton().disabled);
  });
}

for (const changeAnotherField of [false, true]) {
  it(`refreshes mixed values before allowing a same-dialog revert${changeAnotherField ? ' with another change' : ''}`, async () => {
    await render();
    const input = document.querySelector<HTMLInputElement>('input[aria-label="max_num_seqs"]')!;
    const change = (value: string) => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
        input,
        value
      );
      input.dispatchEvent(new Event('input', { bubbles: true }));
    };
    await act(async () => change('32'));
    await act(async () => applyButton().click());
    final = {
      status: 'error',
      error: 'candidate failed',
      partial: true,
      completed_replicas: 1,
      replica_count: 2,
      replicas: [
        { replica_uid: 'model-rep0', status: 'ready' },
        { replica_uid: 'model-rep1', status: 'error' },
      ],
    };
    const get = request.get;
    let configReads = 0;
    let release!: () => void;
    const pending = new Promise<void>((resolve) => {
      release = resolve;
    });
    request.get = async <T,>(url: string) => {
      if (url.endsWith('/config')) {
        configReads++;
        await pending;
      }
      return get<T>(url);
    };
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1100));
    });
    assert.equal(configReads, 1);
    assert.ok(applyButton().disabled, 'The old comparison baseline must remain disabled');
    assert.ok(input.disabled);
    await act(async () => {
      mixedParameters = ['max_num_seqs'];
      release();
      await Promise.resolve();
    });
    assert.equal(input.value, '32', 'Keep the submitted target available for retry');
    assert.equal(input.placeholder, 'runningModels.reload.mixed');
    await act(async () => {
      change('16');
      if (changeAnotherField)
        document.querySelector<HTMLButtonElement>('button[aria-label="enforce_eager"]')!.click();
    });
    assert.equal(applyButton().disabled, false);
    await act(async () => applyButton().click());
    assert.deepEqual(posts[1].body, {
      model_config: { max_num_seqs: 16, ...(changeAnotherField ? { enforce_eager: true } : {}) },
    });
    assert.equal(closed, 0);
    assert.equal(refreshed, 0);
  });
}

for (const outcomeUnknown of [false, true]) {
  it(`keeps the submitted target retryable when actor values match but the operation ${outcomeUnknown ? 'is unconfirmed' : 'failed to commit'}`, async () => {
    await render();
    const input = document.querySelector<HTMLInputElement>('input[aria-label="max_num_seqs"]')!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
        input,
        '32'
      );
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => applyButton().click());
    sequenceLimit = 32;
    final = {
      status: 'error',
      error: 'model-wide commit not confirmed',
      outcome_unknown: outcomeUnknown,
      completed_replicas: 2,
      replica_count: 2,
      replicas: [
        { replica_uid: 'model-rep0', status: 'ready' },
        { replica_uid: 'model-rep1', status: 'ready' },
      ],
    };
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1100));
    });
    assert.equal(input.value, '32');
    assert.ok(!input.disabled && !applyButton().disabled);
    await act(async () => applyButton().click());
    assert.deepEqual(posts[1].body, { model_config: { max_num_seqs: 32 } });
    assert.equal(closed, 0);
    assert.equal(refreshed, 0);
  });
}

it('keeps submission disabled if partial-result metadata cannot be refreshed', async () => {
  initial = { status: 'reloading' };
  final = { status: 'error', partial: true, error: 'candidate failed' };
  await render();
  const get = request.get;
  request.get = async <T,>(url: string) => {
    if (url.endsWith('/config')) throw new Error('configuration unavailable');
    return get<T>(url);
  };
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1100));
  });
  assert.ok(applyButton().disabled);
  assert.equal(document.querySelector('input[aria-label="max_num_seqs"]'), null);
  assert.match(
    document.querySelector('[role="alert"]')?.textContent || '',
    /candidate failed.*configuration unavailable/
  );
  assert.equal(posts.length, 0);
});

it('lets a mixed boolean be explicitly set to false on every replica', async () => {
  mixedParameters = ['enforce_eager', 'max_num_seqs'];
  await render();
  const numeric = document.querySelector<HTMLInputElement>('input[aria-label="max_num_seqs"]')!;
  assert.equal(numeric.placeholder, 'runningModels.reload.mixed');
  const choice = document.querySelector<HTMLSelectElement>('select[aria-label="enforce_eager"]')!;
  assert.equal(choice.value, '');
  await act(async () => {
    choice.value = 'false';
    choice.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await act(async () => applyButton().click());
  assert.deepEqual(posts[0].body, { model_config: { enforce_eager: false } });
});
