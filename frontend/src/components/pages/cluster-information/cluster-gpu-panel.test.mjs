import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const React = require('react');
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  HTMLElement: dom.window.HTMLElement,
  Event: dom.window.Event,
  IS_REACT_ACT_ENVIRONMENT: true,
});
const Module = require('node:module');
const load = Module._load;
Module._load = function (name, ...args) {
  if (name === '@/contexts/i18n-context') return { useI18n: () => ({ t: (key) => key }) };
  return load.call(this, name, ...args);
};
const { default: Panel, memoryRate, ResourceMeter } = require('./cluster-gpu-panel.tsx');
Module._load = load;
const { createRoot } = require('react-dom/client');
const worker = (count) => ({
  ip_address: 'worker-alpha',
  gpu_count: count,
  cpu_count: 32,
  cpu_available: 16,
  mem_used: 128,
  mem_total: 512,
  mem_available: 384,
  gpu_utilization: 50,
  gpus: Array.from({ length: count }, (_, i) => ({
    index: 'gpu-' + i,
    name: 'NVIDIA H100',
    utilization: i * 5,
    memory_used: 20,
    memory_total: 80,
    memory_available: 60,
  })),
});
for (const count of [0, 1, 2, 4, 8, 12, 16])
  test('renders every device for ' + count + ' GPUs', async () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    await React.act(async () =>
      root.render(React.createElement(Panel, { workers: [worker(count)] }))
    );
    const visited = new Set();
    const previous = host.querySelector('[aria-label="clusterInfo.previousDevices"]');
    if (previous) assert.equal(previous.disabled, true);
    for (let page = 0; page < Math.max(1, Math.ceil(count / 8)); page++) {
      const cards = host.querySelectorAll('button[aria-label^="GPU-"]');
      assert.equal(cards.length, Math.min(8, Math.max(0, count - page * 8)));
      assert.equal(host.querySelectorAll('[role="meter"]').length, cards.length * 2);
      for (const card of cards) visited.add(card.getAttribute('aria-label'));
      if (cards.length) {
        const last = cards[cards.length - 1];
        await React.act(async () => last.click());
        assert.equal(last.getAttribute('aria-pressed'), 'true');
      }
      const next = host.querySelector('[aria-label="clusterInfo.nextDevices"]');
      if (page < Math.ceil(count / 8) - 1) {
        assert.equal(next.disabled, false);
        await React.act(async () => next.click());
        assert.equal(
          host
            .querySelector('button[aria-pressed="true"][aria-label^="GPU-"]')
            .getAttribute('aria-label'),
          'GPU-' + (page + 1) * 8
        );
      } else if (next) assert.equal(next.disabled, true);
    }
    assert.equal(visited.size, count);
    if (previous) {
      await React.act(async () => previous.click());
      assert.ok(host.querySelector('[aria-label="GPU-0"]'));
      assert.equal(previous.disabled, true);
    }
    const list = host.querySelector('[aria-label="clusterInfo.listView"]');
    await React.act(async () => list.click());
    assert.equal(list.getAttribute('aria-pressed'), 'true');
    assert.equal(host.querySelectorAll('[role="meter"]').length, count * 2);
    const input = host.querySelector('input');
    await React.act(async () => {
      Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value').set.call(
        input,
        'no-match'
      );
      input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
    });
    assert.ok(host.textContent.includes('clusterInfo.noMatches'));
    await React.act(async () => root.unmount());
    host.remove();
  });
test('unknown GPU memory stays unknown and zero load remains zero', () => {
  assert.equal(memoryRate({ memory_used: 0, memory_total: 0 }), null);
  assert.equal(memoryRate({ memory_used: null, memory_total: 80 }), null);
  assert.equal(memoryRate({ memory_used: 0, memory_total: 80 }), 0);
});

test('page refresh stays in the page header and node categories switch independently', async () => {
  let calls = 0;
  let serviceCompletion;
  Module._load = function (name, ...args) {
    if (name === '@/contexts/i18n-context') return { useI18n: () => ({ t: (key) => key }) };
    if (name === '@/contexts/global-context')
      return {
        useGlobal: () => ({
          globalReady: true,
          clusterUIConfig: { token_router_enabled: true },
          clusterVersion: { version: '4.2' },
        }),
      };
    if (name === '@/hooks/use-menu-auth') return { useMenuAuth: () => ({ isAdmin: true }) };
    if (name === '@/lib/request')
      return {
        __esModule: true,
        default: {
          get: async () => {
            calls++;
            return [
              { ...worker(16), node_type: 'Worker' },
              { ...worker(0), node_type: 'Supervisor', ip_address: 'control-alpha' },
            ];
          },
        },
      };
    if (name === '../worker-relaunch')
      return {
        useWorkerRelaunches: () => ({ operations: [], busyAddresses: new Set() }),
        WorkerRelaunchDialog: () => null,
        WorkerRelaunchHistory: () => null,
      };
    if (name === '../service-relaunch')
      return {
        ServiceRelaunchPanel: ({ onCompleted }) => {
          serviceCompletion = onCompleted;
          return React.createElement('div', { 'data-service-controls': true }, 'API / Supervisor');
        },
      };
    return load.call(this, name, ...args);
  };
  const Page = require('./index.tsx').default;
  Module._load = load;
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await React.act(async () => root.render(React.createElement(Page)));
  const refresh = Array.from(host.querySelectorAll('button')).find(
    (b) => b.textContent === 'clusterInfo.refresh'
  );
  assert.ok(host.querySelector('h1').parentElement.parentElement.contains(refresh));
  assert.ok(host.textContent.includes('worker-alpha'));
  assert.ok(host.querySelector('[data-service-controls]'));
  assert.equal(host.querySelector('[data-service-controls]').closest('details'), null);
  const tabs = host.querySelectorAll('[role="tab"]');
  await React.act(async () => tabs[1].click());
  assert.ok(host.textContent.includes('control-alpha'));
  assert.ok(!host.textContent.includes('worker-alpha'));
  await React.act(async () => refresh.click());
  assert.equal(calls, 2);
  let externalRefreshes = 0;
  await React.act(async () =>
    root.render(
      React.createElement(Page, {
        dataSource: {
          nodes: [{ ...worker(2), node_type: 'Worker' }],
          loaded: true,
          updating: false,
          failed: false,
          updatedAt: '12:00',
          refresh: () => externalRefreshes++,
        },
        decorations: {
          summaryCount: () => '1/2',
          workerBadge: () => React.createElement('span', null, 'worker license'),
          gpuBadge: (_worker, gpu) => React.createElement('span', null, gpu.index + ' license'),
          workerGpuCount: () => '1/2',
        },
      })
    )
  );
  assert.equal(calls, 2, 'an external source must not poll the community endpoint');
  await React.act(async () => host.querySelectorAll('[role="tab"]')[0].click());
  assert.ok(host.textContent.includes('worker license'));
  assert.ok(host.textContent.includes('gpu-1 license'));
  await React.act(async () => refresh.click());
  assert.equal(externalRefreshes, 1);
  await React.act(async () => serviceCompletion());
  assert.equal(externalRefreshes, 2);

  assert.equal(calls, 2);

  await React.act(async () => root.unmount());
  host.remove();
});

test('progress uses the solid theme color through 90% and red only above 90%', async () => {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await React.act(async () =>
    root.render(
      React.createElement(
        'div',
        null,
        ...[0, 90, 90.1, null].map((value, index) =>
          React.createElement(ResourceMeter, { key: index, label: 'metric-' + index, value })
        )
      )
    )
  );
  assert.ok(host.querySelector('[role="meter"]').classList.contains('h-2'));
  const bars = host.querySelectorAll('[role="meter"] > div');
  assert.ok(bars[0].classList.contains('bg-primary'));
  assert.ok(bars[1].classList.contains('bg-primary'));
  assert.ok(bars[2].classList.contains('bg-red-500'));
  assert.equal(host.querySelectorAll('[role="meter"]')[3].hasAttribute('aria-valuenow'), false);
  await React.act(async () => root.unmount());
  host.remove();
});

test('a shrinking GPU list clamps the current page and updates selected details', async () => {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await React.act(async () => root.render(React.createElement(Panel, { workers: [worker(16)] })));
  await React.act(async () => host.querySelector('[aria-label="clusterInfo.nextDevices"]').click());
  assert.ok(host.querySelector('[aria-label="GPU-8"]'));
  await React.act(async () => root.render(React.createElement(Panel, { workers: [worker(4)] })));
  assert.equal(host.querySelectorAll('button[aria-label^="GPU-"]').length, 4);
  assert.ok(host.querySelector('[aria-label="GPU-0"]'));
  assert.equal(host.querySelector('[aria-label="clusterInfo.nextDevices"]'), null);
  await React.act(async () => root.unmount());
  host.remove();
});

test('narrow screens keep two rows and reflow safely on resize', async () => {
  let resize;
  const media = {
    matches: false,
    addEventListener: (_, fn) => {
      resize = fn;
    },
    removeEventListener: () => {},
  };
  window.matchMedia = () => media;
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await React.act(async () => root.render(React.createElement(Panel, { workers: [worker(12)] })));
  assert.equal(host.querySelectorAll('button[aria-label^="GPU-"]').length, 4);
  await React.act(async () => host.querySelector('[aria-label="clusterInfo.nextDevices"]').click());
  await React.act(async () => host.querySelector('[aria-label="clusterInfo.nextDevices"]').click());
  assert.ok(host.querySelector('[aria-label="GPU-8"]'));
  await React.act(async () => {
    media.matches = true;
    resize();
  });
  assert.equal(host.querySelectorAll('button[aria-label^="GPU-"]').length, 4);
  assert.ok(host.querySelector('[aria-label="GPU-8"]'));
  await React.act(async () =>
    host.querySelector('[aria-label="clusterInfo.previousDevices"]').click()
  );
  assert.equal(host.querySelectorAll('button[aria-label^="GPU-"]').length, 8);
  await React.act(async () => root.unmount());
  host.remove();
  delete window.matchMedia;
});

test('all five community locales contain the same active cluster translation keys', () => {
  const flatten = (source, prefix = '') =>
    Object.entries(source).flatMap(([key, value]) =>
      typeof value === 'string' ? [prefix + key] : flatten(value, prefix + key + '.')
    );
  const expected = flatten(require('@/i18n/locales/en.ts').default.clusterInfo).sort();
  for (const locale of ['zh', 'zh-TW', 'ja', 'ko']) {
    const actual = flatten(require('@/i18n/locales/' + locale + '.ts').default.clusterInfo).sort();
    assert.deepEqual(actual, expected, locale);
  }
});
test('operation completion refreshes once for success or failure and ignores historical results', async () => {
  const { useOperationCompletion } = require('@/hooks/use-operation-completion');
  let first = 0,
    latest = 0;
  const host = document.body.appendChild(document.createElement('div'));
  const root = createRoot(host);
  function Probe({ operations, onCompleted }) {
    useOperationCompletion(operations, onCompleted);
    return null;
  }
  const render = async (status, callback) =>
    React.act(async () =>
      root.render(
        React.createElement(Probe, {
          operations: [{ operation_id: 'one', status }],
          onCompleted: callback,
        })
      )
    );
  await render('succeeded', () => first++);
  assert.equal(first, 0);
  await render('starting', () => first++);
  await render('succeeded', () => first++);
  assert.equal(first, 1);
  await render('succeeded', () => latest++);
  assert.equal(latest, 0);
  await render('starting', () => first++);
  await render('failed', () => latest++);
  assert.equal(first, 1);
  assert.equal(latest, 1);
  await render('failed', () => latest++);
  assert.equal(latest, 1);
  await React.act(async () => root.unmount());
  host.remove();
});
