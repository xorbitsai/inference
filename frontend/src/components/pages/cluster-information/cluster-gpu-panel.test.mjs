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
    if (count === 0) assert.ok(host.textContent.includes('clusterInfo.deviceUnavailable'));
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
    assert.ok(!host.textContent.includes('clusterInfo.selectDevice'));
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
  let workerOperations = [];
  const routerNodes = ['router-alpha', 'router-beta'].map((node_id) => ({
    ...worker(0),
    node_type: 'Router',
    node_id,
    ip_address: 'shared-router-host',
    online: true,
    connectivity_status: 'online',
    software_version: node_id === 'router-alpha' ? '4.1' : '',
  }));
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
              ...routerNodes,
            ];
          },
        },
      };
    if (name === '../worker-relaunch')
      return {
        useWorkerRelaunches: () => ({ operations: workerOperations, busyAddresses: new Set() }),
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
  const tabs = host.querySelectorAll('[role="tab"]');
  await React.act(async () => tabs[1].click());
  assert.ok(host.textContent.includes('control-alpha'));
  assert.ok(host.textContent.includes('4.2'), 'Supervisor rows may use the cluster version');
  assert.ok(!host.textContent.includes('worker-alpha'));
  await React.act(async () => refresh.click());
  assert.equal(calls, 2);
  const errors = [];
  const originalError = console.error;
  console.error = (...args) => errors.push(args.join(' '));
  try {
    await React.act(async () => tabs[2].click());
  } finally {
    console.error = originalError;
  }
  assert.deepEqual(errors, [], 'same-host Router Agents must have unique React keys');
  const nodeRow = (id) =>
    Array.from(host.querySelectorAll('p')).find((p) => p.textContent === id).parentElement
      .parentElement;
  const alpha = nodeRow('router-alpha');
  const beta = nodeRow('router-beta');
  assert.notEqual(alpha, beta);
  assert.ok(alpha.textContent.includes('4.1'));
  assert.ok(beta.textContent.includes('—'));
  assert.ok(
    !beta.textContent.includes('4.2'),
    'unknown Router versions must not inherit the Supervisor version'
  );
  for (const row of [alpha, beta]) assert.ok(row.textContent.includes('shared-router-host'));
  await React.act(async () =>
    root.render(
      React.createElement(Page, {
        dataSource: {
          nodes: [...routerNodes].reverse(),
          loaded: true,
          updating: false,
          failed: false,
          updatedAt: '12:00',
          refresh: () => {},
        },
      })
    )
  );
  assert.equal(nodeRow('router-alpha'), alpha, 'Router identity survives reordered snapshots');
  assert.equal(nodeRow('router-beta'), beta);
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
  assert.ok(
    host.querySelector('[aria-label="GPU-0"] span.h-1'),
    'decorations keep the utilization indicator'
  );
  await React.act(async () => refresh.click());
  assert.equal(externalRefreshes, 1);
  await React.act(async () => serviceCompletion());
  assert.equal(externalRefreshes, 2);

  const source = {
    nodes: [],
    loaded: true,
    updating: false,
    failed: false,
    updatedAt: '12:01',
    refresh: () => externalRefreshes++,
  };
  const transition = async (status) => {
    workerOperations = [{ operation_id: 'worker-one', source_worker: 'worker-alpha', status }];
    await React.act(async () => root.render(React.createElement(Page, { dataSource: source })));
  };
  await transition('starting');
  assert.equal(externalRefreshes, 2);
  await transition('succeeded');
  assert.equal(externalRefreshes, 3, 'Worker completion refreshes the active data source');
  await transition('succeeded');
  assert.equal(externalRefreshes, 3, 'the same completed operation must not refresh twice');
  await transition('starting');
  await transition('failed');
  assert.equal(externalRefreshes, 4, 'failed Worker operations also refresh the active source');
  await React.act(async () => host.querySelectorAll('[role="tab"]')[1].click());
  assert.ok(host.textContent.includes('clusterInfo.noSupervisors'));
  assert.ok(!host.textContent.includes('clusterInfo.noRouterInstances'));
  await React.act(async () => host.querySelectorAll('[role="tab"]')[2].click());
  assert.ok(host.textContent.includes('clusterInfo.noRouterInstances'));
  assert.ok(!host.textContent.includes('clusterInfo.noSupervisors'));
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

test('resizing preserves the first visible GPU range on an intermediate page', async () => {
  let resize;
  const media = {
    matches: false,
    addEventListener: (_, fn) => {
      resize = fn;
    },
    removeEventListener: () => {},
  };
  window.matchMedia = () => media;
  const host = document.body.appendChild(document.createElement('div'));
  const root = createRoot(host);
  try {
    await React.act(async () => root.render(React.createElement(Panel, { workers: [worker(16)] })));
    await React.act(async () =>
      host.querySelector('[aria-label="clusterInfo.nextDevices"]').click()
    );
    assert.ok(host.querySelector('[aria-label="GPU-4"]'));
    assert.ok(!host.querySelector('[aria-label="GPU-0"]'));
    await React.act(async () => {
      media.matches = true;
      resize();
    });
    for (const i of [4, 5, 6, 7]) assert.ok(host.querySelector('[aria-label="GPU-' + i + '"]'));
    assert.equal(host.querySelectorAll('button[aria-label^="GPU-"]').length, 8);
    assert.equal(
      host.querySelector('[aria-pressed="true"][aria-label^="GPU-"]').getAttribute('aria-label'),
      'GPU-4'
    );
    await React.act(async () => {
      media.matches = false;
      resize();
    });
    assert.ok(host.querySelector('[aria-label="GPU-4"]'));
    assert.ok(!host.querySelector('[aria-label="GPU-0"]'));
  } finally {
    await React.act(async () => root.unmount());
    host.remove();
    delete window.matchMedia;
  }
});

test('unknown per-device memory never appears as a measured placeholder capacity', async () => {
  const value = worker(1);
  Object.assign(value.gpus[0], { memory_used: null, memory_total: null, memory_available: null });
  const host = document.body.appendChild(document.createElement('div'));
  const root = createRoot(host);
  await React.act(async () => root.render(React.createElement(Panel, { workers: [value] })));
  const card = host.querySelector('[aria-label="GPU-0"]');
  assert.equal(card.querySelectorAll('[role="meter"]')[1].hasAttribute('aria-valuenow'), false);
  assert.ok(host.textContent.includes('clusterInfo.unavailable'));
  assert.ok(!host.textContent.includes('1 B'));
  await React.act(async () => root.unmount());
  host.remove();
});

test('shared resource percentages preserve unknown and zero readings and clamp bounds', () => {
  const { usagePercent, availableUsagePercent } = require('./utils.ts');
  assert.equal(usagePercent(null, 100), null);
  assert.equal(usagePercent(1, 0), null);
  assert.equal(usagePercent(Infinity, 100), null);
  assert.equal(usagePercent(0, 100), 0);
  assert.equal(usagePercent(-10, 100), 0);
  assert.equal(usagePercent(200, 100), 100);
  assert.equal(availableUsagePercent(100, 75), 25);
  assert.equal(availableUsagePercent(100, null), null);
});
