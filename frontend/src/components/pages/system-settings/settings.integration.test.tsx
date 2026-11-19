/* eslint-disable @typescript-eslint/no-require-imports -- Load the page after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
  HTMLElement: dom.window.HTMLElement,
  Element: dom.window.Element,
  Node: dom.window.Node,
  Event: dom.window.Event,
  CustomEvent: dom.window.CustomEvent,
  MutationObserver: dom.window.MutationObserver,
  getComputedStyle: dom.window.getComputedStyle,
  IS_REACT_ACT_ENVIRONMENT: true,
  requestAnimationFrame: (callback: FrameRequestCallback) => setTimeout(callback, 0),
  cancelAnimationFrame: (id: number) => clearTimeout(id),
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
  useI18n: () => ({ t: (key: string) => key }),
};
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => ({ clusterUIConfig: { auth_advanced: true } }),
};
require('@/hooks/use-menu-auth');
require.cache[require.resolve('@/hooks/use-menu-auth')]!.exports = {
  useMenuAuth: () => ({ canWriteSettings: true, isAdmin }),
};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const Page = require('./index').default as React.ComponentType;
const originals = { get: request.get, put: request.put, post: request.post };
const observabilityPath = '/v1/cluster/observability_settings';
let root: Root, host: HTMLDivElement, isAdmin: boolean, failRestart: boolean;
let calls: string[], pending: Array<'api' | 'supervisor'>;
let otel: Record<string, string | number | boolean>;
function data() {
  return {
    opentelemetry: { ...otel },
    restart_required: pending.length > 0,
    restart_components: [...pending],
    restart_supported: true,
  };
}
beforeEach(() => {
  isAdmin = true;
  failRestart = false;
  calls = [];
  pending = [];
  otel = {
    enable_otel: false,
    otlp_base_endpoint: 'http://localhost:4318',
    otlp_trace_endpoint: '',
    otlp_metric_endpoint: '',
    otlp_api_key: '',
    otel_exporter_otlp_protocol: 'http/protobuf',
    otel_sampling_rate: 0.1,
    otel_batch_export_schedule_delay: 5000,
    otel_max_queue_size: 2048,
    otel_max_export_batch_size: 512,
    otel_metric_export_interval: 60000,
    otel_batch_export_timeout: 10000,
    otel_metric_export_timeout: 30000,
  };
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  request.get = (async (url: string) => {
    if (url === observabilityPath) return data();
    if (url.endsWith('-relaunches')) return [];
    return {
      download_source: 'auto',
      hf_endpoint: '',
      hf_token: '',
      pip_index_url: '',
      download_max_attempts: 3,
      hub_detect_timeout: 3,
      model_download_workers: 2,
    };
  }) as typeof request.get;
  request.put = (async (url: string, body: { opentelemetry: typeof otel }) => {
    calls.push(`PUT ${url}`);
    otel = body.opentelemetry;
    pending = ['supervisor', 'api'];
    return data();
  }) as typeof request.put;
  request.post = (async (url: string) => {
    calls.push(`POST ${url}`);
    if (failRestart) throw new Error('OTEL SDK unavailable');
    pending = pending.filter((component) => !url.includes(`/${component}/`));
    return { operation_id: url, status: 'succeeded' };
  }) as typeof request.post;
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  Object.assign(request, originals);
});
async function render() {
  await act(async () => root.render(<Page />));
  await act(async () =>
    host
      .querySelector('[role="tab"][id$="observability"]')!
      .dispatchEvent(new dom.window.MouseEvent('mousedown', { bubbles: true, button: 0 }))
  );
}
async function toggle() {
  await act(async () => host.querySelector<HTMLButtonElement>('#enable-otel')!.click());
}
function restart() {
  return [...host.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
    button.textContent?.includes('systemSettings.restartNow')
  );
}

it('shows the action only when settings change and hides it when restored', async () => {
  await render();
  assert.equal(restart(), undefined);
  await toggle();
  assert.ok(restart());
  await toggle();
  assert.equal(restart(), undefined);
});
it('saves then restarts Supervisor and API and confirms the applied state', async () => {
  await render();
  await toggle();
  await act(async () => restart()!.click());
  assert.deepEqual(calls, [
    `PUT ${observabilityPath}`,
    'POST /v1/supervisor/relaunch',
    'POST /v1/api/relaunch',
  ]);
  assert.equal(restart(), undefined);
});
it('retains a retryable action and stops before API when Supervisor fails', async () => {
  failRestart = true;
  await render();
  await toggle();
  await act(async () => restart()!.click());
  assert.match(host.querySelector('[role="alert"]')!.textContent!, /OTEL SDK unavailable/);
  assert.equal(calls.includes('POST /v1/api/relaunch'), false);
  assert.equal(restart()!.disabled, false);
});
it('does not allow a settings writer to restart without administrator access', async () => {
  isAdmin = false;
  await render();
  await toggle();
  assert.equal(restart()!.disabled, true);
  assert.deepEqual(calls, []);
});
