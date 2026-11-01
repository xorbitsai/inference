/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
import type { Root } from 'react-dom/client';
import type { Locale } from '@/types/common';
import type { NavigationExtension, NavigationGroupExtension } from '@/lib/ui-extensions';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
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
require('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = {
  usePathname: () => '/custom-job',
  useRouter: () => ({ push() {}, replace() {} }),
};
require('next/link');
require.cache[require.resolve('next/link')]!.exports = {
  __esModule: true,
  default: ({ children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a {...props}>{children}</a>
  ),
};
require('next/image');
require.cache[require.resolve('next/image')]!.exports = {
  __esModule: true,
  default: () => <span />,
};
let locale: Locale, isAdmin: boolean;
const hidden = new Set<string>();
require('@/contexts/i18n-context');
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({ locale, t: (key: string) => key }),
};
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => ({
    globalReady: true,
    clusterAuth: { auth: true },
    clusterUIConfig: { auth_advanced: true },
  }),
};
require('@/hooks/use-menu-auth');
require.cache[require.resolve('@/hooks/use-menu-auth')]!.exports = {
  useMenuAuth: () => ({ isAdmin, hasPageAccess: (path: string) => !hidden.has(path) }),
};
const extensions = require('@/lib/ui-extensions') as typeof import('@/lib/ui-extensions');
const items = extensions.navigationExtensions as NavigationExtension[];
const groups = extensions.navigationGroupExtensions as NavigationGroupExtension[];
const { Sidebar } = require('./sidebar') as typeof import('./sidebar');
let host: HTMLDivElement, root: Root;
beforeEach(() => {
  locale = 'zh';
  isAdmin = true;
  hidden.clear();
  items.length = groups.length = 0;
  host = document.body.appendChild(document.createElement('div'));
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  items.length = groups.length = 0;
});
async function render() {
  await act(async () => root.render(<Sidebar />));
}
function section(path: string) {
  return host.querySelector(`a[href="${path}"]`)?.closest('section');
}

it('keeps legacy items in system and adds localized items to existing or custom sections', async () => {
  groups.push({ id: 'jobs', label: 'Jobs', labels: { zh: '任务' }, before: 'monitoring' });
  items.push(
    { path: '/legacy', label: 'Legacy' },
    { path: '/custom-model', label: 'Model tool', labels: { zh: '模型工具' }, group: 'models' },
    { path: '/custom-job', label: 'Custom job', labels: { zh: '自定义任务' }, group: 'jobs' }
  );
  await render();
  assert.match(section('/legacy')!.textContent!, /menu.systemManagement/);
  assert.match(section('/custom-model')!.textContent!, /menu.modelManagement.*模型工具/);
  assert.match(section('/custom-job')!.textContent!, /^任务自定义任务$/);
  const sections = [...host.querySelectorAll('nav section')];
  assert.ok(
    sections.indexOf(section('/custom-job')!) < sections.indexOf(section('/cluster-information')!)
  );
  locale = 'en';
  await render();
  assert.match(section('/custom-job')!.textContent!, /^JobsCustom job$/);
  locale = 'ja';
  await render();
  assert.match(section('/custom-job')!.textContent!, /^JobsCustom job$/);
});

it('filters restricted items before removing empty sections and retains collapsed link labels', async () => {
  groups.push({ id: 'jobs', label: 'Jobs' }, { id: 'empty', label: 'Empty' });
  items.push(
    { path: '/private', label: 'Private', group: 'empty', adminOnly: true },
    { path: '/denied', label: 'Denied', group: 'jobs' },
    { path: '/custom-job', label: 'Job', labels: { zh: '任务入口' }, group: 'jobs' }
  );
  isAdmin = false;
  hidden.add('/denied');
  await render();
  assert.equal(section('/private'), undefined);
  assert.equal(section('/denied'), undefined);
  assert.ok(!host.querySelector('nav')!.textContent!.includes('Empty'));
  await act(async () => host.querySelector<HTMLButtonElement>('button')!.click());
  assert.equal(host.querySelector('a[href="/custom-job"]')!.getAttribute('aria-label'), '任务入口');
});
