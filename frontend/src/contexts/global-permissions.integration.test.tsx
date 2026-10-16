/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { it } from 'node:test';
import React, { act, useEffect, useRef } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  sessionStorage: dom.window.sessionStorage,
  Event: dom.window.Event,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
require('@/contexts/i18n-context');
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({ t: (key: string) => key }),
};
require('@/lib/ui-extensions');
require.cache[require.resolve('@/lib/ui-extensions')]!.exports = {
  pageAccessExtensions: [
    { path: '/cache-management', scope: 'page:cache', appliesWhen: 'pages:restricted' },
  ],
};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { GlobalProvider, useGlobal } =
  require('./global-context') as typeof import('./global-context');
const { useMenuAuth } = require('@/hooks/use-menu-auth') as typeof import('@/hooks/use-menu-auth');
const { PermissionGuard } =
  require('@/components/auth/permission-guard') as typeof import('@/components/auth/permission-guard');
const { setAccessToken, removeAuthTokens } =
  require('@/lib/auth-storage') as typeof import('@/lib/auth-storage');

function Probe() {
  const global = useGlobal();
  const initialized = useRef(false);
  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      global.fetchGlobalAfterAuth();
    }
  }, [global]);
  const auth = useMenuAuth();
  return (
    <div>
      <output
        data-admin={auth.isAdmin}
        data-cache={auth.hasPageAccess('/cache-management')}
        data-read={auth.hasModelsRead}
      />
      <PermissionGuard scope="monitor:view">
        <span id="monitor">protected content</span>
      </PermissionGuard>
    </div>
  );
}

it('refreshes menus and direct-page grants from the server while JWT scopes remain unchanged', async () => {
  const originalGet = request.get;
  let permissions = ['admin'];
  request.get = async <T,>(url: string) => {
    if (url === '/v1/cluster/ui_config') return { auth_advanced: true } as T;
    if (url === '/v1/auth/me') return { permissions: [...permissions] } as T;
    return {} as T;
  };
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const token = `header.${Buffer.from(JSON.stringify({ scopes: ['admin'] })).toString('base64url')}.signature`;
  setAccessToken(token);
  try {
    await act(async () => {
      root.render(
        <GlobalProvider initClusterAuth={{ auth: true }}>
          <Probe />
        </GlobalProvider>
      );
    });
    assert.equal(host.querySelector('output')?.getAttribute('data-admin'), 'true');
    assert.ok(host.querySelector('#monitor'));
    permissions = ['pages:restricted'];
    await act(async () => {
      window.dispatchEvent(new Event('focus'));
    });
    assert.equal(sessionStorage.getItem('token'), token);
    assert.equal(host.querySelector('output')?.getAttribute('data-admin'), 'false');
    assert.equal(host.querySelector('output')?.getAttribute('data-cache'), 'false');
    assert.equal(host.querySelector('#monitor'), null);
    permissions = ['pages:restricted', 'page:cache'];
    await act(async () => {
      window.dispatchEvent(new Event('focus'));
    });
    assert.equal(host.querySelector('output')?.getAttribute('data-cache'), 'true');
    assert.equal(host.querySelector('output')?.getAttribute('data-read'), 'false');
    await act(async () => {
      removeAuthTokens();
    });
    assert.equal(host.querySelector('output')?.getAttribute('data-admin'), 'false');
  } finally {
    await act(async () => {
      root.unmount();
    });
    host.remove();
    request.get = originalGet;
  }
});
