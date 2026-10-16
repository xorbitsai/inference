/* eslint-disable @typescript-eslint/no-require-imports -- Load UI after jsdom. */
import assert from 'node:assert/strict';
import { it } from 'node:test';
import { createServer } from 'node:http';
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
for (const key of [
  'Element',
  'HTMLElement',
  'HTMLInputElement',
  'Node',
  'NodeFilter',
  'MutationObserver',
  'CustomEvent',
  'DocumentFragment',
]) {
  Object.defineProperty(globalThis, key, Object.getOwnPropertyDescriptor(dom.window, key)!);
}
Object.assign(globalThis, { getComputedStyle: dom.window.getComputedStyle });
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
const navigation = require('next/navigation') as typeof import('next/navigation');
require.cache[require.resolve('next/navigation')]!.exports = {
  ...navigation,
  usePathname: () => '/cache-management',
};
require('@/components/layout/sidebar');
require.cache[require.resolve('@/components/layout/sidebar')]!.exports = { Sidebar: () => null };
const { LayoutContent } =
  require('@/components/layout/layout-content') as typeof import('@/components/layout/layout-content');

function tokenFor(userId: number, revision = 0) {
  return `header.${Buffer.from(JSON.stringify({ user_id: userId, sub: `user${userId}`, scopes: ['admin'], exp: revision })).toString('base64url')}.signature`;
}

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
        data-env={auth.hasVirtualEnvList}
        data-password={auth.canChangePassword}
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
    if (url === '/v1/auth/me') return { permissions: [...permissions], source: 'local' } as T;
    return {} as T;
  };
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const token = tokenFor(1);
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

it('keeps form state across same-user token rotation, transient errors and metadata saves', async () => {
  const originalGet = request.get;
  let pendingProfile: Promise<unknown> | undefined;
  let pendingConfig: Promise<unknown> | undefined;
  let failProfile = false;
  let mounts = 0;
  function FormPage() {
    const { fetchGlobalAfterAuth } = useGlobal();
    useEffect(() => {
      mounts += 1;
    }, []);
    return (
      <div>
        <input id="draft" defaultValue="" />
        <button
          onClick={() => {
            void fetchGlobalAfterAuth();
          }}
        >
          Save configuration
        </button>
      </div>
    );
  }
  function InitializedPage() {
    const { fetchGlobalAfterAuth } = useGlobal();
    useEffect(() => {
      void fetchGlobalAfterAuth();
    }, [fetchGlobalAfterAuth]);
    return (
      <LayoutContent>
        <PermissionGuard scope="monitor:view">
          <FormPage />
        </PermissionGuard>
      </LayoutContent>
    );
  }
  request.get = async <T,>(url: string) => {
    if (url === '/v1/cluster/ui_config')
      return (pendingConfig ? await pendingConfig : { auth_advanced: true }) as T;
    if (url === '/v1/auth/me') {
      if (failProfile) throw new Error('temporary network error');
      return (
        pendingProfile ? await pendingProfile : { permissions: ['admin'], source: 'local' }
      ) as T;
    }
    return {} as T;
  };
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  setAccessToken(tokenFor(1));
  try {
    await act(async () => {
      root.render(
        <GlobalProvider initClusterAuth={{ auth: true }}>
          <InitializedPage />
        </GlobalProvider>
      );
    });
    const draft = host.querySelector<HTMLInputElement>('#draft');
    assert.ok(draft);
    draft.value = 'unsaved chat and form state';
    let finishProfile!: (value: unknown) => void;
    pendingProfile = new Promise((resolve) => {
      finishProfile = resolve;
    });
    await act(async () => {
      setAccessToken(tokenFor(1, 1));
    });
    assert.equal(host.querySelector('#draft'), draft);
    await act(async () => {
      finishProfile({ permissions: ['admin'], source: 'local' });
    });
    pendingProfile = undefined;
    failProfile = true;
    await act(async () => {
      window.dispatchEvent(new Event('focus'));
    });
    assert.equal(host.querySelector('#draft'), draft);
    assert.equal(draft.value, 'unsaved chat and form state');
    failProfile = false;
    let finishConfig!: (value: unknown) => void;
    pendingConfig = new Promise((resolve) => {
      finishConfig = resolve;
    });
    await act(async () => {
      host.querySelector('button')!.click();
    });
    assert.equal(host.querySelector('#draft'), draft);
    await act(async () => {
      finishConfig({ auth_advanced: true });
    });
    assert.equal(mounts, 1);
    pendingProfile = new Promise((resolve) => {
      finishProfile = resolve;
    });
    await act(async () => {
      setAccessToken(tokenFor(2));
    });
    assert.equal(host.querySelector('#draft'), null);
    await act(async () => {
      finishProfile({ permissions: [], source: 'oidc' });
    });
    assert.equal(host.querySelector('#draft'), null);
    await act(async () => {
      removeAuthTokens();
    });
    assert.equal(host.querySelector('#draft'), null);
  } finally {
    await act(async () => {
      root.unmount();
    });
    host.remove();
    request.get = originalGet;
  }
});

it('retains unrestricted virtualenv access in no-auth mode and hides OIDC password changes', async () => {
  const originalGet = request.get;
  let noAuth = true;
  request.get = async <T,>(url: string) => {
    if (url === '/v1/cluster/ui_config') return { auth_advanced: !noAuth } as T;
    if (url === '/v1/auth/me') return { permissions: ['admin'], source: 'oidc' } as T;
    return {} as T;
  };
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  try {
    setAccessToken('no_auth');
    await act(async () => {
      root.render(
        <GlobalProvider initClusterAuth={{ auth: false }}>
          <Probe />
        </GlobalProvider>
      );
    });
    assert.equal(host.querySelector('output')?.getAttribute('data-env'), 'true');
    assert.equal(host.querySelector('output')?.getAttribute('data-password'), 'false');
    noAuth = false;
    await act(async () => {
      root.unmount();
    });
    const oidcRoot = createRoot(host);
    setAccessToken(tokenFor(2));
    try {
      await act(async () => {
        oidcRoot.render(
          <GlobalProvider initClusterAuth={{ auth: true }}>
            <Probe />
          </GlobalProvider>
        );
      });
      assert.equal(host.querySelector('output')?.getAttribute('data-password'), 'false');
    } finally {
      await act(async () => {
        oidcRoot.unmount();
      });
    }
  } finally {
    host.remove();
    request.get = originalGet;
  }
});

it('keeps an optional forbidden version request out of global permission notifications', async () => {
  const { eventBus } = require('@/lib/event-bus') as typeof import('@/lib/event-bus');
  const { RequestEvents } = require('@/constants') as typeof import('@/constants');
  const { requestManager } =
    require('@/lib/request-manager') as typeof import('@/lib/request-manager');
  const server = createServer((_, response) => {
    response.writeHead(403, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ detail: 'Not enough permissions' }));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  let notifications = 0;
  const forbidden = () => {
    notifications += 1;
  };
  eventBus.on(RequestEvents.FORBIDDEN, forbidden);
  requestManager.reset403();
  try {
    const url = `http://127.0.0.1:${address.port}/v1/cluster/version`;
    await assert.rejects(request.get(url, { suppressGlobalError: true, proxy: false }));
    assert.equal(notifications, 0);
    await assert.rejects(request.get(url, { proxy: false }));
    assert.equal(notifications, 1);
  } finally {
    eventBus.off(RequestEvents.FORBIDDEN, forbidden);
    requestManager.reset403();
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve()))
    );
  }
});

it('reports each wrong-current-password response without the generic 403 notification', async () => {
  const { ChangePasswordDialog } =
    require('@/components/pages/login/change-password-dialog') as typeof import('@/components/pages/login/change-password-dialog');
  const { toast } = require('sonner') as typeof import('sonner');
  const originalPut = request.put;
  const originalError = toast.error;
  const errors: string[] = [];
  let attempts = 0;
  let changed = 0;
  toast.error = (message) => {
    errors.push(String(message));
    return 0;
  };
  request.put = async (_url, body, config) => {
    attempts += 1;
    assert.equal(body.current_password, 'wrong-password');
    assert.equal(config?.suppressGlobalError, true);
    throw { isAxiosError: true, response: { data: { detail: 'Current password is incorrect' } } };
  };
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  try {
    await act(async () => {
      root.render(
        <ChangePasswordDialog
          open
          userId={1}
          forced={false}
          onChanged={() => {
            changed += 1;
          }}
        />
      );
    });
    const inputs = [...document.querySelectorAll<HTMLInputElement>('input[type="password"]')];
    assert.equal(inputs.length, 3);
    await act(async () => {
      inputs.forEach((input, index) => {
        Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value')!.set!.call(
          input,
          index === 0 ? 'wrong-password' : 'new-password'
        );
        input.dispatchEvent(new Event('input', { bubbles: true }));
      });
    });
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await act(async () => {
        document
          .querySelector('form')!
          .dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      });
    }
    assert.equal(attempts, 2);
    assert.deepEqual(errors, ['Current password is incorrect', 'Current password is incorrect']);
    assert.equal(changed, 0);
  } finally {
    await act(async () => {
      root.unmount();
    });
    host.remove();
    request.put = originalPut;
    toast.error = originalError;
  }
});
