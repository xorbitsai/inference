/* eslint-disable @typescript-eslint/no-require-imports -- Load hooks after jsdom. */
import assert from 'node:assert/strict';
import { beforeEach, afterEach, it } from 'node:test';
import React, { act } from 'react';
import type { Root } from 'react-dom/client';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  sessionStorage: dom.window.sessionStorage,
  Event: dom.window.Event,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
let authenticated: boolean | undefined;
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => ({
    globalReady: authenticated !== undefined,
    clusterAuth: { auth: authenticated },
  }),
};
const { useModelFavorites } =
  require('./use-model-favorites') as typeof import('./use-model-favorites');
const { setAccessToken, removeAuthTokens } =
  require('@/lib/auth-storage') as typeof import('@/lib/auth-storage');
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
let view: ReturnType<typeof useModelFavorites>;
function Probe() {
  view = useModelFavorites();
  return <div>{view.favorites.join(',')}</div>;
}
let root: Root;
let host: HTMLDivElement;
let gets: number;
let writes: { method: string; body: unknown }[];
let read: () => Promise<string[]>;
let write: () => Promise<unknown>;
const original = { get: request.get, put: request.put, delete: request.delete };
function token(user: number) {
  return (
    'header.' + Buffer.from(JSON.stringify({ user_id: user })).toString('base64url') + '.signature'
  );
}
beforeEach(() => {
  authenticated = true;
  sessionStorage.clear();
  window.localStorage.clear();
  setAccessToken(token(1));
  gets = 0;
  writes = [];
  read = async () => [];
  write = async () => ({});
  request.get = async <T,>(url: string) => {
    assert.equal(url, '/v1/model_favorites');
    gets++;
    return (await read()) as T;
  };
  request.put = async <T,>(url: string, body: unknown) => {
    assert.equal(url, '/v1/model_favorites');
    writes.push({ method: 'PUT', body });
    return (await write()) as T;
  };
  request.delete = async <T,>(url: string, config?: { data?: unknown }) => {
    assert.equal(url, '/v1/model_favorites');
    writes.push({ method: 'DELETE', body: config?.data });
    return (await write()) as T;
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  Object.assign(request, original);
});
async function render() {
  await act(async () => root.render(<Probe />));
}

it('reads server favorites and persists individual changes before showing success', async () => {
  read = async () => ['existing'];
  await render();
  assert.deepEqual(view.favorites, ['existing']);
  await act(async () => {
    assert.equal(await view.toggle('org/model'), true);
  });
  assert.deepEqual(view.favorites, ['existing', 'org/model']);
  await act(async () => {
    await view.toggle('existing');
  });
  assert.deepEqual(view.favorites, ['org/model']);
  assert.deepEqual(writes, [
    { method: 'PUT', body: { model_name: 'org/model' } },
    { method: 'DELETE', body: { model_name: 'existing' } },
  ]);
  assert.equal(window.localStorage.getItem('collectionArr'), null);
});

it('keeps failed writes unchanged and blocks duplicate clicks while pending', async () => {
  await render();
  let fail!: (error: Error) => void;
  write = () =>
    new Promise((_, reject) => {
      fail = reject;
    });
  let saved!: Promise<boolean>;
  await act(async () => {
    saved = view.toggle('wanted');
  });
  assert.deepEqual(view.pending, ['wanted']);
  await act(async () => {
    assert.equal(await view.toggle('wanted'), false);
  });
  await act(async () => {
    fail(new Error('offline'));
    assert.equal(await saved, false);
  });
  assert.equal(writes.length, 1);
  assert.deepEqual(view.favorites, []);
  assert.deepEqual(view.pending, []);
});

it('does not overwrite simultaneous changes to different models', async () => {
  await render();
  const finish: (() => void)[] = [];
  write = () =>
    new Promise((resolve) => {
      finish.push(() => resolve({}));
    });
  let first!: Promise<boolean>, second!: Promise<boolean>;
  await act(async () => {
    first = view.toggle('first');
    second = view.toggle('second');
  });
  await act(async () => {
    finish[1]();
    await second;
  });
  await act(async () => {
    finish[0]();
    await first;
  });
  assert.deepEqual([...view.favorites].sort(), ['first', 'second']);
});

it('ignores a late account read and clears favorites immediately on logout', async () => {
  let old!: (names: string[]) => void;
  read = () =>
    new Promise((resolve) => {
      old = resolve;
    });
  await render();
  assert.equal(view.ready, false);
  read = async () => ['account-two'];
  await act(async () => setAccessToken(token(2)));
  await act(async () => old(['account-one']));
  assert.deepEqual(view.favorites, ['account-two']);
  await act(async () => removeAuthTokens());
  assert.deepEqual(view.favorites, []);
  assert.equal(view.ready, false);
});

it('ignores a late write after an account switch', async () => {
  await render();
  let finish!: (value: unknown) => void;
  write = () =>
    new Promise((resolve) => {
      finish = resolve;
    });
  let saved!: Promise<boolean>;
  await act(async () => {
    saved = view.toggle('account-one');
  });
  read = async () => ['account-two'];
  await act(async () => setAccessToken(token(2)));
  await act(async () => {
    finish({});
    assert.equal(await saved, false);
  });
  assert.deepEqual(view.favorites, ['account-two']);
});

it('waits for authentication and retries a failed initial read on focus', async () => {
  authenticated = undefined;
  await render();
  assert.equal(gets, 0);
  authenticated = true;
  read = async () => {
    throw new Error('offline');
  };
  await render();
  assert.equal(view.ready, false);
  await act(async () => {
    assert.equal(await view.toggle('wanted'), false);
  });
  assert.deepEqual(writes, []);
  read = async () => ['restored'];
  await act(async () => window.dispatchEvent(new Event('focus')));
  assert.deepEqual(view.favorites, ['restored']);
  assert.equal(view.ready, true);
  read = async () => {
    throw new Error('offline again');
  };
  await act(async () => window.dispatchEvent(new Event('focus')));
  assert.deepEqual(view.favorites, ['restored']);
  assert.equal(view.ready, false);
});

it('requires explicit import of unscoped browser favorites and retains them on failure', async () => {
  window.localStorage.setItem('collectionArr', JSON.stringify(['old', 'old', 'other']));
  await render();
  assert.deepEqual(view.favorites, []);
  assert.equal(view.hasBrowserFavorites, true);
  assert.deepEqual(writes, []);
  write = async () => {
    throw new Error('offline');
  };
  await act(async () => view.importBrowserFavorites());
  assert.ok(window.localStorage.getItem('collectionArr'));
  assert.equal(view.importing, false);
  write = async () => ({});
  await act(async () => view.importBrowserFavorites());
  assert.deepEqual(view.favorites, ['old', 'other']);
  assert.equal(window.localStorage.getItem('collectionArr'), null);
  assert.equal(view.hasBrowserFavorites, false);
});

it('stops browser import after switching accounts without removing the source', async () => {
  window.localStorage.setItem('collectionArr', JSON.stringify(['first', 'second']));
  await render();
  let finish!: (value: unknown) => void;
  write = () =>
    new Promise((resolve) => {
      finish = resolve;
    });
  let imported!: Promise<void>;
  await act(async () => {
    imported = view.importBrowserFavorites();
  });
  await act(async () => setAccessToken(token(2)));
  await act(async () => {
    finish({});
    await imported;
  });
  assert.equal(writes.length, 1);
  assert.ok(window.localStorage.getItem('collectionArr'));
  assert.deepEqual(view.favorites, []);
});

it('supports the shared anonymous collection without importing old data automatically', async () => {
  authenticated = false;
  removeAuthTokens();
  window.localStorage.setItem('collectionArr', JSON.stringify(['old']));
  read = async () => ['shared'];
  await render();
  assert.deepEqual(view.favorites, ['shared']);
  assert.equal(writes.length, 0);
});
