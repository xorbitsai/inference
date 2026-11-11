/* eslint-disable @typescript-eslint/no-require-imports -- Load browser components after jsdom. */
import assert from 'node:assert/strict';
import { afterEach, beforeEach, it } from 'node:test';
import { createServer } from 'node:http';
import React, { act } from 'react';
import type { Root } from 'react-dom/client';
import type { RequestModelType } from './types';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  Event: dom.window.Event,
  CustomEvent: dom.window.CustomEvent,
  KeyboardEvent: dom.window.KeyboardEvent,
  IS_REACT_ACT_ENVIRONMENT: true,
  requestAnimationFrame: (callback: FrameRequestCallback) => setTimeout(callback, 0),
  cancelAnimationFrame: (id: number) => clearTimeout(id),
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
for (const key of Reflect.ownKeys(dom.window)) {
  if (!(key in globalThis))
    Object.defineProperty(globalThis, key, Object.getOwnPropertyDescriptor(dom.window, key)!);
}
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const chinese = require('@/i18n/locales/zh').default as Record<string, unknown>;
let translateChinese = false;
require('@/contexts/i18n-context');
require.cache[require.resolve('@/contexts/i18n-context')]!.exports = {
  useI18n: () => ({
    t: (key: string) => {
      if (!translateChinese) return key;
      const value = key
        .split('.')
        .reduce<unknown>(
          (value, part) =>
            typeof value === 'object' && value !== null
              ? (value as Record<string, unknown>)[part]
              : undefined,
          chinese
        );
      return typeof value === 'string' ? value : key;
    },
  }),
};
let permissions: string[] | null;
let authenticated: boolean;
let ready: boolean;
require('@/contexts/global-context');
require.cache[require.resolve('@/contexts/global-context')]!.exports = {
  useGlobal: () => ({
    accountPermissions: permissions,
    accountSource: 'local',
    clusterUIConfig: { auth_advanced: true },
    clusterAuth: { auth: authenticated },
    globalReady: ready,
  }),
};
const { ModelType } = require('@/constants') as typeof import('@/constants');
const { LAUNCH_MODEL_UPDATE_OPTIONS } =
  require('@/constants/launch') as typeof import('@/constants/launch');
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { toast } = require('sonner') as typeof import('sonner');
const CatalogRefresh = require('./catalog-refresh')
  .default as typeof import('./catalog-refresh').default;
const navigation = require('next/navigation') as typeof import('next/navigation');
let navigated: string[];
require.cache[require.resolve('next/navigation')]!.exports = {
  ...navigation,
  useRouter: () => ({ push: (path: string) => navigated.push(path) }),
};
// Keep the page/list integration real; engine launch and cache dialogs are unrelated.
for (const path of [
  './launch-dialog/launch-dialog',
  './cache-management.dialog',
  './env-management-dialog',
  './custom-edit-dialog',
]) {
  const filename = require.resolve(path);
  require.cache[filename] = {
    id: filename,
    filename,
    loaded: true,
    exports: { __esModule: true, default: () => null },
  } as NodeModule;
}
const LaunchModel = require('./index').default as typeof import('./index').default;
let root: Root;
let host: HTMLDivElement;
let posts: { url: string; body: unknown }[];
let gets: string[];
let refreshed: string[];
let successes: number;
let catalog: Record<string, unknown>[];
let respond: () => Promise<unknown>;
const originalPost = request.post,
  originalGet = request.get,
  originalSuccess = toast.success;
beforeEach(() => {
  translateChinese = false;
  permissions = ['models:register'];
  authenticated = true;
  ready = true;
  posts = [];
  gets = [];
  refreshed = [];
  navigated = [];
  successes = 0;
  catalog = [{ model_name: 'wanted', model_description: 'Before update', is_builtin: true }];
  respond = async () => ({});
  window.sessionStorage.clear();
  window.localStorage.clear();
  request.post = async <T,>(url: string, body: unknown) => {
    posts.push({ url, body });
    return (await respond()) as T;
  };
  request.get = async <T,>(url: string) => {
    gets.push(url);
    return (
      url.startsWith('/v1/model_registrations/')
        ? catalog
        : url === '/v1/cluster/devices'
          ? '0'
          : {}
    ) as T;
  };
  toast.success = () => {
    successes++;
    return 'updated';
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  request.post = originalPost;
  request.get = originalGet;
  toast.success = originalSuccess;
});
async function render(modelType: RequestModelType = ModelType.LLM) {
  await act(async () =>
    root.render(
      <CatalogRefresh
        key={modelType}
        modelType={modelType}
        onModelTypeChange={() => {}}
        onRefreshed={(type) => refreshed.push(type)}
      />
    )
  );
}
function singleButton() {
  return [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'launchModel.updateOneModel'
  )!;
}
async function openDialog() {
  await act(async () => singleButton().click());
}
async function typeName(value: string) {
  await act(async () => {
    const input = document.querySelector<HTMLInputElement>('[role="dialog"] input')!;
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      value
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
function submit() {
  return document.querySelector<HTMLButtonElement>('[role="dialog"] button[type="submit"]')!;
}
function categoryButton() {
  return [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'common.update'
  )!;
}
function categorySelect() {
  return categoryButton().parentElement!.firstElementChild!.firstElementChild as HTMLDivElement;
}
async function selectCategory(label: string) {
  await act(async () => categorySelect().click());
  const option = [
    ...document.querySelectorAll<HTMLButtonElement>('[data-slot="select-option"]'),
  ].find((button) => button.textContent === label)!;
  await act(async () => option.click());
}

for (const { value, label } of LAUNCH_MODEL_UPDATE_OPTIONS) {
  it(`updates a named ${value} model through the single-model API`, async () => {
    await render(value as RequestModelType);
    await openDialog();
    assert.equal(
      document.querySelector('[role="dialog"] form p')?.textContent,
      `launchModel.modelType: ${label}`
    );
    assert.ok(submit().disabled);
    await typeName('  org/model & variant  ');
    await act(async () => submit().click());
    assert.deepEqual(posts, [
      {
        url: '/v1/models/update_spec',
        body: { model_type: value.toLowerCase(), model_name: 'org/model & variant' },
      },
    ]);
    assert.deepEqual(refreshed, [value]);
    assert.equal(successes, 1);
    assert.equal(document.querySelector('[role="dialog"]'), null);
  });
}

it('keeps whole-category refresh on its existing endpoint and payload', async () => {
  await render();
  const button = [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'common.update'
  )!;
  await act(async () => button.click());
  assert.deepEqual(posts, [{ url: '/v1/models/update_type', body: { model_type: 'llm' } }]);
  assert.deepEqual(refreshed, [ModelType.LLM]);
});

it('rejects blank names, submits Enter once and blocks dismissal while pending', async () => {
  let finish!: () => void;
  respond = () =>
    new Promise<void>((resolve) => {
      finish = resolve;
    });
  await render();
  await openDialog();
  await typeName('   ');
  assert.ok(submit().disabled);
  await act(async () =>
    document
      .querySelector('form')!
      .dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
  );
  assert.deepEqual(posts, []);
  await typeName('wanted');
  await act(async () => {
    const form = document.querySelector('form')!;
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
  assert.equal(posts.length, 1);
  assert.ok(submit().disabled);
  assert.ok(document.querySelector<HTMLInputElement>('[role="dialog"] input')!.disabled);
  await act(async () =>
    document
      .querySelector('[role="dialog"]')!
      .dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
  );
  assert.ok(document.querySelector('[role="dialog"]'));
  await act(async () => finish());
  assert.equal(refreshed.length, 1);
});

it('retains input on request failure without falling back to category refresh', async () => {
  respond = async () => {
    throw new Error('Catalog refresh failed');
  };
  await render();
  await openDialog();
  await typeName('wanted');
  await act(async () => submit().click());
  assert.equal(document.querySelector<HTMLInputElement>('[role="dialog"] input')!.value, 'wanted');
  assert.ok(!submit().disabled);
  assert.equal(refreshed.length, 0);
  assert.equal(successes, 0);
  assert.deepEqual(
    posts.map((post) => post.url),
    ['/v1/models/update_spec']
  );
  respond = async () => ({});
  await act(async () => submit().click());
  assert.equal(refreshed.length, 1);
  assert.equal(posts.length, 2);
  assert.ok(posts.every((post) => post.url === '/v1/models/update_spec'));
});

it('accepts registration scope aliases and respects pending/revoked live permissions', async () => {
  for (const scopes of [null, [], ['models:read']]) {
    permissions = scopes;
    await render();
    assert.equal(host.querySelector('button'), null);
  }
  for (const scopes of [['models:register'], ['models:add'], ['models:unregister'], ['admin']]) {
    permissions = scopes;
    await render();
    assert.ok(singleButton());
    assert.ok(categoryButton());
  }
  ready = false;
  permissions = ['admin'];
  await render();
  assert.equal(host.querySelector('button'), null);
  ready = true;
  await render();
  assert.ok(singleButton());
  let finish!: () => void;
  respond = () =>
    new Promise<void>((resolve) => {
      finish = resolve;
    });
  await openDialog();
  await typeName('wanted');
  await act(async () => submit().click());
  permissions = [];
  await render();
  assert.equal(document.querySelector('[role="dialog"]'), null);
  await act(async () => finish());
  assert.equal(successes, 0);
  assert.deepEqual(refreshed, []);
});

it('allows public no-auth refresh and ignores completion after a category change', async () => {
  authenticated = false;
  permissions = [];
  let finish!: () => void;
  respond = () =>
    new Promise<void>((resolve) => {
      finish = resolve;
    });
  await render();
  await openDialog();
  await typeName('wanted');
  await act(async () => submit().click());
  await render(ModelType.Embedding);
  await act(async () => finish());
  assert.deepEqual(refreshed, []);
  assert.equal(successes, 0);
  assert.ok(!singleButton().disabled);
});

it('renders catalog actions from the Chinese dictionary', async () => {
  translateChinese = true;
  await act(async () =>
    root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
  );
  const labels = [...host.querySelectorAll('button')].map((button) => button.textContent?.trim());
  assert.ok(labels.includes('模型详情'));
  assert.ok(labels.includes('启动'));
});

for (const single of [false, true]) {
  it(`reloads the visible list after ${single ? 'single-model' : 'category'} refresh`, async () => {
    respond = async () => {
      catalog = [{ model_name: 'wanted', model_description: 'After update', is_builtin: true }];
      return {};
    };
    await act(async () =>
      root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
    );
    assert.match(host.textContent || '', /Before update/);
    if (single) {
      await openDialog();
      await typeName('wanted');
      await act(async () => submit().click());
    } else {
      await act(async () =>
        [...host.querySelectorAll<HTMLButtonElement>('button')]
          .find((button) => button.textContent === 'common.update')!
          .click()
      );
    }
    assert.match(host.textContent || '', /After update/);
    assert.deepEqual(
      gets.filter((url) => url.startsWith('/v1/model_registrations/')),
      ['/v1/model_registrations/LLM?detailed=true', '/v1/model_registrations/LLM?detailed=true']
    );
    assert.deepEqual(navigated, []);
  });

  it(`ignores ${single ? 'single-model' : 'category'} completion after changing the page route`, async () => {
    let finish!: () => void;
    respond = () =>
      new Promise<void>((resolve) => {
        finish = resolve;
      });
    await act(async () =>
      root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
    );
    if (single) {
      await openDialog();
      await typeName('wanted');
      await act(async () => submit().click());
    } else {
      await act(async () => categoryButton().click());
    }
    assert.equal(posts.length, 1);
    await act(async () =>
      root.render(<LaunchModel routeType={ModelType.Embedding} initialCustomType={ModelType.LLM} />)
    );
    const pageRequests = [...gets];
    await act(async () => finish());
    assert.equal(document.querySelector('[role="dialog"]'), null);
    assert.deepEqual(navigated, []);
    assert.equal(successes, 0);
    assert.deepEqual(gets, pageRequests);
    assert.ok(!categoryButton().disabled);
  });

  it(`navigates to a different built-in type after ${single ? 'single-model' : 'category'} refresh`, async () => {
    await act(async () =>
      root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
    );
    const select = categorySelect();
    const button = categoryButton();
    button.focus();
    await selectCategory('Embedding');
    assert.ok(categorySelect() === select, 'changing type must preserve the category selector');
    assert.ok(document.activeElement === button, 'changing type must preserve existing focus');
    if (single) {
      await openDialog();
      await typeName('wanted');
      await act(async () => submit().click());
    } else {
      await act(async () => categoryButton().click());
    }
    assert.deepEqual(posts, [
      {
        url: single ? '/v1/models/update_spec' : '/v1/models/update_type',
        body: { model_type: 'embedding', ...(single ? { model_name: 'wanted' } : {}) },
      },
    ]);
    assert.deepEqual(navigated, ['/launch-model/embedding']);
    assert.equal(successes, 1);
  });
}

it('blocks type changes during category refresh and re-enables the controls after failure', async () => {
  let fail!: (reason: Error) => void;
  respond = () =>
    new Promise((_, reject) => {
      fail = reject;
    });
  await act(async () =>
    root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
  );
  await act(async () => categoryButton().click());
  assert.ok(categoryButton().disabled);
  assert.ok(singleButton().disabled);
  await act(async () => categorySelect().click());
  assert.equal(document.querySelector('[data-slot="select-dropdown"]'), null);
  await act(async () => fail(new Error('Category refresh failed')));
  assert.ok(!categoryButton().disabled);
  assert.ok(!singleButton().disabled);
  await selectCategory('Embedding');
  assert.equal(categorySelect().textContent, 'Embedding');
  assert.deepEqual(posts, [{ url: '/v1/models/update_type', body: { model_type: 'llm' } }]);
  assert.deepEqual(navigated, []);
  assert.equal(successes, 0);
});

it('moves from custom registrations to the refreshed built-in category', async () => {
  await act(async () =>
    root.render(<LaunchModel routeType={ModelType.Custom} initialCustomType={ModelType.LLM} />)
  );
  await openDialog();
  await typeName('wanted');
  await act(async () => submit().click());
  assert.deepEqual(navigated, ['/launch-model/llm']);
});

it('labels featured models independently of favorites and explains the badge on focus', async () => {
  authenticated = false;
  catalog = [
    { model_name: 'ordinary', is_builtin: true },
    { model_name: 'favorite-only', is_builtin: true, featured: false },
    { model_name: 'recommended', is_builtin: true, featured: true },
  ];
  const beforeGet = request.get;
  request.get = async <T,>(url: string) =>
    (url === '/v1/model_favorites' ? ['favorite-only'] : await beforeGet<T>(url)) as T;
  await act(async () =>
    root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
  );
  const cards = [...host.querySelectorAll('h3')].map(
    (heading) => heading.parentElement!.parentElement!
  );
  assert.deepEqual(
    cards.map((card) => card.querySelector('h3')!.textContent),
    ['recommended', 'favorite-only', 'ordinary']
  );
  const badge = cards[0].querySelector<HTMLSpanElement>('span[tabindex="0"]')!;
  assert.equal(badge.textContent, 'launchModel.featuredModel');
  assert.equal(
    cards[0].querySelector('[aria-label="Favorite model"]')!.getAttribute('aria-pressed'),
    'false'
  );
  assert.equal(
    cards[1].querySelector('[aria-label="Favorite model"]')!.getAttribute('aria-pressed'),
    'true'
  );
  for (const card of cards.slice(1)) {
    assert.ok(!card.textContent!.includes('launchModel.featuredModel'));
  }
  await act(async () => badge.focus());
  assert.equal(
    document.querySelector('[role="tooltip"]')?.textContent,
    'launchModel.featuredModelDescription'
  );
});

it('keeps featured badges off custom registrations', async () => {
  catalog = [{ model_name: 'custom', is_builtin: false, featured: true }];
  await act(async () =>
    root.render(<LaunchModel routeType={ModelType.Custom} initialCustomType={ModelType.LLM} />)
  );
  assert.equal(host.querySelector('h3')?.textContent, 'custom');
  assert.ok(!host.textContent!.includes('launchModel.featuredModel'));
});

it('connects the catalog star to the current account favorites API', async () => {
  window.sessionStorage.setItem(
    'token',
    'header.' + Buffer.from(JSON.stringify({ user_id: 1 })).toString('base64url') + '.signature'
  );
  const beforeGet = request.get,
    beforePut = request.put,
    beforeDelete = request.delete;
  const changes: { method: string; body: unknown }[] = [];
  request.get = async <T,>(url: string) =>
    (url === '/v1/model_favorites' ? ['wanted'] : await beforeGet<T>(url)) as T;
  request.put = async <T,>(url: string, body: unknown) => {
    assert.equal(url, '/v1/model_favorites');
    changes.push({ method: 'PUT', body });
    return {} as T;
  };
  request.delete = async <T,>(url: string, config?: { data?: unknown }) => {
    assert.equal(url, '/v1/model_favorites');
    changes.push({ method: 'DELETE', body: config?.data });
    return {} as T;
  };
  try {
    await act(async () =>
      root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
    );
    const star = host.querySelector<HTMLButtonElement>('button[aria-label="Favorite model"]')!;
    assert.equal(star.getAttribute('aria-pressed'), 'true');
    assert.equal(star.disabled, false);
    await act(async () => star.click());
    assert.equal(star.getAttribute('aria-pressed'), 'false');
    await act(async () => star.click());
    assert.equal(star.getAttribute('aria-pressed'), 'true');
    assert.deepEqual(changes, [
      { method: 'DELETE', body: { model_name: 'wanted' } },
      { method: 'PUT', body: { model_name: 'wanted' } },
    ]);
  } finally {
    request.get = beforeGet;
    request.put = beforePut;
    request.delete = beforeDelete;
  }
});

it('explains that anonymous browser imports are shared with all visitors', async () => {
  authenticated = false;
  window.localStorage.setItem('collectionArr', JSON.stringify(['wanted']));
  const beforeGet = request.get;
  request.get = async <T,>(url: string) =>
    (url === '/v1/model_favorites' ? [] : await beforeGet<T>(url)) as T;
  try {
    await act(async () =>
      root.render(<LaunchModel routeType={ModelType.LLM} initialCustomType={ModelType.LLM} />)
    );
    const button = [...host.querySelectorAll('button')].find(
      (b) => b.textContent === 'launchModel.importFavorites'
    )!;
    assert.equal(button.getAttribute('title'), 'launchModel.importFavoritesSharedHint');
    assert.equal(button.disabled, false);
  } finally {
    request.get = beforeGet;
  }
});

it('uses the shared request client to send the session header and only catalog fields', async () => {
  const observed: { path: string; authorization?: string; body: unknown }[] = [];
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', () => {
      observed.push({
        path: req.url!,
        authorization: req.headers.authorization,
        body: JSON.parse(body),
      });
      res.setHeader('Content-Type', 'application/json');
      res.end('{}');
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === 'object');
    const endpoint = `http://127.0.0.1:${address.port}`;
    window.sessionStorage.setItem('token', 'session-test-token');
    let response: Promise<unknown>;
    request.post = <T,>(url: string, body: unknown) => {
      const pending = originalPost<T>(endpoint + url, body, { adapter: 'http', proxy: false });
      response = pending;
      return pending;
    };
    await render();
    await openDialog();
    await typeName('wanted');
    await act(async () => {
      submit().click();
      await response;
    });
    assert.deepEqual(observed, [
      {
        path: '/v1/models/update_spec',
        authorization: 'Bearer session-test-token',
        body: { model_type: 'llm', model_name: 'wanted' },
      },
    ]);
    assert.equal(refreshed.length, 1);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
