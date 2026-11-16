/* eslint-disable @typescript-eslint/no-require-imports -- Initialize jsdom before loading UI. */
import assert from 'node:assert/strict';
import { it } from 'node:test';
import React, { act } from 'react';
import type { UserAuthorizationConfig, UserAuthorizationContext } from './authorization';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  sessionStorage: dom.window.sessionStorage,
  localStorage: dom.window.localStorage,
  Event: dom.window.Event,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: dom.window.navigator });
const { createRoot } = require('react-dom/client') as typeof import('react-dom/client');
const State = React.createContext({ open: false, change: (_value: boolean) => {} });
const wrapper = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
for (const [name, exports] of [
  ['@/contexts/i18n-context', { useI18n: () => ({ t: (key: string) => key }) }],
  [
    '@/components/pages/user-management/permissions',
    {
      PermissionSelector: () => <div data-permissions="true" />,
      PermissionBadges: ({ permissions }: { permissions: string[] }) => (
        <span data-permission-badges>{permissions.join(',')}</span>
      ),
    },
  ],
  [
    '@/components/ui/dialog',
    {
      Dialog: ({
        open,
        onOpenChange,
        children,
      }: {
        open: boolean;
        onOpenChange: (value: boolean) => void;
        children: React.ReactNode;
      }) => <State.Provider value={{ open, change: onOpenChange }}>{children}</State.Provider>,
      DialogTrigger: ({ children }: { children: React.ReactElement<{ onClick?: () => void }> }) => {
        const state = React.useContext(State);
        return React.cloneElement(children, { onClick: () => state.change(true) });
      },
      DialogContent: ({ children }: { children: React.ReactNode }) =>
        React.useContext(State).open ? <section role="dialog">{children}</section> : null,
      DialogHeader: wrapper,
      DialogTitle: wrapper,
      DialogFooter: wrapper,
    },
  ],
] as const) {
  require(name);
  require.cache[require.resolve(name)]!.exports = exports;
}
const failures: string[] = [];
require('sonner').toast.success = () => {};
require('sonner').toast.error = (message: string) => failures.push(message);
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { CreateUserDialog } =
  require('./create-user-dialog') as typeof import('./create-user-dialog');
const UserManagement = require('./index').default as typeof import('./index').default;
const PageEntry = require('./page-entry').default as typeof import('./page-entry').default;
const { FormField } =
  require('@/components/ui/form-field') as typeof import('@/components/ui/form-field');
const { Input } = require('@/components/ui/input') as typeof import('@/components/ui/input');
const createFields = (
  <FormField name="group" rules={[{ required: true, message: 'Choose a group' }]}>
    <Input aria-label="Group" />
  </FormField>
);
async function mount(element: React.ReactElement) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await act(async () => root.render(element));
  return {
    host,
    root,
    dispose: async () => {
      await act(async () => root.unmount());
      host.remove();
    },
  };
}
async function click(host: HTMLElement, label: string) {
  const button = [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (item) => item.textContent === label
  )!;
  assert.ok(button, label);
  await act(async () => button.click());
}
async function fill(host: HTMLElement, selector: string, value: string) {
  const input = host.querySelector<HTMLInputElement>(selector);
  assert.ok(input, selector);
  await act(async () => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      value
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
async function accountFields(host: HTMLElement) {
  await fill(host, '[placeholder="userManagement.usernamePlaceholder"]', 'alice');
  await fill(host, '[placeholder="userManagement.passwordPlaceholder"]', 'test-password');
  await fill(host, '[placeholder="userManagement.confirmPasswordPlaceholder"]', 'test-password');
}
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

it('preserves the default permission selector and community creation request', async () => {
  const original = request.post;
  let body: unknown;
  let completed = 0;
  request.post = async <T,>(url: string, values: unknown) => {
    assert.equal(url, '/v1/admin/users');
    body = values;
    return {} as T;
  };
  const view = await mount(
    <CreateUserDialog
      trigger={<button>Open</button>}
      onSuccess={() => {
        completed++;
      }}
    />
  );
  try {
    await click(view.host, 'Open');
    assert.ok(view.host.querySelector('[data-permissions]'));
    await accountFields(view.host);
    await click(view.host, 'userManagement.create');
    assert.deepEqual(body, { username: 'alice', password: 'test-password', permissions: [] });
    assert.equal(completed, 1);
    assert.equal(view.host.querySelector('form'), null);
  } finally {
    await view.dispose();
    request.post = original;
  }
});

it('validates custom fields, delegates creation and resets fields when reopened', async () => {
  const original = request.post;
  let submitted: Record<string, unknown> | undefined;
  request.post = async () => {
    assert.fail('The default API must not run for custom creation');
  };
  const view = await mount(
    <CreateUserDialog
      trigger={<button>Open</button>}
      onSuccess={() => {}}
      authorization={{
        createFields,
        onCreateUser: async (values) => {
          submitted = values;
        },
      }}
    />
  );
  try {
    await click(view.host, 'Open');
    assert.equal(view.host.querySelector('[data-permissions]'), null);
    await accountFields(view.host);
    await click(view.host, 'userManagement.create');
    assert.equal(submitted, undefined);
    await fill(view.host, '[aria-label="Group"]', 'researchers');
    await click(view.host, 'userManagement.create');
    assert.deepEqual(submitted, {
      username: 'alice',
      password: 'test-password',
      confirm_password: 'test-password',
      group: 'researchers',
    });
    await click(view.host, 'Open');
    assert.equal(view.host.querySelector<HTMLInputElement>('[aria-label="Group"]')!.value, '');
    assert.equal(
      view.host.querySelector<HTMLInputElement>(
        '[placeholder="userManagement.usernamePlaceholder"]'
      )!.value,
      ''
    );
  } finally {
    await view.dispose();
    request.post = original;
  }
});

it('retains a failed custom creation draft, resets loading and supports retry', async () => {
  let attempts = 0;
  const view = await mount(
    <CreateUserDialog
      trigger={<button>Open</button>}
      onSuccess={() => {}}
      authorization={{
        createFields,
        onCreateUser: async () => {
          if (++attempts === 1) {
            require('sonner').toast.error('Creation rejected');
            throw new Error('Rejected');
          }
        },
      }}
    />
  );
  try {
    await click(view.host, 'Open');
    await accountFields(view.host);
    await fill(view.host, '[aria-label="Group"]', 'researchers');
    await click(view.host, 'userManagement.create');
    assert.equal(failures.at(-1), 'Creation rejected');
    assert.ok(view.host.querySelector('form'));
    assert.equal(
      view.host.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled,
      false
    );
    assert.equal(
      view.host.querySelector<HTMLInputElement>('[aria-label="Group"]')!.value,
      'researchers'
    );
    await click(view.host, 'userManagement.create');
    assert.equal(attempts, 2);
    assert.equal(view.host.querySelector('form'), null);
  } finally {
    await view.dispose();
  }
});

it('honors createDisabled and enables creation once authorization data is ready', async () => {
  let calls = 0;
  const render = (disabled: boolean) => (
    <CreateUserDialog
      trigger={<button>Open</button>}
      onSuccess={() => {}}
      authorization={{
        createFields,
        createDisabled: disabled,
        onCreateUser: async () => {
          calls++;
        },
      }}
    />
  );
  const view = await mount(render(true));
  try {
    await click(view.host, 'Open');
    await accountFields(view.host);
    await fill(view.host, '[aria-label="Group"]', 'researchers');
    await click(view.host, 'userManagement.create');
    assert.equal(calls, 0);
    await act(async () => view.root.render(render(false)));
    await click(view.host, 'userManagement.create');
    assert.equal(calls, 1);
  } finally {
    await view.dispose();
  }
});

it('can reopen during list refresh without an old refresh unlocking a later creation', async () => {
  const first = deferred(),
    second = deferred(),
    refresh = deferred();
  let creates = 0,
    refreshes = 0;
  const view = await mount(
    <CreateUserDialog
      trigger={<button>Open</button>}
      onSuccess={async () => {
        if (++refreshes === 1) await refresh.promise;
      }}
      authorization={{
        createFields,
        onCreateUser: async () => {
          await (++creates === 1 ? first.promise : second.promise);
        },
      }}
    />
  );
  try {
    await click(view.host, 'Open');
    await accountFields(view.host);
    await fill(view.host, '[aria-label="Group"]', 'one');
    await click(view.host, 'userManagement.create');
    assert.ok(view.host.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled);
    await act(async () => first.resolve());
    assert.equal(view.host.querySelector('form'), null);
    await click(view.host, 'Open');
    assert.ok(view.host.querySelector('form'));
    assert.equal(
      view.host.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled,
      false
    );
    await accountFields(view.host);
    await fill(view.host, '[aria-label="Group"]', 'two');
    await click(view.host, 'userManagement.create');
    await act(async () => refresh.resolve());
    assert.ok(view.host.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled);
    await act(async () => second.resolve());
    assert.equal(view.host.querySelector('form'), null);
  } finally {
    await act(async () => {
      first.resolve();
      second.resolve();
      refresh.resolve();
    });
    await view.dispose();
  }
});

const users = [3, 2].map((id) => ({
  id,
  username: 'user-' + id,
  source: 'local',
  enabled: true,
  must_change_password: false,
  permissions: ['models:list'],
  created_at: '',
}));
it('keeps the default page header, permission badges and permission editing actions', async () => {
  const original = request.get;
  request.get = async <T,>() => users as T;
  const view = await mount(<PageEntry />);
  try {
    assert.equal(view.host.querySelector('h1')!.textContent, 'menu.userManagement');
    assert.ok(view.host.textContent?.includes('userManagement.pageDescription'));
    assert.ok(
      [...view.host.querySelectorAll('th')].some(
        (item) => item.textContent === 'userManagement.permissions'
      )
    );
    assert.equal(view.host.querySelectorAll('[data-permission-badges]').length, 2);
    assert.equal(view.host.querySelectorAll('button[title="common.edit"]').length, 2);
  } finally {
    await view.dispose();
    request.get = original;
  }
});

for (const identity of ['known', 'invalid', 'missing-id'] as const) {
  const knownIdentity = identity === 'known';
  it(
    'passes row identity and refresh context to custom renderers with ' + identity + ' identity',
    async () => {
      const original = request.get;
      let reads = 0;
      sessionStorage.setItem(
        'token',
        knownIdentity
          ? 'header.' +
              Buffer.from(JSON.stringify({ user_id: 2 })).toString('base64url') +
              '.signature'
          : identity === 'missing-id'
            ? 'header.' +
              Buffer.from(JSON.stringify({ user_id: null })).toString('base64url') +
              '.signature'
            : 'invalid-token'
      );
      request.get = async <T,>() => {
        reads++;
        return users as T;
      };
      const summaries = new Map<number, boolean | null>();
      const actions = new Map<number, UserAuthorizationContext>();
      const authorization: UserAuthorizationConfig = {
        columnTitle: 'Assigned group',
        description: knownIdentity ? 'Custom authorization' : undefined,
        createFields,
        onCreateUser: async () => {},
        renderUser: (user, context) => {
          summaries.set(user.id, context.isCurrentUser);
          return <span>Group for {user.username}</span>;
        },
        renderActions: (user, context) => {
          actions.set(user.id, context);
          return context.isCurrentUser === false ? (
            <button data-group-user={user.id}>Edit group</button>
          ) : null;
        },
      };
      const view = await mount(<UserManagement authorization={authorization} />);
      try {
        assert.ok(
          [...view.host.querySelectorAll('th')].some(
            (item) => item.textContent === 'Assigned group'
          )
        );
        assert.ok(
          view.host.textContent?.includes(
            knownIdentity ? 'Custom authorization' : 'userManagement.pageDescription'
          )
        );
        assert.equal(summaries.get(2), knownIdentity ? true : null);
        assert.equal(summaries.get(3), knownIdentity ? false : null);
        assert.equal(actions.get(2)!.isCurrentUser, knownIdentity ? true : null);
        assert.equal(Boolean(view.host.querySelector('[data-group-user="3"]')), knownIdentity);
        assert.equal(view.host.querySelector('[data-group-user="2"]'), null);
        const before = reads;
        await act(async () => actions.get(3)!.onSuccess());
        assert.equal(reads, before + 1);
        if (knownIdentity) {
          await act(async () => {
            sessionStorage.setItem('token', 'invalid-token');
            window.dispatchEvent(new Event('xinference-auth-change'));
          });
          assert.equal(actions.get(2)!.isCurrentUser, null);
          assert.equal(view.host.querySelector('[data-group-user]'), null);
        }
      } finally {
        await view.dispose();
        request.get = original;
        sessionStorage.removeItem('token');
      }
    }
  );
}
