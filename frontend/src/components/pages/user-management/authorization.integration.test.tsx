/* eslint-disable @typescript-eslint/no-require-imports -- Initialize jsdom before loading UI. */
import assert from 'node:assert/strict';
import { it } from 'node:test';
import React, { act } from 'react';
// @ts-expect-error -- jsdom has no declarations in this dependency tree.
import { JSDOM } from 'jsdom';
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
Object.assign(globalThis, {
  window: dom.window,
  sessionStorage: dom.window.sessionStorage,
  localStorage: dom.window.localStorage,
  document: dom.window.document,
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
    { PermissionSelector: () => <div data-permissions="true" /> },
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
        React.useContext(State).open ? <section>{children}</section> : null,
      DialogHeader: wrapper,
      DialogTitle: wrapper,
      DialogFooter: wrapper,
    },
  ],
] as const) {
  require(name);
  require.cache[require.resolve(name)]!.exports = exports;
}
require('sonner').toast.success = () => {};
const request = require('@/lib/request').default as typeof import('@/lib/request').default;
const { CreateUserDialog } =
  require('./create-user-dialog') as typeof import('./create-user-dialog');
const { FormField } =
  require('@/components/ui/form-field') as typeof import('@/components/ui/form-field');
const { Input } = require('@/components/ui/input') as typeof import('@/components/ui/input');
async function click(host: HTMLElement, text: string) {
  const button = [...host.querySelectorAll('button')].find((item) => item.textContent === text)!;
  assert.ok(button, text);
  await act(async () => button.click());
}
async function fill(input: HTMLInputElement, value: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      value
    );
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
for (const mode of [undefined, 'permissions', 'roles'] as const) {
  const custom = mode === 'roles';
  it(
    custom
      ? 'uses injected authorization fields and submit handler'
      : 'preserves the community create API with mode ' + String(mode),
    async () => {
      const original = request.post;
      let standard: unknown,
        alternate: unknown,
        completed = 0;
      request.post = async <T,>(url: string, body: unknown) => {
        assert.equal(url, '/v1/admin/users');
        standard = body;
        return {} as T;
      };
      const host = document.createElement('div');
      document.body.append(host);
      const root = createRoot(host);
      try {
        await act(async () =>
          root.render(
            <CreateUserDialog
              trigger={<button>Open</button>}
              onSuccess={() => {
                completed++;
              }}
              {...(custom
                ? {
                    authorizationMode: 'roles' as const,
                    authorizationConfig: {
                      createFields: (
                        <FormField
                          name="group"
                          rules={[{ required: true, message: 'Choose a group' }]}
                        >
                          <Input />
                        </FormField>
                      ),
                      onCreateUser: async (values: Record<string, unknown>) => {
                        alternate = values;
                      },
                    },
                  }
                : mode === 'permissions'
                  ? { authorizationMode: 'permissions' as const }
                  : {})}
            />
          )
        );
        await click(host, 'Open');
        assert.equal(Boolean(host.querySelector('[data-permissions]')), !custom);
        const inputs = host.querySelectorAll<HTMLInputElement>('input');
        await fill(inputs[0], 'alice');
        await fill(inputs[1], 'test-password');
        await fill(inputs[2], 'test-password');
        if (custom) {
          await click(host, 'userManagement.create');
          assert.equal(completed, 0);
          await fill(inputs[3], 'researchers');
        }
        await click(host, 'userManagement.create');
        assert.equal(completed, 1);
        if (custom) {
          assert.equal(standard, undefined);
          assert.equal((alternate as { group: string }).group, 'researchers');
        } else
          assert.deepEqual(standard, {
            username: 'alice',
            password: 'test-password',
            permissions: [],
          });
        assert.equal(host.querySelector('form'), null);
      } finally {
        await act(async () => root.unmount());
        host.remove();
        request.post = original;
      }
    }
  );
}

it('uses role renderers while hiding actions only for the signed-in user', async () => {
  const UserManagement = require('./index').default as typeof import('./index').default;
  const originalGet = request.get;
  sessionStorage.setItem(
    'token',
    'header.' + Buffer.from(JSON.stringify({ user_id: 2 })).toString('base64url') + '.signature'
  );
  request.get = async <T,>() =>
    [3, 2].map((id) => ({
      id,
      username: 'user-' + id,
      source: 'local',
      enabled: true,
      must_change_password: false,
      permissions: [],
      created_at: '',
    })) as T;
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  try {
    await act(async () =>
      root.render(
        <UserManagement
          authorizationMode="roles"
          authorizationConfig={{
            columnTitle: 'Role',
            createFields: null,
            onCreateUser: async () => {},
            renderUser: (user) => <span>Role of {user.username}</span>,
            renderActions: (user) => <button data-role-user={user.id}>Edit role</button>,
          }}
        />
      )
    );
    assert.ok(host.textContent?.includes('Role of user-2'));
    assert.ok(host.querySelector('[data-role-user="3"]'));
    assert.equal(host.querySelector('[data-role-user="2"]'), null);
  } finally {
    await act(async () => root.unmount());
    host.remove();
    request.get = originalGet;
    sessionStorage.removeItem('token');
  }
});
