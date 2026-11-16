# Xinference Frontend

This is the Xinference Web UI, a Next.js app. In production it is built as a
**static export**, bundled into the `xinference` Python package, and served by
the Xinference backend itself — a single process, no Node.js runtime needed.

## Local Development

Start the backend first:

```bash
xinference-local --host 127.0.0.1 --port 9997
```

Then start the frontend dev server:

```bash
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:3999.

`.env.development` points the browser straight at `http://127.0.0.1:9997`.
Keep it that way unless you know you want the proxy: with `NEXT_PUBLIC_API_URL`
unset the browser talks to the Next dev server instead, and its rewrite proxy
buffers responses — streaming endpoints (chat completions and friends) then
deliver the whole answer in one go, which looks like the backend lost its
streaming support.

To use a different backend endpoint:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:6735 npm run dev
```

`XINFERENCE_API_URL` sets the proxy target instead, for the same-origin setup
where the browser goes through the dev server.

## Build

```bash
npm run build
```

This produces the static export in `out/` and stages it at
`xinference/ui/web/dist` (postbuild hook), where the backend serves it. After
building, `http://<backend-host>:9997/` serves the UI directly.

Notes on the static export:

- Dynamic routes (`launch-model/[modelType]`, `running-model/[modelUid]`, ...)
  are emitted as `__shell__` placeholder pages; the backend maps any real URL
  to the matching shell and the client reads the actual params from the URL
  (see `src/lib/route-params.ts` and `xinference/api/frontend_static.py`).
- `next.config.ts` only enables rewrites (the dev API proxy) outside export
  mode; in production the UI is same-origin with the API, so no proxy is
  needed.
- To build a self-contained Node server instead (no static export), use
  `NEXT_OUTPUT=standalone npm run build`.

## Distribution request-error handlers

A distribution can replace src/lib/request-extensions.ts at build time, exporting
requestErrorExtensions as a readonly RequestErrorExtension array. The community
registry is empty. Import types from @/lib/request-extension-types and register
statuses, event, and a React Handler receiving { event } (null while idle).
Events must be unique and follow extension:<distribution>:<name>, using lowercase
letters, digits and hyphens in both names. Multiple statuses may share one entry;
if status lists overlap, the first matching entry wins.

Token refresh runs first. Built-in 401, 403 and client-cancelled 499 handling is
reserved; authentication-ban 429 responses are detected before extension dispatch.
Other 429 responses may be handled by extensions. suppressGlobalError skips
extension dispatch while the request still rejects. Request-specific errorMessages
supply the event message and preserve the existing 429 ban-message override.

Call event.release() after a toast/action finishes or when a dialog closes. Each
event shares a lock across concurrent requests. Locks expire after 30 seconds as
a fallback for missed subscriptions or handlers that forget to release; dialogs
lasting longer may receive another event. Stale release callbacks cannot unlock a
newer event. Listener unmount also releases the active event.

RequestExtensionHandlers is mounted inside I18nProvider, GlobalProvider and
ThemeProvider alongside Toaster in the root layout. Handlers can use those contexts
and toast APIs, but cannot assume page-local providers exist. Each handler has an
error boundary: rendering failures disable that handler until remount and leave
the app running. Custom layouts must retain RequestProvider for built-in events
and mount RequestExtensionHandlers within the desired app contexts.

## Custom user authorization

The `/user-management` route loads
`src/components/pages/user-management/page-entry.tsx`. Replace that entry at build
time to provide custom authorization fields and actions while reusing the account
list, password changes, enabled state and deletion UI. For example, an entry can
import a locally supplied authorization configuration:

```tsx
import UserManagement from './index';
import { userAuthorization } from './user-authorization';

export default function UserManagementPageEntry() {
  return <UserManagement authorization={userAuthorization} />;
}
```

`user-authorization` is supplied by the integration and exports a
`UserAuthorizationConfig` (type exported from `./authorization`). Omitting the
`authorization` prop keeps the built-in permission selector, badges and editing
workflow, including the existing `/v1/admin/users` creation request. Providing it
replaces only the authorization-specific UI and creation handler; it does not add
a role-management API to the backend.

The configuration includes:

- `columnTitle` and optional `description`: authorization column heading and page
  subtitle. Omitting `description` preserves the default subtitle.
- `createFields`: extra fields rendered inside the shared creation `Form`. Use
  `FormField` with unique names so validation and reset-on-reopen work. Do not
  store these form values separately or redefine the built-in account fields.
- `createDisabled`: optionally disable creation until required data, such as role
  options, is available.
- `onCreateUser(values)`: create the account and apply its authorization. Values
  contain `username`, `password`, `confirm_password`, and custom `FormField`
  values. The default create request is not also sent. Report errors through the
  request interceptor or your own toast, then reject to retain the draft and allow
  retry. Resolve only when creation succeeds.
- `renderUser(user, context)` and `renderActions(user, context)`: render the summary
  and authorization actions. Both receive `context.onSuccess()` to refresh the
  user list and `context.isCurrentUser`, which is `true` for the signed-in user,
  `false` for another user, and `null` when the token identity cannot be decoded.

Renderers own their action visibility. A role editor that must not modify the
signed-in account can return no action unless `context.isCurrentUser === false`.
These display rules are not authorization enforcement: the backend must validate
all account, role and permission mutations. Built-in permission and deletion
policies are unchanged by supplying a custom authorization renderer.

Run the authorization regression suite with `npm run test:user-management`.
