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
