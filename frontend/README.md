# Xinference Frontend

This is the default Xinference Web UI. It runs as a separate Next.js app and
talks to a running Xinference backend.

## Local Development

Start the backend first:

```bash
xinference-local --host 127.0.0.1 --port 9997
```

Then start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:3999.

By default, the frontend proxies API requests to `http://127.0.0.1:9997`. To
use a different backend endpoint:

```bash
XINFERENCE_API_URL=http://127.0.0.1:6735 npm run dev
```

For direct browser requests to a non-default backend, use
`NEXT_PUBLIC_API_URL` instead.

## Build

```bash
npm run build
```
