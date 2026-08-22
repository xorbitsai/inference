import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveRouterCapabilities } from './router-capabilities.mjs';

test('keeps router controls hidden until global configuration is ready', () => {
  assert.deepEqual(
    resolveRouterCapabilities({
      globalReady: false,
      authAdvanced: false,
      canWriteRouters: true,
      canOperateRouters: true,
    }),
    { canWriteRouters: false, canOperateRouters: false }
  );
});

test('allows all router controls when advanced auth is disabled', () => {
  assert.deepEqual(
    resolveRouterCapabilities({
      globalReady: true,
      authAdvanced: false,
      canWriteRouters: false,
      canOperateRouters: false,
    }),
    { canWriteRouters: true, canOperateRouters: true }
  );
});

test('preserves distinct write and operate scopes with advanced auth', () => {
  assert.deepEqual(
    resolveRouterCapabilities({
      globalReady: true,
      authAdvanced: true,
      canWriteRouters: true,
      canOperateRouters: false,
    }),
    { canWriteRouters: true, canOperateRouters: false }
  );
  assert.deepEqual(
    resolveRouterCapabilities({
      globalReady: true,
      authAdvanced: true,
      canWriteRouters: false,
      canOperateRouters: true,
    }),
    { canWriteRouters: false, canOperateRouters: true }
  );
});
