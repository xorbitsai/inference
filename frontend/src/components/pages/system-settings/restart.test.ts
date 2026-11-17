import assert from 'node:assert/strict';
import { afterEach, it } from 'node:test';

import request from '@/lib/request';
import { restartComponent } from './restart';

const originalPost = request.post;
const originalGet = request.get;
afterEach(() => {
  request.post = originalPost;
  request.get = originalGet;
});

it('waits for Supervisor completion before returning', async () => {
  const calls: string[] = [];
  request.post = (async (url: string) => {
    calls.push(url);
    return { operation_id: 'one', status: 'starting' };
  }) as typeof request.post;
  request.get = (async (url: string) => {
    calls.push(url);
    return { operation_id: 'one', status: 'succeeded' };
  }) as typeof request.get;
  await restartComponent('supervisor', new AbortController().signal);
  assert.deepEqual(calls, ['/v1/supervisor/relaunch', '/v1/supervisor-relaunches/one']);
});

it('recovers an accepted restart when its POST response is lost', async () => {
  let requestId: string;
  request.post = (async (_url: string, body: { request_id: string }) => {
    requestId = body.request_id;
    throw new Error('response lost');
  }) as typeof request.post;
  request.get = (async (url: string) => {
    assert.equal(url, '/v1/api-relaunches');
    return [{ operation_id: 'accepted', request_id: requestId, status: 'draining' }];
  }) as typeof request.get;
  await restartComponent('api', new AbortController().signal);
});

it('reports replacement failure instead of accepting the saved settings as applied', async () => {
  request.post = (async () => ({
    operation_id: 'failed',
    status: 'failed',
    error: 'OTEL SDK unavailable',
  })) as typeof request.post;
  await assert.rejects(
    restartComponent('supervisor', new AbortController().signal),
    /OTEL SDK unavailable/
  );
});

it('sends no restart after the page has unmounted', async () => {
  const controller = new AbortController();
  controller.abort();
  request.post = (() => {
    throw new Error('must not restart');
  }) as typeof request.post;
  await assert.rejects(restartComponent('api', controller.signal), { name: 'AbortError' });
});
