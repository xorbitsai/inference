import { isAxiosError } from 'axios';

import { createRelaunchRequestId } from '@/lib/relaunch-request-id';
import request from '@/lib/request';

type Component = 'api' | 'supervisor';
interface RestartOperation {
  operation_id: string;
  request_id?: string;
  status: string;
  error?: string;
}

function transient(error: unknown) {
  const status = isAxiosError(error) ? error.response?.status : undefined;
  return status === undefined || status === 408 || status === 429 || status >= 500;
}

function delay(signal: AbortSignal): Promise<void> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, 1000);
    signal.addEventListener('abort', abort, { once: true });
  });
}

export async function restartComponent(component: Component, signal: AbortSignal) {
  signal.throwIfAborted();
  const requestId = createRelaunchRequestId(component);
  const options = { signal, suppressGlobalError: true };
  let operation: RestartOperation;
  try {
    operation = await request.post<RestartOperation>(
      `/v1/${component}/relaunch`,
      { request_id: requestId },
      options
    );
  } catch (error) {
    if (signal.aborted || !transient(error)) throw error;
    // A lost response may still have started the replacement. Recover its ID.
    const history = await request.get<RestartOperation[]>(`/v1/${component}-relaunches`, options);
    const accepted = history.find((job) => job.request_id === requestId);
    if (!accepted) throw error;
    operation = accepted;
  }
  const deadline = Date.now() + 300_000;
  while (!signal.aborted) {
    if (operation.status === 'failed') throw new Error(operation.error || 'Restart failed');
    if (
      operation.status === 'succeeded' ||
      (component === 'api' && operation.status === 'draining')
    )
      return;
    if (Date.now() >= deadline)
      throw new Error('Restart is still in progress; check Cluster Information');
    try {
      operation = await request.get<RestartOperation>(
        `/v1/${component}-relaunches/${encodeURIComponent(operation.operation_id)}`,
        options
      );
    } catch (error) {
      if (signal.aborted || !transient(error)) throw error;
    }
    if (operation.status !== 'succeeded') await delay(signal);
  }
  throw signal.reason;
}
