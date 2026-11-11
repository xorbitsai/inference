'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { RequestEvents } from '@/constants';
import { eventBus } from '@/lib/event-bus';
import { requestManager } from '@/lib/request-manager';
import request from '@/lib/request';
import type { AddReplicaRequest, LaunchProgressResponse, ReplicaItem } from '@/types/services';

export type AddReplicaPhase = 'pending' | 'succeeded' | 'failed' | 'unknown';
export interface AddReplicaOperation {
  token: number;
  modelUid: string;
  count: number;
  startedAt: number;
  phase: AddReplicaPhase;
  baseline: ReplicaItem[];
  baselineReady: boolean;
  replicas: ReplicaItem[];
  progress: LaunchProgressResponse | null;
  refreshFailed: boolean;
  error?: string;
}

function errorDetail(error: unknown): string | undefined {
  const detail = axios.isAxiosError<{ detail?: unknown }>(error)
    ? error.response?.data?.detail
    : undefined;
  if (typeof detail === 'string') return detail;
  if (detail != null) return JSON.stringify(detail);
  return error instanceof Error ? error.message : undefined;
}

export function isUncertainReplicaError(error: unknown): boolean {
  if (!axios.isAxiosError<{ detail?: unknown }>(error) || !error.response) return true;
  const { status, data } = error.response;
  if ([408, 499, 502, 504].includes(status)) return true;
  return status >= 500 && (!data || typeof data !== 'object' || !('detail' in data));
}

// A reused numeric ID can refer to a different runtime after deletion/recreation.
export function isSameReplica(left: ReplicaItem, right: ReplicaItem): boolean {
  return (
    left.replica_model_uid === right.replica_model_uid &&
    left.replica_id === right.replica_id &&
    left.created_ts === right.created_ts &&
    (!left.model_address || !right.model_address || left.model_address === right.model_address) &&
    (!left.replica_uid || !right.replica_uid || left.replica_uid === right.replica_uid)
  );
}

function preserveForbiddenHandling(error: unknown) {
  // suppressGlobalError keeps 401 refresh handling, but also suppresses 403.
  if (
    axios.isAxiosError(error) &&
    error.response?.status === 403 &&
    requestManager.canHandle403()
  ) {
    eventBus.emit(RequestEvents.FORBIDDEN);
  }
}

export function useAddReplica({
  onReplicas,
  onSettled,
  onSuccessRefreshFailed,
}: {
  onReplicas: (modelUid: string, replicas: ReplicaItem[]) => void;
  onSettled: (operation: AddReplicaOperation) => void;
  onSuccessRefreshFailed?: (operation: AddReplicaOperation) => void;
}) {
  const [operation, setOperation] = useState<AddReplicaOperation | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const current = useRef<AddReplicaOperation | null>(null);
  const mounted = useRef(true);
  const sequence = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const inFlight = useRef<Promise<void> | null>(null);
  const callbacks = useRef({ onReplicas, onSettled, onSuccessRefreshFailed });
  callbacks.current = { onReplicas, onSettled, onSuccessRefreshFailed };

  const publish = useCallback((next: AddReplicaOperation) => {
    if (!mounted.current || current.current?.token !== next.token) return;
    current.current = next;
    setOperation(next);
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
      current.current = null;
    };
  }, []);

  const read = useCallback(
    (token: number): Promise<void> => {
      if (inFlight.current) return inFlight.current;
      const op = current.current;
      if (!op || op.token !== token || !mounted.current) return Promise.resolve();
      const url = `/v1/models/${encodeURIComponent(op.modelUid)}`;
      const config = { suppressGlobalError: true, signal: controller.current?.signal };
      const task = (async () => {
        const [replicas, progress] = await Promise.allSettled([
          request.get<ReplicaItem[]>(`${url}/replicas`, config),
          request.get<LaunchProgressResponse>(`${url}/progress`, config),
        ]);
        const latest = current.current;
        if (!mounted.current || latest?.token !== token) return;
        const next = {
          ...latest,
          refreshFailed: replicas.status === 'rejected' || progress.status === 'rejected',
        };
        if (replicas.status === 'fulfilled' && Array.isArray(replicas.value)) {
          next.replicas = replicas.value;
          callbacks.current.onReplicas(op.modelUid, replicas.value);
        }
        if (progress.status === 'fulfilled') next.progress = progress.value;
        for (const result of [replicas, progress]) {
          if (result.status === 'rejected') preserveForbiddenHandling(result.reason);
        }
        publish(next);
      })().finally(() => {
        if (inFlight.current === task) inFlight.current = null;
      });
      inFlight.current = task;
      return task;
    },
    [publish]
  );

  const tracking = operation?.phase === 'pending' || operation?.phase === 'unknown';
  const token = operation?.token;
  const startedAt = operation?.startedAt;
  const baselineReady = operation?.baselineReady;
  useEffect(() => {
    if (!tracking || !baselineReady || token === undefined) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    const poll = async () => {
      await read(token);
      if (stopped) return;
      failures = current.current?.refreshFailed ? failures + 1 : 0;
      timer = setTimeout(() => void poll(), Math.min(2000 * 2 ** Math.min(failures, 4), 30000));
    };
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [tracking, baselineReady, token, read]);

  useEffect(() => {
    if (startedAt === undefined || !tracking) return;
    const update = () =>
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    update();
    const timer = setInterval(update, 1000);
    return () => clearInterval(timer);
  }, [tracking, token, startedAt]);

  const submit = useCallback(
    async (modelUid: string, body: AddReplicaRequest) => {
      if (!mounted.current || ['pending', 'unknown'].includes(current.current?.phase ?? '')) return;
      controller.current?.abort();
      controller.current = new AbortController();
      const op: AddReplicaOperation = {
        token: ++sequence.current,
        modelUid,
        count: body.replica,
        startedAt: Date.now(),
        phase: 'pending',
        baseline: [],
        baselineReady: false,
        replicas: [],
        progress: null,
        refreshFailed: false,
      };
      current.current = op; // Lock synchronously, before React renders again.
      setOperation(op);
      setElapsedSeconds(0);
      const url = `/v1/models/${encodeURIComponent(modelUid)}/replicas`;
      let posted = false;
      try {
        // Establish the baseline before the POST and before starting status polling.
        const baseline = await request.get<ReplicaItem[]>(url, {
          suppressGlobalError: true,
          signal: controller.current.signal,
        });
        if (!Array.isArray(baseline)) throw new Error('Invalid replica snapshot');
        if (!mounted.current || current.current?.token !== op.token) return;
        publish({ ...current.current, baseline, baselineReady: true, replicas: baseline });
        posted = true;
        await request.post(url, body, {
          noTimeout: true,
          suppressGlobalError: true,
          signal: controller.current.signal,
        });
        if (!mounted.current || current.current?.token !== op.token) return;
        publish({ ...current.current, phase: 'succeeded' });
      } catch (error) {
        if (!mounted.current || current.current?.token !== op.token) return;
        preserveForbiddenHandling(error);
        publish({
          ...current.current,
          phase: posted && isUncertainReplicaError(error) ? 'unknown' : 'failed',
          error: errorDetail(error),
        });
      }
      if (!mounted.current || current.current?.token !== op.token) return;
      callbacks.current.onSettled(current.current);
      // Wait for a slow poll before the final refresh; never overlap polling rounds.
      await inFlight.current;
      if (!mounted.current || current.current?.token !== op.token) return;
      await read(op.token);
      const latest = current.current;
      if (!mounted.current || latest?.token !== op.token) return;
      if (latest.phase === 'succeeded' && latest.refreshFailed) {
        // Notify once here, after the final read, rather than on every poll or render.
        callbacks.current.onSuccessRefreshFailed?.(latest);
      }
    },
    [publish, read]
  );

  const reset = useCallback(() => {
    if (current.current?.phase === 'pending') return;
    controller.current?.abort();
    current.current = null;
    setOperation(null);
  }, []);

  return {
    operation,
    elapsedSeconds,
    submit,
    reset,
    refresh: () => operation && read(operation.token),
    newReplicas: operation?.baselineReady
      ? operation.replicas.filter(
          (replica) => !operation.baseline.some((old) => isSameReplica(old, replica))
        )
      : [],
  };
}
