'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import request from '@/lib/request';
import type { LaunchProgressResponse, ModelInstanceInfo, RunningModelItem } from '@/types/services';
import {
  isTokenRouterModel,
  transformRunningModelDetail,
} from '@/components/pages/running-model-detail/utils';

export type ModelListItem = Partial<RunningModelItem> &
  Pick<RunningModelItem, 'id' | 'model_name'> & {
    instance?: ModelInstanceInfo;
    runtimeAvailable: boolean;
  };

const ACTIVE_STATUSES = new Set(['CREATING', 'LOADING', 'UPDATING']);

export function isModelLaunching(model?: ModelListItem): boolean {
  if (model?.instance?.can_cancel_launch) return true;
  if (!model?.instance || ['ERROR', 'TERMINATING', 'TERMINATED'].includes(model.instance.status)) {
    return false;
  }
  return Boolean(
    ACTIVE_STATUSES.has(model.instance.status) ||
    model.instance.replica_statuses?.some((replica) => ACTIVE_STATUSES.has(replica.status))
  );
}

export function isModelReady(model: ModelListItem): boolean {
  return (
    model.runtimeAvailable &&
    !model.instance?.can_cancel_launch &&
    (!model.instance || model.instance.status === 'READY')
  );
}

export function useModelInstances() {
  const [runningModels, setRunningModels] = useState<RunningModelItem[]>([]);
  const [instances, setInstances] = useState<ModelInstanceInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshFailed, setRefreshFailed] = useState(false);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async (initial = false) => {
      if (cancelled) return;

      if (initial) setLoading(true);
      const [modelsResult, instancesResult] = await Promise.allSettled([
        request.get<{ data: RunningModelItem[] }>('/v1/models', { suppressGlobalError: true }),
        request.get<ModelInstanceInfo[]>('/v1/models/instances', { suppressGlobalError: true }),
      ]);
      if (cancelled) return;

      setRefreshFailed(modelsResult.status === 'rejected' || instancesResult.status === 'rejected');

      // Keep each last successful snapshot through transient query failures.
      if (modelsResult.status === 'fulfilled') {
        setRunningModels(Array.isArray(modelsResult.value?.data) ? modelsResult.value.data : []);
      }
      if (instancesResult.status === 'fulfilled') {
        setInstances(Array.isArray(instancesResult.value) ? instancesResult.value : []);
      }
      setLoading(false);
      timer = setTimeout(() => void poll(), 2000);
    };

    void poll(true);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [revision]);

  const models = useMemo(() => {
    const byUid = new Map<string, ModelListItem>();
    for (const model of runningModels) {
      if (
        isTokenRouterModel(model) &&
        ['unavailable', 'disabled'].includes(String(model.router_status || '').toLowerCase())
      ) {
        continue;
      }
      byUid.set(model.id, {
        ...model,
        ...transformRunningModelDetail(model),
        runtimeAvailable: true,
      });
    }
    for (const instance of instances) {
      if (instance.status === 'TERMINATED') continue;
      const running = byUid.get(instance.model_uid);
      if (running && isTokenRouterModel(running)) continue;
      byUid.set(instance.model_uid, {
        id: instance.model_uid,
        model_name: instance.model_name,
        model_ability: instance.model_ability,
        replica: instance.replica,
        n_worker: instance.n_worker ?? undefined,
        runtimeAvailable: false,
        ...running,
        instance,
      });
    }
    return [...byUid.values()];
  }, [instances, runningModels]);

  return { models, loading, refreshFailed, refresh };
}

export function useInstanceLaunchProgress(model?: ModelListItem) {
  const uid = model?.id;
  const created = model?.instance?.instance_created_ts;
  const active = isModelLaunching(model);
  const key = JSON.stringify([uid, created]);
  const [snapshot, setSnapshot] = useState<{ key: string; data: LaunchProgressResponse }>();

  useEffect(() => {
    if (!uid || !active) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      if (cancelled) return;
      try {
        const result = await request.get<number | string | LaunchProgressResponse>(
          `/v1/models/${encodeURIComponent(uid)}/progress`,
          { suppressGlobalError: true }
        );
        if (!cancelled && result && typeof result === 'object') {
          setSnapshot({ key, data: result });
        }
      } catch {
        // Startup can race progress record creation; retry without losing state.
      } finally {
        if (!cancelled) timer = setTimeout(() => void poll(), 1000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [uid, key, active]);

  return snapshot?.key === key ? snapshot.data : null;
}
