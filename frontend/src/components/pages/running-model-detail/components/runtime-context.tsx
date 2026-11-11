'use client';

import { useEffect, useState } from 'react';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import type { RunningModelDetail } from '@/types/services';

export interface ModelRuntime {
  runtime_context_length: number | null;
  replicas: { replica_uid: string; runtime_context_length: number | null }[];
}

export function useModelRuntime(modelUid: string, model: RunningModelDetail | null) {
  const [result, setResult] = useState<{
    modelUid: string;
    model: RunningModelDetail;
    data: ModelRuntime;
  } | null>(null);

  useEffect(() => {
    if (!model) return;
    let active = true;
    request
      .get<ModelRuntime>(`/v1/models/${encodeURIComponent(modelUid)}/runtime`, {
        suppressGlobalError: true,
      })
      .then((data) => {
        if (active) setResult({ modelUid, model, data });
      })
      .catch(() => {
        if (active) setResult(null);
      });
    return () => {
      active = false;
    };
  }, [modelUid, model]);

  return result?.modelUid === modelUid && result.model === model ? result.data : null;
}

export function RuntimeContext({
  runtime,
  registeredContextLength,
}: {
  runtime: ModelRuntime | null;
  registeredContextLength?: number;
}) {
  const { t } = useI18n();
  const unknown = t('runningModels.detail.unknownContextLength');
  return (
    <div className="min-w-0 rounded-2xl bg-muted/40 px-4 py-3">
      <div className="text-xs font-medium tracking-wide text-muted-foreground">
        {t('runningModels.detail.runtimeContextLength')}
      </div>
      <div className="mt-1 text-sm font-medium text-foreground">
        {runtime?.runtime_context_length ?? unknown}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {t('runningModels.detail.runtimeContextDescription')}
      </p>
      {registeredContextLength != null && (
        <p className="mt-1 text-xs text-muted-foreground">
          {t('runningModels.detail.registeredContextLength')}: {registeredContextLength}
        </p>
      )}
      {runtime && runtime.replicas.length > 1 && (
        <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
          {runtime.replicas.map((replica) => (
            <li key={replica.replica_uid} className="break-all">
              {replica.replica_uid}: {replica.runtime_context_length ?? unknown}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
