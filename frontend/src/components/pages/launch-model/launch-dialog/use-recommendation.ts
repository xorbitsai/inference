'use client';

import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type { FormInstance } from '@/types/form';
import type { ModelEngine } from '@/types/services';
import request from '@/lib/request';
import {
  recommendationPatch,
  recommendationRequest,
  type RecommendationResponse,
} from './recommendation';

export function useRecommendation(
  form: FormInstance,
  modelName: string | undefined,
  active: boolean,
  markEdited: () => void,
  applyEngines: (engines: ModelEngine, enableVirtualEnv: boolean) => void
) {
  const generation = useRef(0);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [failed, setFailed] = useState(false);
  const invalidate = useCallback(() => {
    generation.current += 1;
    setPending(false);
    setResult(null);
    setFailed(false);
  }, []);

  useLayoutEffect(() => {
    invalidate();
    const unsubscribe = form.subscribe(() => {
      generation.current += 1;
      setPending(false);
      // Linked fields may initialize after applying the recommendation. Keep
      // feedback about the last completed action while invalidating requests.
    });
    return () => {
      generation.current += 1;
      unsubscribe();
    };
  }, [form, modelName, active, invalidate]);

  const recommend = async () => {
    if (!active || !modelName) return;
    const values = structuredClone(form.getFieldsValue());
    const id = ++generation.current;
    setPending(true);
    setResult(null);
    setFailed(false);
    try {
      const payload = recommendationRequest(modelName, values);
      const response = await request.post<RecommendationResponse>('/v1/models/recommend', payload);
      if (id !== generation.current) return;
      if (response.status === 'recommended') {
        if (!response.config || typeof response.config.enable_virtual_env !== 'boolean')
          throw new Error('missing effective environment');
        const engines = await request.get<ModelEngine>(
          `/v1/engines/${encodeURIComponent(modelName)}`,
          {
            params: { enable_virtual_env: response.config.enable_virtual_env },
          }
        );
        if (id !== generation.current) return;
        const patch = recommendationPatch(response, values, engines);
        if (!patch) throw new Error('missing recommendation');
        markEdited();
        applyEngines(engines, response.config.enable_virtual_env);
        form.setFieldsValue(patch);
      }
      setResult(response);
    } catch {
      if (id === generation.current) setFailed(true);
    } finally {
      if (id === generation.current) setPending(false);
    }
  };

  return { recommend, pending, result, failed, invalidate };
}
