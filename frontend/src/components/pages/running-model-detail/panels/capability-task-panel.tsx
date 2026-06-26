'use client';

import { useMemo, useRef, useState, useImperativeHandle, forwardRef, useEffect } from 'react';
import { Copy, RotateCcw, Sparkles } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { ModelAbility, ModelType } from '@/constants';
import { createForm } from '@/hooks/use-form';
import request from '@/lib/request';
import { cn, copyText, sleep } from '@/lib/utils';
import { isNumber } from '@/lib/is';
import type {
  RunningModelDetail,
  CompletionResponse,
  RerankResponse,
  EmbeddingsResponse,
} from '@/types/services';
import type { FormValues } from '@/types/form';

import type { CapabilityConfig } from '../types';
import { createId } from '../utils';

interface CapabilityTaskPanelProps {
  config: CapabilityConfig;
  model: RunningModelDetail;
  modelUid: string;
}

export interface CapabilityTaskPanelMethod {
  reset: () => void;
}
type ProgressResponse = { progress?: number };

function normalizeProgress(response: ProgressResponse) {
  if (!isNumber(response?.progress)) {
    return undefined;
  }
  return Math.max(0, Math.min(100, response.progress * 100));
}

const CapabilityTaskPanel = forwardRef<CapabilityTaskPanelMethod, CapabilityTaskPanelProps>(
  ({ config, model, modelUid }, ref) => {
    const form = useMemo(() => createForm(), []);
    const runTokenRef = useRef(0);
    const ResultPanel = config.resultPanel;
    const FormPanel = config.formPanel;
    const Icon = config.icon;
    const [result, setResult] = useState<unknown>();
    const [resultValues, setResultValues] = useState<FormValues | undefined>();
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState<number | undefined>();

    const showCopyResult = useMemo(() => {
      return (
        config.ability === ModelAbility.Generate ||
        model.model_type === ModelType.Rerank ||
        model.model_type === ModelType.Embedding
      );
    }, [config.ability, model.model_type]);
    const copyResultValue = useMemo(() => {
      if (result === undefined) {
        return '';
      }
      if (config.ability === ModelAbility.Generate) {
        const text = (result as CompletionResponse)?.choices?.[0]?.text;
        return typeof text === 'string' ? text : '';
      }
      if (model.model_type === ModelType.Rerank) {
        const results = (result as RerankResponse)?.results;
        try {
          return JSON.stringify(results, null, 2) || '';
        } catch {
          return String(results);
        }
      }

      if (model.model_type === ModelType.Embedding) {
        const data = (result as EmbeddingsResponse)?.data;
        try {
          return JSON.stringify(data, null, 2) || '';
        } catch {
          return String(data);
        }
      }

      return '';
    }, [result, config.ability, model.model_type]);

    const trackProgress = async (
      requestId: string,
      runToken: number,
      isFinished: () => boolean
    ) => {
      await sleep(1000);

      while (runTokenRef.current === runToken && !isFinished()) {
        try {
          const response = await request.get<ProgressResponse>(
            `/v1/requests/${requestId}/progress`
          );
          const nextProgress = normalizeProgress(response);

          if (nextProgress !== undefined && runTokenRef.current === runToken) {
            setProgress(nextProgress);
          }
        } catch {
          if (runTokenRef.current === runToken) {
            setProgress(undefined);
          }

          return;
        }

        await sleep(1000);
      }
    };

    const submit = () => {
      const runToken = runTokenRef.current + 1;
      const requestId = config.showProgress ? createId('request') : undefined;
      let finished = false;

      runTokenRef.current = runToken;
      setLoading(true);
      setProgress(config.showProgress ? 0 : undefined);
      setResult(undefined);
      setResultValues(undefined);

      const values = form.getFieldsValue();
      const body = config.transformValues({ modelUid, model, values, requestId });
      const requestPromise = request.post(config.requestApi, body, {
        ...(config.responseType === 'blob' ? { responseType: 'blob' as const } : {}),
        noTimeout: true,
      });
      requestPromise
        .then((response) => {
          if (runTokenRef.current !== runToken) return;

          setResult(response);
          setResultValues(values);
        })
        .finally(() => {
          finished = true;

          if (runTokenRef.current === runToken) {
            setLoading(false);
            setProgress(undefined);
          }
        });

      if (config.showProgress && requestId) {
        void trackProgress(requestId, runToken, () => finished);
      }
    };
    const reset = () => {
      runTokenRef.current += 1;
      form.resetFields();
      setResult(undefined);
      setResultValues(undefined);
      setLoading(false);
      setProgress(undefined);
    };

    useEffect(() => {
      return () => {
        runTokenRef.current += 1;
      };
    }, []);
    useImperativeHandle(ref, () => ({
      reset,
    }));
    return (
      <div className="grid min-h-[calc(100vh-216px)] grid-cols-1 gap-5 xl:grid-cols-[400px_minmax(0,1fr)]">
        <section className="rounded-xl border bg-card shadow-sm flex flex-col">
          <div className="border-b px-4 py-3 shrink-0">
            <div className="flex items-center gap-3">
              <span className="flex size-8 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                <Icon className="size-4" />
              </span>
              <h2 className="min-w-0 truncate text-lg font-semibold">{config.label}</h2>
            </div>
          </div>

          <Form
            form={form}
            initialValues={config.initialValues}
            onFinish={submit}
            className="flex-1 min-h-0 flex flex-col"
          >
            <div className="min-h-0 flex-1 space-y-5 p-4">
              <FormPanel form={form} model={model} modelUid={modelUid} />
            </div>
            <div className="flex items-center gap-3 border-t p-4">
              <Button type="submit" className="h-11 flex-1 rounded-full" loading={loading}>
                <Sparkles className={cn('size-4', loading && 'hidden')} />
                Generate
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="icon"
                className="size-11 rounded-full"
                disabled={loading}
                onClick={reset}
              >
                <RotateCcw className="size-4" />
              </Button>
            </div>
          </Form>
        </section>

        <section className="relative overflow-hidden rounded-xl border bg-background shadow-sm">
          <div className="flex items-center justify-between border-b bg-card/80 p-4">
            <h3 className="text-base font-semibold">Results</h3>
            {showCopyResult && copyResultValue && !loading && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8 rounded-full text-muted-foreground"
                onClick={() => copyText(copyResultValue)}
              >
                <Copy className="size-4" />
              </Button>
            )}
          </div>
          <div className="p-4">
            <ResultPanel
              result={result}
              values={resultValues}
              loading={loading}
              progress={progress}
              ability={config.ability}
            />
          </div>
        </section>
      </div>
    );
  }
);

CapabilityTaskPanel.displayName = 'CapabilityTaskPanel';

export default CapabilityTaskPanel;
