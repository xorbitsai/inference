'use client';

import {
  useMemo,
  useRef,
  useState,
  useImperativeHandle,
  forwardRef,
  useEffect,
  useCallback,
} from 'react';
import { Copy, RotateCcw, Sparkles } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { ModelAbility, ModelType, RequestEvents } from '@/constants';
import { createForm } from '@/hooks/use-form';
import request from '@/lib/request';
import { EventStreamController, postEventStreamFetcher } from '@/lib/eventStream';
import { eventBus } from '@/lib/event-bus';
import { cn, copyToClipboard, sleep } from '@/lib/utils';
import { isNumber } from '@/lib/is';
import type {
  RunningModelDetail,
  CompletionResponse,
  RerankResponse,
  EmbeddingsResponse,
  AudioEmbeddingResponse,
} from '@/types/services';
import type { FormValues } from '@/types/form';

import { AudioStreamSession } from '../audio-stream';
import type { CapabilityConfig } from '../types';
import { booleanValue, createId, stringValue } from '../utils';

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

function formatLatency(latencyMs: number) {
  return latencyMs < 1000 ? String(latencyMs) + ' ms' : (latencyMs / 1000).toFixed(2) + ' s';
}

function audioFileName(modelName: string, blob: Blob) {
  const mimeSubtype = blob.type.split(';', 1)[0].split('/')[1];
  const extension = mimeSubtype === 'mpeg' ? 'mp3' : mimeSubtype || 'mp3';
  const timestamp = new Date().toLocaleString('sv-SE').replace(/\D/g, '');
  const safeModelName = modelName.replace(/[<>:"/\\|?*]/g, '_');
  return `${safeModelName}_${timestamp}.${extension}`;
}

const CapabilityTaskPanel = forwardRef<CapabilityTaskPanelMethod, CapabilityTaskPanelProps>(
  ({ config, model, modelUid }, ref) => {
    const form = useMemo(() => createForm(), []);
    const runTokenRef = useRef(0);
    const activeRequestRef = useRef<
      { modelUid: string; requestId: string; runToken: number } | undefined
    >(undefined);
    const audioStreamRef = useRef<AudioStreamSession | undefined>(undefined);
    const completionStreamRef = useRef<EventStreamController | undefined>(undefined);
    const ResultPanel = config.resultPanel;
    const FormPanel = config.formPanel;
    const Icon = config.icon;
    const [result, setResult] = useState<unknown>();
    const [resultValues, setResultValues] = useState<FormValues | undefined>();
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState<number | undefined>();
    const [latencyMs, setLatencyMs] = useState<number | undefined>();
    const showLiveProgress = Boolean(
      config.showProgress &&
      (model.model_type !== ModelType.World ||
        model.model_family === 'Astra' ||
        model.model_name === 'Astra' ||
        model.model_family === 'HY-WorldPlay' ||
        model.model_name === 'HY-WorldPlay-5B')
    );

    const showCopyResult = useMemo(() => {
      return (
        config.ability === ModelAbility.Generate ||
        config.ability === ModelAbility.SpeakerEmbedding ||
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
      if (config.ability === ModelAbility.SpeakerEmbedding) {
        const embedding = (result as AudioEmbeddingResponse)?.embedding;
        try {
          return JSON.stringify(embedding, null, 2) || '';
        } catch {
          return String(embedding);
        }
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

    const abortActiveRequest = useCallback(() => {
      const activeRequest = activeRequestRef.current;
      activeRequestRef.current = undefined;
      if (!activeRequest) return;

      void request
        .post(
          `/v1/models/${encodeURIComponent(activeRequest.modelUid)}/requests/${encodeURIComponent(activeRequest.requestId)}/abort`,
          { block_duration: 30 }
        )
        .catch(() => undefined);
    }, []);

    const disposeCompletionStream = useCallback(() => {
      completionStreamRef.current?.terminate();
      completionStreamRef.current = undefined;
    }, []);

    const disposeAudioStream = useCallback(() => {
      audioStreamRef.current?.dispose();
      audioStreamRef.current = undefined;
    }, []);

    const submit = () => {
      disposeAudioStream();
      disposeCompletionStream();
      const runToken = runTokenRef.current + 1;
      const requestId = config.showProgress ? createId('request') : undefined;
      let finished = false;

      runTokenRef.current = runToken;
      if (requestId) {
        activeRequestRef.current = { modelUid, requestId, runToken };
      }
      const startedAt = performance.now();
      setLoading(true);
      setLatencyMs(undefined);
      setProgress(showLiveProgress ? 0 : undefined);

      const values = form.getFieldsValue();
      const streamAudio = config.responseType === 'audio-stream' && booleanValue(values.stream);
      const audioStreamSession = streamAudio
        ? new AudioStreamSession(stringValue(values.response_format, 'mp3'))
        : undefined;

      if (audioStreamSession) {
        audioStreamRef.current = audioStreamSession;
        setResult(audioStreamSession.result());
        setResultValues(values);
      } else {
        setResult(undefined);
        setResultValues(undefined);
      }

      let requestStarted = false;
      let streamResponseStarted = false;
      const requestPromise = Promise.resolve(
        config.transformValues({ modelUid, model, values, requestId })
      ).then(async (body) => {
        requestStarted = true;
        if (config.stream) {
          streamResponseStarted = true;
          const controller = new EventStreamController();
          completionStreamRef.current = controller;
          return new Promise<CompletionResponse>((resolve, reject) => {
            let aggregate: CompletionResponse = {
              id: createId('completion'),
              model: modelUid,
              choices: [{ index: 0, text: '', finish_reason: null }],
            };
            void postEventStreamFetcher<CompletionResponse>(
              {
                url: config.requestApi,
                data: body,
                options: {
                  onData: (chunk) => {
                    const nextChoice = chunk?.choices?.[0];
                    const nextText = stringValue(nextChoice?.text);
                    const currentText = stringValue(aggregate.choices[0]?.text);
                    aggregate = {
                      ...aggregate,
                      ...chunk,
                      choices: [
                        {
                          ...aggregate.choices[0],
                          ...nextChoice,
                          text: currentText + nextText,
                        },
                      ],
                    };
                    if (runTokenRef.current === runToken) {
                      setResult(aggregate);
                      setResultValues(values);
                    }
                  },
                  onError: (message) => reject(new Error(message)),
                  onEnd: () => {
                    if (completionStreamRef.current === controller) {
                      completionStreamRef.current = undefined;
                    }
                    resolve(aggregate);
                  },
                },
              },
              controller
            );
          });
        }
        if (audioStreamSession) {
          const stream = await request.post<ReadableStream<Uint8Array<ArrayBuffer>>>(
            config.requestApi,
            body,
            {
              adapter: 'fetch',
              responseType: 'stream',
              noTimeout: true,
              signal: audioStreamSession.signal,
            }
          );

          if (!stream || typeof stream.getReader !== 'function') {
            throw new Error('The browser did not return a readable audio stream.');
          }

          streamResponseStarted = true;
          return audioStreamSession.consume(stream, () => {
            if (runTokenRef.current === runToken) {
              setResult(audioStreamSession.result());
            }
          });
        }

        return request.post(config.requestApi, body, {
          ...(config.responseType ? { responseType: 'blob' as const } : {}),
          noTimeout: true,
        });
      });
      requestPromise
        .then((response) => {
          if (runTokenRef.current !== runToken) return;

          if (audioStreamSession) {
            const blob = response as Blob;
            const file = new File([blob], audioFileName(model.model_name, blob), {
              type: blob.type,
            });
            setResult(audioStreamSession.result(file));
            setResultValues(values);
            return;
          }

          setResult(
            response instanceof Blob
              ? new File([response], audioFileName(model.model_name, response), {
                  type: response.type,
                })
              : response
          );
          setResultValues(values);
        })
        .catch((error: unknown) => {
          if (audioStreamSession && runTokenRef.current === runToken) {
            if (audioStreamRef.current === audioStreamSession) {
              audioStreamRef.current = undefined;
            }
            audioStreamSession.dispose();
            setResult(undefined);
            setResultValues(undefined);
          }

          // HTTP errors are reported by the shared interceptor. Transform failures and
          // stream read failures happen outside that boundary.
          if (
            (!requestStarted || streamResponseStarted) &&
            !audioStreamSession?.signal.aborted &&
            runTokenRef.current === runToken
          ) {
            eventBus.emit(
              RequestEvents.SERVER_ERROR,
              error instanceof Error ? error.message : String(error)
            );
          }
        })
        .finally(() => {
          finished = true;

          if (activeRequestRef.current?.runToken === runToken) {
            activeRequestRef.current = undefined;
          }

          if (runTokenRef.current === runToken) {
            setLoading(false);
            setLatencyMs(Math.round(performance.now() - startedAt));
            setProgress(undefined);
          }
        });

      if (showLiveProgress && requestId) {
        void trackProgress(requestId, runToken, () => finished);
      }
    };
    const reset = () => {
      abortActiveRequest();
      disposeAudioStream();
      disposeCompletionStream();
      runTokenRef.current += 1;
      form.resetFields();
      setResult(undefined);
      setResultValues(undefined);
      setLoading(false);
      setLatencyMs(undefined);
      setProgress(undefined);
    };

    useEffect(() => {
      return () => {
        abortActiveRequest();
        disposeAudioStream();
        disposeCompletionStream();
        runTokenRef.current += 1;
      };
    }, [abortActiveRequest, config.ability, disposeAudioStream, disposeCompletionStream, modelUid]);
    useImperativeHandle(ref, () => ({
      reset,
    }));
    const actionBar = (
      <div className="flex items-center gap-3">
        <Button type="submit" className="h-11 flex-1 rounded-full" loading={loading}>
          <Sparkles className={cn('size-4', loading && 'hidden')} />
          {config.submitLabel || 'Generate'}
        </Button>
        <Button
          type="button"
          variant="secondary"
          size="icon"
          className="size-11 rounded-full"
          disabled={loading && !config.showProgress && audioStreamRef.current === undefined}
          onClick={reset}
        >
          <RotateCcw className="size-4" />
        </Button>
      </div>
    );

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
              <FormPanel
                form={form}
                model={model}
                modelUid={modelUid}
                actions={config.ability === ModelAbility.Generate ? actionBar : undefined}
              />
              {config.ability !== ModelAbility.Generate && actionBar}
            </div>
          </Form>
        </section>

        <section className="relative min-w-0 overflow-hidden rounded-xl border bg-background shadow-sm">
          <div className="flex items-center justify-between border-b bg-card/80 p-4">
            <div className="flex items-center gap-3">
              <h3 className="text-base font-semibold">Results</h3>
              {latencyMs !== undefined && !loading && (
                <span className="text-xs font-medium text-muted-foreground">
                  Latency {formatLatency(latencyMs)}
                </span>
              )}
            </div>
            {showCopyResult && copyResultValue && !loading && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8 rounded-full text-muted-foreground"
                aria-label="Copy result"
                onClick={() => copyToClipboard(copyResultValue)}
              >
                <Copy className="size-4" />
              </Button>
            )}
          </div>
          <div className="min-w-0 p-4">
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
