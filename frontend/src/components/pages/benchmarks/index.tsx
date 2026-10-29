'use client';

import { useCallback, useEffect, useState } from 'react';
import { Activity, Download, RefreshCw, Square, Trash2 } from 'lucide-react';
import PageContainer from '@/components/ui/page-container';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';
import BenchmarkCharts from './charts';

interface Distribution {
  mean: number;
  p50: number;
  p95: number;
  p99: number;
}
interface Stage {
  concurrency: number;
  successful: number;
  failed: number;
  request_throughput: number;
  output_throughput: number | null;
  input_throughput: number | null;
  latency_ms: Distribution | null;
  ttft_ms: Distribution | null;
  tpot_ms: Distribution | null;
  itl_ms: Distribution | null;
  prompt_tokens: Distribution | null;
  output_tokens: Distribution | null;
}
interface Job {
  id: string;
  name: string;
  created_at: string;
  state: string;
  progress: number;
  config: {
    model_uid: string;
    input_tokens: number;
    output_tokens: number;
    requests: number;
    warmup: number;
    seed: number;
  };
  snapshot: Record<string, unknown>;
  runner: { name: string; version: string; metrics_version: string };
  results: Stage[];
  logs: string[];
  error: string | null;
}
interface Model {
  id: string;
  model_type: string;
  model_ability?: string[];
}
const active = (job: Job) => ['queued', 'running'].includes(job.state);
const format = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : value.toFixed(2);
const cell = 'px-3 py-2 text-left whitespace-nowrap';

export default function Benchmarks() {
  const { t } = useI18n();
  const { canWriteModels, hasModelsRead } = useMenuAuth();
  const allowed = canWriteModels && hasModelsRead;
  const [jobs, setJobs] = useState<Job[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [detailId, setDetailId] = useState<string>();
  const [deleteId, setDeleteId] = useState<string>();
  const [modelUid, setModelUid] = useState('');
  const [name, setName] = useState('');
  const [inputTokens, setInputTokens] = useState('256');
  const [outputTokens, setOutputTokens] = useState('128');
  const [concurrency, setConcurrency] = useState('1, 4, 16');
  const [requests, setRequests] = useState('100');
  const [warmup, setWarmup] = useState('3');
  const [seed, setSeed] = useState('42');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [runnerLog, setRunnerLog] = useState('');

  const refresh = useCallback(async () => {
    const data = await request.get<Job[]>('/v1/benchmarks');
    setJobs(data);
  }, []);

  useEffect(() => {
    if (!allowed) return;
    let mounted = true;
    const poll = async () => {
      try {
        const data = await request.get<Job[]>('/v1/benchmarks');
        if (mounted) setJobs(data);
      } catch {
        // The shared request handler displays server errors.
      }
    };
    void poll();
    const timer = setInterval(poll, 3000);
    void request
      .get<{ data: Model[] }>('/v1/models')
      .then((data) => {
        if (!mounted) return;
        const candidates = data.data.filter(
          (m) => m.model_type === 'LLM' && m.model_ability?.includes('chat')
        );
        setModels(candidates);
        const initial = new URLSearchParams(window.location.search).get('model');
        setModelUid(candidates.find((m) => m.id === initial)?.id ?? candidates[0]?.id ?? '');
      })
      .catch(() => {});
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [allowed]);

  useEffect(() => {
    setRunnerLog('');
  }, [detailId]);

  if (!allowed) return <p>{t('common.accessDenied')}</p>;

  const detail = jobs.find((job) => job.id === detailId);
  const compared = jobs.filter((job) => selected.includes(job.id));
  const create = async (event: React.FormEvent) => {
    event.preventDefault();
    const points = concurrency.split(',').map((value) => Number(value.trim()));
    if (
      points.some((value) => !Number.isInteger(value) || value < 1 || value > 256) ||
      points.length > 10 ||
      new Set(points).size !== points.length
    ) {
      setError(t('benchmarks.invalidConcurrency'));
      return;
    }
    setBusy(true);
    setError('');
    try {
      const job = await request.post<Job>('/v1/benchmarks', {
        model_uid: modelUid,
        name,
        runner: 'xinference',
        input_tokens: Number(inputTokens),
        output_tokens: Number(outputTokens),
        concurrency: points,
        requests: Number(requests),
        warmup: Number(warmup),
        seed: Number(seed),
      });
      setDetailId(job.id);
      await refresh();
    } finally {
      setBusy(false);
    }
  };
  const mutate = async (jobId: string, operation: 'stop' | 'delete') => {
    setBusy(true);
    try {
      if (operation === 'stop') await request.post(`/v1/benchmarks/${jobId}/stop`);
      else {
        await request.delete(`/v1/benchmarks/${jobId}`);
        setSelected((ids) => ids.filter((id) => id !== jobId));
        if (detailId === jobId) setDetailId(undefined);
        setDeleteId(undefined);
      }
      await refresh();
    } finally {
      setBusy(false);
    }
  };
  const download = async (jobId: string) => {
    const data = await request.get<Job>(`/v1/benchmarks/${jobId}/export`);
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = `benchmark-${jobId}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };
  const results = (items: Job[]) => (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50">
          <tr>
            {[
              t('benchmarks.name'),
              t('benchmarks.model'),
              t('benchmarks.concurrency'),
              t('benchmarks.successFailed'),
              'req/s',
              t('benchmarks.inputThroughput'),
              t('benchmarks.outputThroughput'),
              t('benchmarks.actualInput'),
              t('benchmarks.actualOutput'),
              'TTFT mean / P95 / P99 (ms)',
              'TPOT mean / P95 / P99 (ms)',
              'ITL mean / P95 / P99 (ms)',
              t('benchmarks.latency'),
            ].map((label) => (
              <th key={label} className={cell}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.flatMap((job) =>
            job.results.map((stage) => (
              <tr key={`${job.id}-${stage.concurrency}`} className="border-t">
                <td className={cell}>{job.name}</td>
                <td className={cell}>{job.config.model_uid}</td>
                <td className={cell}>{stage.concurrency}</td>
                <td className={cell}>
                  {stage.successful} / {stage.failed}
                </td>
                <td className={cell}>{format(stage.request_throughput)}</td>
                <td className={cell}>{format(stage.input_throughput)}</td>
                <td className={cell}>{format(stage.output_throughput)}</td>
                <td className={cell}>{format(stage.prompt_tokens?.mean)}</td>
                <td className={cell}>{format(stage.output_tokens?.mean)}</td>
                {[stage.ttft_ms, stage.tpot_ms, stage.itl_ms].map((metric, index) => (
                  <td className={cell} key={index}>
                    {[metric?.mean, metric?.p95, metric?.p99].map(format).join(' / ')}
                  </td>
                ))}
                <td className={cell}>{format(stage.latency_ms?.p95)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );

  return (
    <PageContainer
      title={t('benchmarks.title')}
      subTitle={t('benchmarks.description')}
      extraContent={
        <Button variant="outline" onClick={() => void refresh()}>
          <RefreshCw />
          {t('common.refresh')}
        </Button>
      }
    >
      <div className="space-y-6">
        <form
          onSubmit={(event) => void create(event)}
          className="rounded-xl border bg-card p-5 space-y-4"
        >
          <h2 className="font-semibold flex items-center gap-2">
            <Activity className="size-4" />
            {t('benchmarks.create')}
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.model')}</span>
              <select
                className="h-9 w-full rounded-md border bg-background px-3"
                value={modelUid}
                onChange={(event) => setModelUid(event.target.value)}
                required
              >
                <option value="" disabled>
                  {t('benchmarks.selectModel')}
                </option>
                {models.map((m) => (
                  <option value={m.id} key={m.id}>
                    {m.id}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.name')}</span>
              <Input
                value={name}
                maxLength={128}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.inputTokens')}</span>
              <Input
                type="number"
                min={16}
                max={32768}
                required
                value={inputTokens}
                onChange={(event) => setInputTokens(event.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.outputTokens')}</span>
              <Input
                type="number"
                min={1}
                max={8192}
                required
                value={outputTokens}
                onChange={(event) => setOutputTokens(event.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.concurrency')}</span>
              <Input
                required
                value={concurrency}
                onChange={(event) => setConcurrency(event.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.requests')}</span>
              <Input
                type="number"
                min={1}
                max={5000}
                required
                value={requests}
                onChange={(event) => setRequests(event.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.warmup')}</span>
              <Input
                type="number"
                min={0}
                max={10}
                required
                value={warmup}
                onChange={(event) => setWarmup(event.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <span>{t('benchmarks.seed')}</span>
              <Input
                type="number"
                min={0}
                max={2147483647}
                required
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
              />
            </label>
          </div>
          <p className="text-xs text-muted-foreground">{t('benchmarks.workloadHint')}</p>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <Button type="submit" loading={busy} disabled={!modelUid}>
            {t('benchmarks.start')}
          </Button>
        </form>

        <section className="space-y-3">
          <h2 className="font-semibold">{t('benchmarks.history')}</h2>
          {jobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t('benchmarks.empty')}</p>
          ) : (
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    {[
                      t('benchmarks.compare'),
                      t('benchmarks.name'),
                      t('benchmarks.model'),
                      t('benchmarks.created'),
                      t('benchmarks.status'),
                      t('benchmarks.actions'),
                    ].map((label) => (
                      <th key={label} className={cell}>
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => (
                    <tr key={job.id} className="border-t">
                      <td className={cell}>
                        <input
                          type="checkbox"
                          aria-label={`${t('benchmarks.compare')} ${job.name}`}
                          checked={selected.includes(job.id)}
                          onChange={(event) =>
                            setSelected((ids) =>
                              event.target.checked
                                ? [...ids, job.id]
                                : ids.filter((id) => id !== job.id)
                            )
                          }
                        />
                      </td>
                      <td className={cell}>
                        <button
                          type="button"
                          className="text-primary underline-offset-4 hover:underline"
                          onClick={() => setDetailId(job.id)}
                        >
                          {job.name}
                        </button>
                      </td>
                      <td className={cell}>{job.config.model_uid}</td>
                      <td className={cell}>{new Date(job.created_at).toLocaleString()}</td>
                      <td className={cell}>
                        {t(`benchmarks.states.${job.state}`)} {active(job) && `${job.progress}%`}
                      </td>
                      <td className={`${cell} flex gap-1`}>
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label={t('benchmarks.export')}
                          onClick={() => void download(job.id)}
                        >
                          <Download />
                        </Button>
                        {active(job) ? (
                          <Button
                            variant="ghost"
                            size="icon"
                            disabled={busy}
                            aria-label={t('benchmarks.stop')}
                            onClick={() => void mutate(job.id, 'stop')}
                          >
                            <Square />
                          </Button>
                        ) : (
                          <Button
                            variant="ghost"
                            size="icon"
                            aria-label={t('common.delete')}
                            onClick={() => setDeleteId(job.id)}
                          >
                            <Trash2 />
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {compared.length > 0 && (
          <section className="space-y-3">
            <h2 className="font-semibold">
              {t('benchmarks.compare')} ({compared.length})
            </h2>
            <p className="text-xs text-muted-foreground">{t('benchmarks.compareHint')}</p>
            <BenchmarkCharts runs={compared} />
            {results(compared)}
            <div className="grid gap-3 lg:grid-cols-2">
              {compared.map((job) => (
                <details key={job.id} className="rounded-lg border p-3">
                  <summary>
                    {job.name} · {job.runner.name} {job.runner.version}
                  </summary>
                  <pre className="mt-3 overflow-auto text-xs">
                    {JSON.stringify(
                      { config: job.config, runner: job.runner, snapshot: job.snapshot },
                      null,
                      2
                    )}
                  </pre>
                </details>
              ))}
            </div>
          </section>
        )}

        {detail && (
          <section className="rounded-xl border p-5 space-y-3">
            <h2 className="font-semibold">
              {detail.name} · {t(`benchmarks.states.${detail.state}`)}
            </h2>
            {detail.error && (
              <p role="alert" className="text-sm text-destructive">
                {detail.error}
              </p>
            )}
            {detail.results.length > 0 && (
              <>
                <BenchmarkCharts runs={[detail]} />
                {results([detail])}
              </>
            )}
            <pre className="overflow-auto text-xs text-muted-foreground">
              {detail.logs.join('\n')}
            </pre>
            <details>
              <summary>{t('benchmarks.configuration')}</summary>
              <pre className="mt-3 overflow-auto text-xs">
                {JSON.stringify(
                  { config: detail.config, runner: detail.runner, snapshot: detail.snapshot },
                  null,
                  2
                )}
              </pre>
            </details>
            <Button
              variant="outline"
              onClick={async () => {
                const data = await request.get<{ logs: string }>(
                  `/v1/benchmarks/${detail.id}/logs`
                );
                setRunnerLog(data.logs);
              }}
            >
              {t('benchmarks.logs')}
            </Button>
            {runnerLog && (
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap text-xs">{runnerLog}</pre>
            )}
          </section>
        )}
        <ConfirmDialog
          isOpen={Boolean(deleteId)}
          onOpenChange={(open) => {
            if (!open) setDeleteId(undefined);
          }}
          onConfirm={() => {
            if (deleteId) void mutate(deleteId, 'delete');
          }}
          isLoading={busy}
        />
      </div>
    </PageContainer>
  );
}
