'use client';

import { useId, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useI18n } from '@/contexts/i18n-context';
import {
  buildChartData,
  haveDifferentWorkloads,
  type ChartMetric,
  type ChartRun,
  type LatencyStatistic,
} from './chart-data';

const colors = ['#2563eb', '#ea580c', '#16a34a', '#9333ea', '#0891b2', '#db2777', '#a16207'];
const metrics: ChartMetric[] = ['output_throughput', 'ttft_ms', 'tpot_ms'];

export default function BenchmarkCharts({ runs }: { runs: ChartRun[] }) {
  const { t } = useI18n();
  const selectId = useId();
  const [statistic, setStatistic] = useState<LatencyStatistic>('mean');
  const labels = runs.map((run) => `${run.name} · ${new Date(run.created_at).toLocaleString()}`);

  return (
    <div className="min-w-0 space-y-3" data-testid="benchmark-charts">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-medium">{t('benchmarks.charts.title')}</h3>
        <div className="flex items-center gap-2 text-sm">
          <label htmlFor={selectId}>{t('benchmarks.charts.statistic')}</label>
          <select
            id={selectId}
            className="h-8 rounded-md border bg-background px-2"
            value={statistic}
            onChange={(event) => setStatistic(event.target.value as LatencyStatistic)}
          >
            <option value="mean">{t('benchmarks.charts.mean')}</option>
            <option value="p95">P95</option>
            <option value="p99">P99</option>
          </select>
        </div>
      </div>
      {haveDifferentWorkloads(runs) && (
        <p
          role="status"
          className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm"
        >
          {t('benchmarks.charts.differentWorkloads')}
        </p>
      )}
      <ul
        className="flex flex-wrap gap-x-5 gap-y-2 text-xs"
        aria-label={t('benchmarks.charts.legend')}
      >
        {runs.map((run, index) => (
          <li key={run.id} className="flex min-w-0 items-center gap-2">
            <span
              className="inline-block h-0.5 w-5 shrink-0"
              style={{ backgroundColor: colors[index % colors.length] }}
              aria-hidden="true"
            />
            <span>
              {labels[index]}
              {run.results.length === 0 && ` · ${t('benchmarks.charts.noData')}`}
            </span>
          </li>
        ))}
      </ul>
      <div className="grid min-w-0 gap-4 xl:grid-cols-3">
        {metrics.map((metric) => {
          const data = buildChartData(runs, metric, statistic);
          const hasData = data.some((row) => runs.some((_, index) => row[`run${index}`] !== null));
          const metricTitle = t(`benchmarks.charts.${metric}`);
          const statisticTitle =
            statistic === 'mean' ? t('benchmarks.charts.mean') : statistic.toUpperCase();
          const title =
            metric === 'output_throughput' ? metricTitle : `${metricTitle} · ${statisticTitle}`;
          const unit = metric === 'output_throughput' ? 'tok/s' : 'ms';
          const first = data[0]?.concurrency ?? 0;
          const last = data[data.length - 1]?.concurrency ?? 1;
          return (
            <figure key={metric} className="min-w-0 rounded-lg border bg-card p-3">
              <figcaption className="mb-3 text-sm font-medium">
                {title} <span className="text-xs text-muted-foreground">({unit})</span>
              </figcaption>
              <div className="h-64 min-w-0" role="img" aria-label={`${title} (${unit})`}>
                {hasData ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.12} />
                      <XAxis
                        dataKey="concurrency"
                        type="number"
                        domain={first === last ? [0, Math.max(last, 1)] : [first, last]}
                        ticks={data.map((row) => row.concurrency)}
                        allowDecimals={false}
                        tick={{ fontSize: 11, fill: 'currentColor' }}
                        label={{
                          value: t('benchmarks.charts.concurrency'),
                          position: 'insideBottom',
                          offset: -12,
                          fontSize: 11,
                          fill: 'currentColor',
                        }}
                      />
                      <YAxis
                        width={52}
                        domain={[0, 'auto']}
                        tick={{ fontSize: 11, fill: 'currentColor' }}
                        tickFormatter={(value: number) =>
                          new Intl.NumberFormat(undefined, {
                            notation: 'compact',
                            maximumFractionDigits: 1,
                          }).format(value)
                        }
                      />
                      <Tooltip
                        labelFormatter={(value) =>
                          `${t('benchmarks.charts.concurrency')}: ${value}`
                        }
                        formatter={(value) => `${Number(value).toFixed(2)} ${unit}`}
                        contentStyle={{
                          backgroundColor: 'hsl(var(--card))',
                          color: 'hsl(var(--foreground))',
                          borderColor: 'hsl(var(--border))',
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      {runs.map((run, index) => (
                        <Line
                          key={run.id}
                          dataKey={`run${index}`}
                          name={labels[index]}
                          type="linear"
                          stroke={colors[index % colors.length]}
                          strokeDasharray={index >= colors.length ? '5 3' : undefined}
                          strokeWidth={2}
                          dot={{ r: 3 }}
                          activeDot={{ r: 5 }}
                          connectNulls={false}
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    {t('benchmarks.charts.noData')}
                  </p>
                )}
              </div>
            </figure>
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground">{t('benchmarks.charts.hint')}</p>
    </div>
  );
}
