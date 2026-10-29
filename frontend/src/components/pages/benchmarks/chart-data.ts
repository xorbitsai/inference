export interface ChartDistribution {
  mean: number;
  p95: number;
  p99: number;
}

export interface ChartRun {
  id: string;
  name: string;
  created_at: string;
  config: {
    input_tokens: number;
    output_tokens: number;
    requests: number;
    warmup: number;
    seed: number;
  };
  runner: { name: string; metrics_version: string };
  results: Array<{
    concurrency: number;
    output_throughput: number | null;
    ttft_ms: ChartDistribution | null;
    tpot_ms: ChartDistribution | null;
  }>;
}

export type LatencyStatistic = 'mean' | 'p95' | 'p99';
export type ChartMetric = 'output_throughput' | 'ttft_ms' | 'tpot_ms';
export type ChartRow = { concurrency: number } & Record<string, number | null>;

export function buildChartData(
  runs: ChartRun[],
  metric: ChartMetric,
  statistic: LatencyStatistic
): ChartRow[] {
  const points = [...new Set(runs.flatMap((run) => run.results.map((p) => p.concurrency)))].sort(
    (a, b) => a - b
  );
  const indexed = runs.map((run) => new Map(run.results.map((p) => [p.concurrency, p])));
  return points.map((concurrency) => {
    const row: ChartRow = { concurrency };
    indexed.forEach((stages, index) => {
      const stage = stages.get(concurrency);
      const value = metric === 'output_throughput' ? stage?.[metric] : stage?.[metric]?.[statistic];
      // A missing point/metric is unavailable, never a measured zero.
      row[`run${index}`] = typeof value === 'number' && Number.isFinite(value) ? value : null;
    });
    return row;
  });
}

export function haveDifferentWorkloads(runs: ChartRun[]): boolean {
  const signatures = runs.map((run) =>
    JSON.stringify([
      run.config.input_tokens,
      run.config.output_tokens,
      run.config.requests,
      run.config.warmup,
      run.config.seed,
      run.runner.name,
      run.runner.metrics_version,
    ])
  );
  return new Set(signatures).size > 1;
}
