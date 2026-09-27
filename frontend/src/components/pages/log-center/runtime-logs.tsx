'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import PageContainer from '@/components/ui/page-container';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { tailLogLines } from './runtime-log-utils';

interface LogSource {
  id: string;
  label: string;
}

interface RuntimeLogChunk {
  text: string;
  cursor: string;
  has_more: boolean;
  reset: boolean;
}

const MAX_DISPLAY_CHARS = 2 * 1024 * 1024;
const DEFAULT_MAX_DISPLAY_LINES = 100;
const MAX_DISPLAY_LINES = 10000;

export default function RuntimeLogs() {
  const { t } = useI18n();
  const [sources, setSources] = useState<LogSource[]>([]);
  const [source, setSource] = useState('');
  const [logs, setLogs] = useState('');
  const [search, setSearch] = useState('');
  const [maxLinesInput, setMaxLinesInput] = useState(String(DEFAULT_MAX_DISPLAY_LINES));
  const [paused, setPaused] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState(false);
  const pausedRef = useRef(paused);
  const viewerRef = useRef<HTMLDivElement>(null);
  const parsedMaxLines = Number(maxLinesInput);
  const maxLines =
    Number.isSafeInteger(parsedMaxLines) && parsedMaxLines > 0
      ? Math.min(parsedMaxLines, MAX_DISPLAY_LINES)
      : DEFAULT_MAX_DISPLAY_LINES;

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    let active = true;
    const fetchSources = () => {
      request
        .get<{ sources: LogSource[] }>('/v1/cluster/runtime-logs/sources')
        .then((data) => {
          if (!active) return;
          setSources(data.sources);
          setSource((current) =>
            data.sources.some((item) => item.id === current) ? current : (data.sources[0]?.id ?? '')
          );
        })
        .catch(() => {
          if (active) setError(true);
        });
    };
    fetchSources();
    const timer = setInterval(fetchSources, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!source) return;
    let active = true;
    let busy = false;
    let cursor = '';
    setLogs('');
    setError(false);

    const poll = async () => {
      if (busy || pausedRef.current) return;
      busy = true;
      try {
        for (let read = 0; read < 4; read++) {
          const params = new URLSearchParams({ source, cursor });
          const chunk = await request.get<RuntimeLogChunk>(
            `/v1/cluster/runtime-logs?${params.toString()}`
          );
          if (!active) return;
          cursor = chunk.cursor;
          setLogs((previous) => {
            const combined = chunk.reset ? chunk.text : previous + chunk.text;
            return combined.slice(-MAX_DISPLAY_CHARS);
          });
          if (!chunk.has_more) break;
        }
        setError(false);
      } catch {
        if (active) setError(true);
      } finally {
        busy = false;
      }
    };

    poll();
    const timer = setInterval(poll, 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [source, refreshKey]);

  useEffect(() => {
    if (!paused && !search && viewerRef.current) {
      viewerRef.current.scrollTop = viewerRef.current.scrollHeight;
    }
  }, [logs, paused, search]);

  const displayedLogs = useMemo(() => {
    const recentLogs = tailLogLines(logs, maxLines);
    if (!search.trim()) return recentLogs;
    const needle = search.toLowerCase();
    return recentLogs
      .split(/(?=^\d{4}-\d{2}-\d{2}T|^\{"@timestamp":)/m)
      .filter((entry) => entry.toLowerCase().includes(needle))
      .join('');
  }, [logs, maxLines, search]);

  return (
    <PageContainer
      title={t('menu.logCenter')}
      subTitle={t('logCenter.runtimeDescription')}
      className="h-full gap-4"
    >
      <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-3">
        <div className="flex items-center gap-2">
          <label htmlFor="runtime-log-source" className="text-sm font-medium">
            {t('logCenter.node')}
          </label>
          <select
            id="runtime-log-source"
            className="h-9 max-w-full rounded-md border border-input bg-background px-3 text-sm"
            value={source}
            onChange={(event) => setSource(event.target.value)}
          >
            {sources.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id === 'local'
                  ? t('logCenter.local')
                  : item.id === 'supervisor'
                    ? t('clusterInfo.supervisor')
                    : `${t('clusterInfo.worker')} ${item.id}`}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="runtime-log-max-lines" className="text-sm font-medium">
            {t('logCenter.maxLines')}
          </label>
          <Input
            id="runtime-log-max-lines"
            type="number"
            min={1}
            max={MAX_DISPLAY_LINES}
            step={1}
            className="w-24"
            aria-label={t('logCenter.maxLines')}
            value={maxLinesInput}
            onChange={(event) => setMaxLinesInput(event.target.value)}
            onBlur={() => setMaxLinesInput(String(maxLines))}
          />
        </div>
        <Input
          className="w-64"
          aria-label={t('logCenter.searchPlaceholder')}
          placeholder={t('logCenter.searchPlaceholder')}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <Button variant="outline" onClick={() => setPaused((value) => !value)}>
          {paused ? <Play className="size-4" /> : <Pause className="size-4" />}
          {paused ? t('logCenter.resume') : t('logCenter.pause')}
        </Button>
        <Button
          variant="outline"
          onClick={() => {
            setPaused(false);
            pausedRef.current = false;
            setRefreshKey((value) => value + 1);
          }}
        >
          <RefreshCw className="size-4" />
          {t('logCenter.refresh')}
        </Button>
      </div>
      {error && <p className="text-sm text-destructive">{t('logCenter.runtimeUnavailable')}</p>}
      <div
        ref={viewerRef}
        className="h-[calc(100vh-16rem)] min-h-64 overflow-auto rounded-md border bg-muted/30 p-4"
        role="log"
        aria-live="off"
      >
        <pre className="whitespace-pre-wrap break-all font-mono text-xs leading-5">
          {displayedLogs || t('logCenter.noLogs')}
        </pre>
      </div>
    </PageContainer>
  );
}
