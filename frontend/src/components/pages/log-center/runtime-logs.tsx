'use client';

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, ChevronDown, Pause, Play } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { cn } from '@/lib/utils';

import { LogDetail } from './log-detail';
import { LogToolbar } from './log-toolbar';
import {
  filterRuntimeLogEntries,
  getRuntimeHistoricalSearch,
  mergeRuntimeLogEntries,
  normalizeRuntimeLogSource,
  parseRuntimeLogEntries,
  parseRuntimeLogPayload,
  runtimeLogSourceSearchText,
  tailRuntimeLogEntries,
  trimRuntimeLogBuffer,
  type NormalizedRuntimeLogSource,
  type RuntimeLogEntry,
  type RuntimeLogSource,
  type RuntimeLogSourceRole,
} from './runtime-log-utils';
import type { HistoricalLogHandoff, LogNodeOption, LogRow, RuntimeNodeHandoff } from './types';
import { formatLogTime, formatLogTimeTitle, LogLevelBadge, NodeRoleBadge } from './utils';

interface RuntimeLogsProps {
  active: boolean;
  historicalSearchEnabled: boolean;
  targetNodeRequest?: RuntimeNodeHandoff;
  maxLinesInput: string;
  followTail: boolean;
  onFollowTailChange: (follow: boolean) => void;
  onViewHistorical: (request: Omit<HistoricalLogHandoff, 'token'>) => void;
}

interface RuntimeLogChunk {
  text: string;
  cursor: string;
  has_more: boolean;
  reset: boolean;
}

interface RuntimeLogSourceState {
  cursor: string;
  rawText: string;
  error: boolean;
  errorMessage: string;
  missing: boolean;
}

const POLL_INTERVAL_MS = 2000;
const SOURCE_REFRESH_INTERVAL_MS = 10000;
const MAX_SELECTED_SOURCES = 10;
const MAX_SOURCE_BUFFER_CHARS = 2 * 1024 * 1024;
const MAX_TOTAL_BUFFER_CHARS = 8 * 1024 * 1024;
const DEFAULT_MAX_DISPLAY_LINES = 100;
const MAX_DISPLAY_LINES = 10000;

function createSourceState(): RuntimeLogSourceState {
  return { cursor: '', rawText: '', error: false, errorMessage: '', missing: false };
}

function errorStatus(error: unknown): number | undefined {
  if (typeof error !== 'object' || error === null || !('response' in error)) return undefined;
  const response = (error as { response?: { status?: unknown } }).response;
  return typeof response?.status === 'number' ? response.status : undefined;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '';
}

function normalizedLevel(level?: string): string {
  return level === 'WARN' ? 'WARNING' : level || '';
}

export default function RuntimeLogs({
  active,
  historicalSearchEnabled,
  targetNodeRequest,
  maxLinesInput,
  followTail,
  onFollowTailChange,
  onViewHistorical,
}: RuntimeLogsProps) {
  const { t } = useI18n();
  const [sources, setSources] = useState<NormalizedRuntimeLogSource[]>([]);
  const [sourceRegistry, setSourceRegistry] = useState<Record<string, NormalizedRuntimeLogSource>>(
    {}
  );
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [sourceStates, setSourceStates] = useState<Record<string, RuntimeLogSourceState>>({});
  const [sourceListUnavailable, setSourceListUnavailable] = useState(false);
  const [search, setSearch] = useState('');
  const [appliedSearch, setAppliedSearch] = useState('');
  const [selectedLevels, setSelectedLevels] = useState<string[]>([]);
  const [paused, setPaused] = useState(false);
  const [pollKey, setPollKey] = useState(0);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number>();
  const [expandedEntryId, setExpandedEntryId] = useState<string>();
  const [targetNodeUnavailable, setTargetNodeUnavailable] = useState('');
  const [targetNodeSelectionLimitReached, setTargetNodeSelectionLimitReached] = useState(false);
  const sourceStatesRef = useRef(sourceStates);
  const sourceRegistryRef = useRef(sourceRegistry);
  const selectedSourcesRef = useRef(selectedSources);
  const pausedRef = useRef(paused);
  const busySourcesRef = useRef<Record<string, boolean>>({});
  const initializedSelectionRef = useRef(false);
  const appliedTargetNodeRef = useRef<number | undefined>(undefined);
  const viewerRef = useRef<HTMLDivElement>(null);

  const parsedMaxLines = Number(maxLinesInput);
  const maxLines =
    Number.isSafeInteger(parsedMaxLines) && parsedMaxLines > 0
      ? Math.min(parsedMaxLines, MAX_DISPLAY_LINES)
      : DEFAULT_MAX_DISPLAY_LINES;

  const replaceSourceStates = useCallback((next: Record<string, RuntimeLogSourceState>) => {
    sourceStatesRef.current = next;
    setSourceStates(next);
  }, []);

  const updateSourceState = useCallback(
    (source: string, updater: (current: RuntimeLogSourceState) => RuntimeLogSourceState): void => {
      const current = sourceStatesRef.current[source] || createSourceState();
      replaceSourceStates({
        ...sourceStatesRef.current,
        [source]: updater(current),
      });
    },
    [replaceSourceStates]
  );

  const roleLabel = useCallback(
    (role: RuntimeLogSourceRole) => {
      if (role === 'local') return t('logCenter.local');
      if (role === 'supervisor') return t('clusterInfo.supervisor');
      return t('clusterInfo.worker');
    },
    [t]
  );

  const sourceMetadata = useCallback((source: string): NormalizedRuntimeLogSource => {
    return (
      sourceRegistryRef.current[source] || normalizeRuntimeLogSource({ id: source, label: source })
    );
  }, []);

  const sourceIdentity = useCallback(
    (source: string) => {
      const metadata = sourceMetadata(source);
      return `${roleLabel(metadata.role)} ${metadata.displayNodeName}`;
    },
    [roleLabel, sourceMetadata]
  );

  const applySelectedSources = useCallback(
    (nextSelected: string[]) => {
      const normalized = Array.from(new Set(nextSelected)).slice(0, MAX_SELECTED_SOURCES);
      const selected = new Set(normalized);
      const nextStates: Record<string, RuntimeLogSourceState> = {};
      normalized.forEach((source) => {
        nextStates[source] = sourceStatesRef.current[source] || createSourceState();
      });
      Object.keys(busySourcesRef.current).forEach((source) => {
        if (!selected.has(source)) delete busySourcesRef.current[source];
      });
      selectedSourcesRef.current = normalized;
      setSelectedSources(normalized);
      replaceSourceStates(nextStates);
      setTargetNodeUnavailable('');
      setTargetNodeSelectionLimitReached(false);
      onFollowTailChange(true);
    },
    [onFollowTailChange, replaceSourceStates]
  );

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    const timer = setTimeout(() => setAppliedSearch(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    if (!active) return;
    let alive = true;

    const fetchSources = async () => {
      try {
        const data = await request.get<{ sources: RuntimeLogSource[] }>(
          '/v1/cluster/runtime-logs/sources',
          { suppressGlobalError: true }
        );
        if (!alive) return;
        const nextSources = Array.isArray(data.sources)
          ? data.sources.map(normalizeRuntimeLogSource)
          : [];
        const available = new Set(nextSources.map((source) => source.id));
        const nextRegistry = { ...sourceRegistryRef.current };
        nextSources.forEach((source) => {
          nextRegistry[source.id] = source;
        });
        sourceRegistryRef.current = nextRegistry;
        setSourceRegistry(nextRegistry);
        setSources(nextSources);
        setSourceListUnavailable(false);

        if (!initializedSelectionRef.current) {
          initializedSelectionRef.current = true;
          if (nextSources[0]) applySelectedSources([nextSources[0].id]);
          return;
        }

        const nextStates = { ...sourceStatesRef.current };
        let changed = false;
        selectedSourcesRef.current.forEach((source) => {
          const current = nextStates[source] || createSourceState();
          const missing = !available.has(source);
          if (current.missing !== missing) {
            nextStates[source] = missing
              ? { ...current, missing: true, error: true }
              : { ...current, missing: false, error: false, errorMessage: '' };
            changed = true;
          }
        });
        if (changed) replaceSourceStates(nextStates);
      } catch {
        if (alive) setSourceListUnavailable(true);
      }
    };

    void fetchSources();
    const timer = window.setInterval(() => void fetchSources(), SOURCE_REFRESH_INTERVAL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [active, applySelectedSources, replaceSourceStates]);

  useEffect(() => {
    if (
      !active ||
      !targetNodeRequest ||
      appliedTargetNodeRef.current === targetNodeRequest.token ||
      sources.length === 0
    ) {
      return;
    }

    const target = sources.find(
      (source) =>
        source.id === targetNodeRequest.nodeName ||
        source.nodeName === targetNodeRequest.nodeName ||
        source.displayNodeName === targetNodeRequest.nodeName
    );
    if (!target) {
      setTargetNodeUnavailable(targetNodeRequest.nodeName);
      setTargetNodeSelectionLimitReached(false);
      return;
    }

    appliedTargetNodeRef.current = targetNodeRequest.token;
    setTargetNodeUnavailable('');
    setTargetNodeSelectionLimitReached(false);
    if (!selectedSourcesRef.current.includes(target.id)) {
      if (selectedSourcesRef.current.length >= MAX_SELECTED_SOURCES) {
        setTargetNodeSelectionLimitReached(true);
        return;
      }
      applySelectedSources([...selectedSourcesRef.current, target.id]);
    }
  }, [active, applySelectedSources, sources, targetNodeRequest]);

  useEffect(() => {
    if (!active) return;
    let alive = true;

    const pollSource = async (source: string) => {
      const initial = sourceStatesRef.current[source] || createSourceState();
      if (busySourcesRef.current[source] || initial.missing || pausedRef.current) return;
      busySourcesRef.current[source] = true;

      let cursor = initial.cursor;
      let rawText = initial.rawText;
      try {
        for (let read = 0; read < 4; read++) {
          const params = new URLSearchParams({ source, cursor });
          const chunk = await request.get<RuntimeLogChunk>(
            `/v1/cluster/runtime-logs?${params.toString()}`,
            { suppressGlobalError: true }
          );
          if (!alive || pausedRef.current || !selectedSourcesRef.current.includes(source)) return;

          cursor = chunk.cursor;
          rawText = chunk.reset ? chunk.text : rawText + chunk.text;
          const sourceCount = Math.max(1, selectedSourcesRef.current.length);
          const sourceBudget = Math.min(
            MAX_SOURCE_BUFFER_CHARS,
            Math.floor(MAX_TOTAL_BUFFER_CHARS / sourceCount)
          );
          rawText = trimRuntimeLogBuffer(rawText, sourceBudget);
          if (!chunk.has_more) break;
        }

        if (!alive || pausedRef.current || !selectedSourcesRef.current.includes(source)) return;
        updateSourceState(source, (current) => ({
          ...current,
          cursor,
          rawText,
          error: false,
          errorMessage: '',
          missing: false,
        }));
        setLastUpdatedAt(Date.now());
      } catch (error) {
        if (!alive || !selectedSourcesRef.current.includes(source)) return;
        const status = errorStatus(error);
        updateSourceState(source, (current) => ({
          ...current,
          error: true,
          errorMessage: errorMessage(error),
          missing: status === 404,
        }));
      } finally {
        busySourcesRef.current[source] = false;
      }
    };

    const pollAll = () => {
      if (pausedRef.current) return;
      selectedSourcesRef.current.forEach((source) => void pollSource(source));
    };

    pollAll();
    const timer = window.setInterval(pollAll, POLL_INTERVAL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [active, pollKey, selectedSources, updateSourceState]);

  const entries = useMemo(() => {
    const entriesBySource = Object.fromEntries(
      selectedSources.map((source) => [
        source,
        parseRuntimeLogEntries(sourceStates[source]?.rawText || '', source),
      ])
    );
    return mergeRuntimeLogEntries(entriesBySource, selectedSources);
  }, [selectedSources, sourceStates]);

  const sourceSearchText = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(sourceRegistry).map(([id, source]) => [
          id,
          runtimeLogSourceSearchText(source, roleLabel(source.role)),
        ])
      ),
    [roleLabel, sourceRegistry]
  );

  const displayedEntries = useMemo(() => {
    const recentEntries = tailRuntimeLogEntries(entries, maxLines);
    const searchedEntries = filterRuntimeLogEntries(recentEntries, appliedSearch, sourceSearchText);
    if (selectedLevels.length === 0) return searchedEntries;
    return searchedEntries.filter((entry) => selectedLevels.includes(normalizedLevel(entry.level)));
  }, [appliedSearch, entries, maxLines, selectedLevels, sourceSearchText]);

  useEffect(() => {
    if (active && !paused && !appliedSearch && followTail && viewerRef.current) {
      viewerRef.current.scrollTop = viewerRef.current.scrollHeight;
    }
  }, [active, appliedSearch, displayedEntries, followTail, paused]);

  const nodeOptions = useMemo<LogNodeOption[]>(
    () =>
      sources.map((source) => ({
        value: source.id,
        label: source.displayNodeName,
        role: source.role,
        roleLabel: roleLabel(source.role),
        fullAddress: source.nodeName,
        searchText: runtimeLogSourceSearchText(source, roleLabel(source.role)),
      })),
    [roleLabel, sources]
  );

  const toggleLevel = (level: string) => {
    setSelectedLevels((current) =>
      current.includes(level) ? current.filter((value) => value !== level) : [...current, level]
    );
  };

  const handleResetFilters = () => {
    setSearch('');
    setAppliedSearch('');
    setSelectedLevels([]);
  };

  const handleRefresh = () => {
    const resetStates = Object.fromEntries(
      selectedSources.map((source) => [source, createSourceState()])
    );
    replaceSourceStates(resetStates);
    setPaused(false);
    pausedRef.current = false;
    onFollowTailChange(true);
    setExpandedEntryId(undefined);
    setPollKey((value) => value + 1);
  };

  const handleRetry = (source: string) => {
    const available = sources.some((item) => item.id === source);
    updateSourceState(source, (current) => ({
      ...current,
      error: false,
      errorMessage: '',
      missing: !available,
    }));
    if (available) setPollKey((value) => value + 1);
  };

  const viewHistorical = (entry: RuntimeLogEntry) => {
    const metadata = sourceMetadata(entry.source);
    const search = getRuntimeHistoricalSearch(entry);
    onViewHistorical({
      nodeName: metadata.nodeName,
      timestamp: entry.timestamp,
      level: normalizedLevel(entry.level),
      ...search,
    });
  };

  const runtimeDetailRow = (entry: RuntimeLogEntry): LogRow => {
    const metadata = sourceMetadata(entry.source);
    return {
      ...parseRuntimeLogPayload(entry.raw),
      '@timestamp': entry.timestamp,
      level: normalizedLevel(entry.level),
      role: metadata.role,
      address: metadata.nodeName,
      node_name: metadata.nodeName,
      node: metadata.nodeName,
      source_address: metadata.nodeName,
      message: entry.message,
      raw: entry.raw,
    };
  };

  const normalCount = selectedSources.filter((source) => {
    const state = sourceStates[source];
    return state && !state.error && !state.missing;
  }).length;
  const retryingCount = selectedSources.filter(
    (source) => sourceStates[source]?.error && !sourceStates[source]?.missing
  ).length;
  const missingCount = selectedSources.filter((source) => sourceStates[source]?.missing).length;

  return (
    <div className="flex h-[calc(100vh-15rem)] min-h-[36rem] flex-col overflow-hidden rounded-md border bg-background">
      <LogToolbar
        nodeOptions={nodeOptions}
        selectedNodes={selectedSources}
        onSelectedNodesChange={applySelectedSources}
        searchText={search}
        onSearchTextChange={setSearch}
        onSearchCommit={() => setAppliedSearch(search)}
        selectedLevels={selectedLevels}
        onToggleLevel={toggleLevel}
        maxSelectedNodes={MAX_SELECTED_SOURCES}
        onReset={handleResetFilters}
        actionLabel={t('logCenter.refresh')}
        onAction={handleRefresh}
        actionDisabled={selectedSources.length === 0}
      />

      <div className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-2 text-xs text-muted-foreground">
        <span>
          {t('logCenter.runtimeStatus', {
            selected: selectedSources.length,
            normal: normalCount,
            retrying: retryingCount,
            missing: missingCount,
          })}
        </span>
        <div className="flex items-center gap-2">
          <span className={cn('font-medium', paused ? 'text-amber-600' : 'text-emerald-600')}>
            {paused ? t('logCenter.runtimePaused') : t('logCenter.runtimeLive')}
            {lastUpdatedAt
              ? ` · ${t('logCenter.lastUpdated')} ${new Date(lastUpdatedAt).toLocaleTimeString()}`
              : ''}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={selectedSources.length === 0}
            onClick={() => {
              const nextPaused = !paused;
              setPaused(nextPaused);
              pausedRef.current = nextPaused;
              if (!nextPaused) setPollKey((value) => value + 1);
            }}
          >
            {paused ? <Play className="size-4" /> : <Pause className="size-4" />}
            {paused ? t('logCenter.resume') : t('logCenter.pause')}
          </Button>
        </div>
      </div>

      {(sourceListUnavailable || targetNodeUnavailable || targetNodeSelectionLimitReached) && (
        <div className="border-b px-4 py-2 text-sm text-destructive">
          {sourceListUnavailable && <p>{t('logCenter.sourceListUnavailable')}</p>}
          {targetNodeUnavailable && (
            <p>{t('logCenter.runtimeNodeUnavailable', { node: targetNodeUnavailable })}</p>
          )}
          {targetNodeSelectionLimitReached && (
            <p>{t('logCenter.selectionLimit', { count: MAX_SELECTED_SOURCES })}</p>
          )}
        </div>
      )}

      {selectedSources.some(
        (source) => sourceStates[source]?.error || sourceStates[source]?.missing
      ) && (
        <div className="flex flex-col gap-1.5 border-b p-3">
          {selectedSources.map((source) => {
            const state = sourceStates[source];
            if (!state?.error && !state?.missing) return null;
            return (
              <div
                key={source}
                className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
                title={state.errorMessage || undefined}
              >
                <span>
                  {state.missing
                    ? t('logCenter.nodeMissing', { node: sourceIdentity(source) })
                    : t('logCenter.nodeRetrying', { node: sourceIdentity(source) })}
                </span>
                <span className="flex gap-1">
                  <Button size="sm" variant="ghost" onClick={() => handleRetry(source)}>
                    {t('logCenter.retryNow')}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() =>
                      applySelectedSources(selectedSources.filter((item) => item !== source))
                    }
                  >
                    {t('logCenter.remove')}
                  </Button>
                </span>
              </div>
            );
          })}
        </div>
      )}

      <div className="relative min-h-0 flex-1">
        <div
          ref={viewerRef}
          className="h-full overflow-auto"
          role="log"
          aria-live="off"
          onScroll={() => {
            const viewer = viewerRef.current;
            if (!viewer) return;
            const distance = viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight;
            onFollowTailChange(distance < 48);
          }}
        >
          <Table size="small" className="min-w-[960px] table-auto">
            <colgroup>
              <col className="w-9" />
              <col className="w-px" />
              <col className="w-px" />
              <col className="w-[18%]" />
              <col className="w-px" />
              <col />
            </colgroup>
            <TableHeader className="sticky top-0 z-10">
              <TableRow>
                <TableHead />
                <TableHead className="whitespace-nowrap">{t('logCenter.time')}</TableHead>
                <TableHead className="whitespace-nowrap">{t('logCenter.nodeRole')}</TableHead>
                <TableHead>{t('logCenter.nodeName')}</TableHead>
                <TableHead className="whitespace-nowrap">{t('logCenter.level')}</TableHead>
                <TableHead>{t('logCenter.message')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {selectedSources.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6}>
                    <p className="py-10 text-center text-sm text-muted-foreground">
                      {t('logCenter.noNodeSelected')}
                    </p>
                  </TableCell>
                </TableRow>
              )}
              {selectedSources.length > 0 && displayedEntries.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6}>
                    <p className="py-10 text-center text-sm text-muted-foreground">
                      {t('logCenter.noRuntimeLogs')}
                    </p>
                  </TableCell>
                </TableRow>
              )}
              {displayedEntries.map((entry) => {
                const metadata = sourceMetadata(entry.source);
                const expanded = expandedEntryId === entry.id;
                return (
                  <Fragment key={entry.id}>
                    <TableRow
                      className={cn('cursor-pointer', expanded && '[&>td]:border-b-0')}
                      onClick={() => setExpandedEntryId(expanded ? undefined : entry.id)}
                    >
                      <TableCell>
                        <ChevronDown
                          className={cn('size-4 transition-transform', expanded && 'rotate-180')}
                        />
                      </TableCell>
                      <TableCell
                        className="whitespace-nowrap font-mono text-xs text-muted-foreground"
                        title={formatLogTimeTitle(entry.timestamp)}
                      >
                        {formatLogTime(entry.timestamp)}
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-xs">
                        <NodeRoleBadge role={metadata.role}>
                          {roleLabel(metadata.role)}
                        </NodeRoleBadge>
                      </TableCell>
                      <TableCell
                        className="min-w-[140px] max-w-[220px] whitespace-normal break-words [overflow-wrap:anywhere] font-mono text-xs"
                        title={metadata.nodeName}
                      >
                        {metadata.displayNodeName}
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-xs">
                        <LogLevelBadge level={normalizedLevel(entry.level)} />
                      </TableCell>
                      <TableCell className="whitespace-pre-wrap break-words [overflow-wrap:anywhere] font-mono text-xs">
                        {entry.message}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell colSpan={6} className="p-0">
                        {expanded && (
                          <LogDetail
                            mode="runtime"
                            row={runtimeDetailRow(entry)}
                            onFilter={() => undefined}
                            fieldFilters={[]}
                            appliedSearch={appliedSearch}
                            selectedLevels={selectedLevels}
                            selectedLogType=""
                            nodeField="node_name"
                            onViewHistorical={
                              historicalSearchEnabled ? () => viewHistorical(entry) : undefined
                            }
                          />
                        )}
                      </TableCell>
                    </TableRow>
                  </Fragment>
                );
              })}
            </TableBody>
          </Table>
        </div>
        {!followTail && displayedEntries.length > 0 && (
          <Button
            size="sm"
            className="absolute bottom-4 right-4 shadow-md"
            onClick={() => {
              onFollowTailChange(true);
              if (viewerRef.current) viewerRef.current.scrollTop = viewerRef.current.scrollHeight;
            }}
          >
            <ArrowDown className="size-4" />
            {t('logCenter.backToBottom')}
          </Button>
        )}
      </div>
    </div>
  );
}
