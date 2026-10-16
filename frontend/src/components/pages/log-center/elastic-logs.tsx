'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { DEFAULT_LOG_TIME_RANGE, LOG_PAGE_SIZE } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';

import { FilterChipBar } from './filter-chip-bar';
import { LogPagination } from './pagination';
import { LogTable } from './log-table';
import { LogToolbar } from './log-toolbar';
import type {
  FieldFilter,
  FieldFilterOp,
  HistoricalLogHandoff,
  LogNodeOption,
  LogNodeRole,
  LogNodesResponse,
  LogsResponse,
  TimeRangeValue,
} from './types';
import {
  buildHistoricalHandoffQueryState,
  normalizeRuntimeLogSource,
  type RuntimeLogSource,
} from './runtime-log-utils';
import {
  buildLogQueryParams,
  getLogNodeFilterValue,
  getLogNodeName,
  getLogNodeRole,
  normalizeLogNodeRole,
  resolveHistoricalNodeRole,
} from './utils';

interface ElasticLogsProps {
  active: boolean;
  enabled: boolean;
  handoff?: HistoricalLogHandoff;
  timeRange: TimeRangeValue;
  onTimeRangeChange: (value: TimeRangeValue) => void;
  refreshInterval: number;
  onRefreshIntervalChange: (value: number) => void;
  onViewRuntimeNode: (nodeName: string) => void;
}

interface HistoricalQuerySnapshot {
  logs: LogsResponse['hits'];
  total: number;
  searchText: string;
  appliedSearch: string;
  selectedLevels: string[];
  selectedLogType: string;
  selectedNodes: string[];
  pageFrom: number;
  fieldFilters: FieldFilter[];
  timeRange: TimeRangeValue;
  refreshInterval: number;
}

const HANDOFF_WINDOW_MS = 5 * 60 * 1000;

export default function ElasticLogs({
  active,
  enabled,
  handoff,
  timeRange,
  onTimeRangeChange,
  refreshInterval,
  onRefreshIntervalChange,
  onViewRuntimeNode,
}: ElasticLogsProps) {
  const { t } = useI18n();
  const [logs, setLogs] = useState<LogsResponse['hits']>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [appliedSearch, setAppliedSearch] = useState('');
  const [selectedLevels, setSelectedLevels] = useState<string[]>([]);
  const [selectedLogType, setSelectedLogType] = useState('');
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [nodes, setNodes] = useState<string[]>([]);
  const [nodeField, setNodeField] = useState('node');
  const [historicalNodeRoles, setHistoricalNodeRoles] = useState<Record<string, LogNodeRole>>({});
  const [runtimeNodeRoles, setRuntimeNodeRoles] = useState<Record<string, LogNodeRole>>({});
  const [pageFrom, setPageFrom] = useState(0);
  const [fieldFilters, setFieldFilters] = useState<FieldFilter[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [handoffSnapshot, setHandoffSnapshot] = useState<HistoricalQuerySnapshot>();
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const appliedHandoffRef = useRef<number | undefined>(undefined);
  const appliedHandoffNodeRef = useRef<number | undefined>(undefined);
  const previousTimeRangeRef = useRef(timeRange);
  const preservePageForTimeRangeRef = useRef(false);
  const lastSuccessfulRequestRef = useRef<{
    queryKey: string;
    pageFrom: number;
  } | null>(null);

  const nodeRoleMap = useMemo(() => {
    const roles = new Map<string, ReturnType<typeof getLogNodeRole>>();
    (logs || []).forEach((row) => {
      const nodeName = getLogNodeName(row, nodeField);
      const filterValue = getLogNodeFilterValue(row, nodeField);
      const role = getLogNodeRole(row);
      if (nodeName && role) roles.set(nodeName, role);
      if (filterValue && role) roles.set(filterValue, role);
    });
    return roles;
  }, [logs, nodeField]);

  const nodeOptions = useMemo<LogNodeOption[]>(
    () =>
      nodes.map((node) => {
        const role = resolveHistoricalNodeRole({
          nodeName: node,
          historicalRole: historicalNodeRoles[node],
          resultRole: nodeRoleMap.get(node),
          runtimeRole: runtimeNodeRoles[node],
        });
        const roleLabel =
          role === 'supervisor'
            ? t('clusterInfo.supervisor')
            : role === 'worker'
              ? t('clusterInfo.worker')
              : role === 'local'
                ? t('logCenter.local')
                : t('logCenter.unknownNodeType');
        const displayNodeName = node;
        return {
          value: node,
          label: displayNodeName,
          role,
          roleLabel,
          fullAddress: node,
          searchText: `${roleLabel} ${displayNodeName} ${node}`,
        };
      }),
    [historicalNodeRoles, nodeRoleMap, nodes, runtimeNodeRoles, t]
  );

  useEffect(() => {
    if (!enabled || !active) return;

    let alive = true;
    const fetchNodes = async () => {
      const [historicalResult, runtimeResult] = await Promise.allSettled([
        request.get<LogNodesResponse>('/v1/cluster/logs/nodes'),
        request.get<{ sources: RuntimeLogSource[] }>('/v1/cluster/runtime-logs/sources', {
          suppressGlobalError: true,
        }),
      ]);
      if (!alive) return;

      if (historicalResult.status === 'fulfilled') {
        setNodes(Array.isArray(historicalResult.value.nodes) ? historicalResult.value.nodes : []);
        setNodeField(historicalResult.value.node_field || 'node');
        const roles: Record<string, LogNodeRole> = {};
        Object.entries(historicalResult.value.node_roles || {}).forEach(([node, value]) => {
          const role = normalizeLogNodeRole(value);
          if (role) roles[node] = role;
        });
        setHistoricalNodeRoles(roles);
      } else {
        setNodes([]);
        setNodeField('node');
        setHistoricalNodeRoles({});
      }

      if (runtimeResult.status === 'fulfilled' && Array.isArray(runtimeResult.value.sources)) {
        const roles: Record<string, LogNodeRole> = {};
        runtimeResult.value.sources.map(normalizeRuntimeLogSource).forEach((source) => {
          roles[source.id] = source.role;
          roles[source.nodeName] = source.role;
          roles[source.displayNodeName] = source.role;
        });
        setRuntimeNodeRoles(roles);
      } else {
        setRuntimeNodeRoles({});
      }
    };

    void fetchNodes();
    return () => {
      alive = false;
    };
  }, [active, enabled]);

  useEffect(() => {
    if (!handoff || !active || appliedHandoffRef.current === handoff.token) return;
    appliedHandoffRef.current = handoff.token;
    setHandoffSnapshot(
      (current) =>
        current || {
          logs,
          total,
          searchText,
          appliedSearch,
          selectedLevels,
          selectedLogType,
          selectedNodes,
          pageFrom,
          fieldFilters,
          timeRange,
          refreshInterval,
        }
    );

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    const linkedQuery = buildHistoricalHandoffQueryState(handoff);
    setSearchText(linkedQuery.searchText);
    setAppliedSearch(linkedQuery.appliedSearch);
    setSelectedLevels(linkedQuery.selectedLevels);
    setSelectedLogType(linkedQuery.selectedLogType);
    setSelectedNodes(linkedQuery.selectedNodes);
    setPageFrom(linkedQuery.pageFrom);
    setFieldFilters(linkedQuery.fieldFilters);

    if (handoff.timestamp) {
      const timestamp = new Date(handoff.timestamp).getTime();
      if (!Number.isNaN(timestamp)) {
        onTimeRangeChange({
          from: String(timestamp - HANDOFF_WINDOW_MS),
          to: String(timestamp + HANDOFF_WINDOW_MS),
        });
      }
    }
  }, [
    active,
    appliedSearch,
    fieldFilters,
    handoff,
    logs,
    pageFrom,
    refreshInterval,
    searchText,
    selectedLevels,
    selectedLogType,
    selectedNodes,
    timeRange,
    total,
    onTimeRangeChange,
  ]);

  useEffect(() => {
    if (
      !handoff ||
      !active ||
      appliedHandoffNodeRef.current === handoff.token ||
      nodes.length === 0
    ) {
      return;
    }
    appliedHandoffNodeRef.current = handoff.token;
    const matchedNode = nodes.find((node) => node === handoff.nodeName);
    setSelectedNodes(matchedNode ? [matchedNode] : []);
    setPageFrom(0);
  }, [active, handoff, nodes]);

  useEffect(() => {
    const previous = previousTimeRangeRef.current;
    const changed = previous.from !== timeRange.from || previous.to !== timeRange.to;
    previousTimeRangeRef.current = timeRange;
    if (!changed) return;
    if (preservePageForTimeRangeRef.current) {
      preservePageForTimeRangeRef.current = false;
      return;
    }
    setPageFrom(0);
  }, [timeRange]);

  const fetchLogs = useCallback(
    async (isActive: () => boolean) => {
      if (!enabled || !active || !isActive()) return;

      setLoading(true);
      const params = buildLogQueryParams({
        appliedSearch,
        selectedLevels,
        selectedLogType,
        selectedNode: selectedNodes.join(','),
        nodeField,
        timeRange,
        pageFrom,
        fieldFilters,
        size: LOG_PAGE_SIZE,
      });
      const queryKeyParams = new URLSearchParams(params);
      queryKeyParams.delete('page_from');
      queryKeyParams.delete('size');
      const queryKey = queryKeyParams.toString();

      try {
        const data = await request.get<LogsResponse>(`/v1/cluster/logs?${params.toString()}`);
        if (!isActive()) return;
        setLogs(data.hits || []);
        setTotal(data.total || 0);
        lastSuccessfulRequestRef.current = { queryKey, pageFrom };
      } catch {
        if (!isActive()) return;
        const lastSuccessfulRequest = lastSuccessfulRequestRef.current;
        if (lastSuccessfulRequest?.queryKey === queryKey) {
          setPageFrom(lastSuccessfulRequest.pageFrom);
        } else {
          setLogs([]);
          setTotal(0);
          setPageFrom(0);
        }
      } finally {
        if (isActive()) setLoading(false);
      }
    },
    [
      active,
      appliedSearch,
      enabled,
      fieldFilters,
      nodeField,
      pageFrom,
      selectedLevels,
      selectedLogType,
      selectedNodes,
      timeRange,
    ]
  );

  useEffect(() => {
    if (!enabled || !active) return;

    let alive = true;
    let busy = false;
    const poll = async () => {
      if (!alive || busy) return;
      busy = true;
      try {
        await fetchLogs(() => alive);
      } finally {
        busy = false;
      }
    };

    void poll();
    const timer = refreshInterval > 0 ? window.setInterval(poll, refreshInterval) : undefined;
    return () => {
      alive = false;
      if (timer) window.clearInterval(timer);
    };
  }, [active, enabled, fetchLogs, refreshInterval, refreshKey]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const commitSearch = useCallback(
    (value = searchText) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      setAppliedSearch(value);
      setPageFrom(0);
    },
    [searchText]
  );

  const handleSearchTextChange = (value: string) => {
    setSearchText(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => commitSearch(value), 500);
  };

  const toggleLevel = (level: string) => {
    setSelectedLevels((current) =>
      current.includes(level) ? current.filter((item) => item !== level) : [...current, level]
    );
    setPageFrom(0);
  };

  const handleFieldFilter = useCallback((key: string, value: unknown, op: FieldFilterOp) => {
    const valueString = String(value);
    setFieldFilters((current) => {
      const exists = current.find(
        (filter) => filter.key === key && filter.value === valueString && filter.op === op
      );
      if (exists) return current.filter((filter) => filter !== exists);
      return [...current, { key, value: valueString, op }];
    });
    setPageFrom(0);
  }, []);

  const handleReset = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setSearchText('');
    setAppliedSearch('');
    setSelectedLevels([]);
    setSelectedLogType('');
    setSelectedNodes([]);
    setPageFrom(0);
    setFieldFilters([]);
    onTimeRangeChange(DEFAULT_LOG_TIME_RANGE);
    onRefreshIntervalChange(0);
  };

  const restoreHandoffSnapshot = () => {
    if (!handoffSnapshot) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setLogs(handoffSnapshot.logs);
    setTotal(handoffSnapshot.total);
    setSearchText(handoffSnapshot.searchText);
    setAppliedSearch(handoffSnapshot.appliedSearch);
    setSelectedLevels(handoffSnapshot.selectedLevels);
    setSelectedLogType(handoffSnapshot.selectedLogType);
    setSelectedNodes(handoffSnapshot.selectedNodes);
    setPageFrom(handoffSnapshot.pageFrom);
    setFieldFilters(handoffSnapshot.fieldFilters);
    preservePageForTimeRangeRef.current =
      handoffSnapshot.timeRange.from !== timeRange.from ||
      handoffSnapshot.timeRange.to !== timeRange.to;
    onTimeRangeChange(handoffSnapshot.timeRange);
    onRefreshIntervalChange(handoffSnapshot.refreshInterval);
    setHandoffSnapshot(undefined);
  };

  if (!enabled) {
    return (
      <div className="flex min-h-[34rem] items-center justify-center rounded-md border bg-background p-6 text-center">
        <div className="max-w-xl space-y-3">
          <AlertTriangle className="mx-auto size-8 text-amber-500" />
          <h2 className="text-lg font-semibold">{t('logCenter.historicalUnavailableTitle')}</h2>
          <p className="text-sm leading-6 text-muted-foreground">
            {t('logCenter.historicalUnavailableDescription')}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-15rem)] min-h-[36rem] flex-col overflow-hidden rounded-md border bg-background">
      <LogToolbar
        nodeOptions={nodeOptions}
        selectedNodes={selectedNodes}
        onSelectedNodesChange={(values) => {
          setSelectedNodes(values);
          setPageFrom(0);
        }}
        searchText={searchText}
        onSearchTextChange={handleSearchTextChange}
        onSearchCommit={() => commitSearch()}
        selectedLevels={selectedLevels}
        onToggleLevel={toggleLevel}
        selectedLogType={selectedLogType}
        onSelectedLogTypeChange={(value) => {
          setSelectedLogType(value);
          setPageFrom(0);
        }}
        emptySelectionMeansAll
        onReset={handleReset}
        actionLabel={t('logCenter.query')}
        onAction={() => setRefreshKey((current) => current + 1)}
      />

      {handoffSnapshot && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-primary/5 px-4 py-2 text-sm">
          <span className="text-muted-foreground">{t('logCenter.linkedQueryNotice')}</span>
          <Button variant="ghost" size="sm" onClick={restoreHandoffSnapshot}>
            <RotateCcw className="size-4" />
            {t('logCenter.returnToPreviousQuery')}
          </Button>
        </div>
      )}

      <FilterChipBar
        filters={fieldFilters}
        clearLabel={t('logCenter.clearFilters')}
        onRemove={(index) => {
          setFieldFilters((current) => current.filter((_, filterIndex) => filterIndex !== index));
          setPageFrom(0);
        }}
        onClear={() => {
          setFieldFilters([]);
          setPageFrom(0);
        }}
      />
      <LogTable
        logs={logs || []}
        loading={loading}
        fieldFilters={fieldFilters}
        appliedSearch={appliedSearch}
        selectedLevels={selectedLevels}
        selectedLogType={selectedLogType}
        nodeField={nodeField}
        nodeRoles={historicalNodeRoles}
        onFieldFilter={handleFieldFilter}
        onViewRuntimeNode={onViewRuntimeNode}
      />
      <LogPagination total={total} pageFrom={pageFrom} onPageFromChange={setPageFrom} />
    </div>
  );
}
