'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Select } from '@/components/ui/select';
import PageContainer from '@/components/ui/page-container';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { DEFAULT_LOG_TIME_RANGE, LOG_PAGE_SIZE, LOG_REFRESH_OPTIONS } from '@/constants/logs';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';

import { FilterChipBar } from './filter-chip-bar';
import { LogPagination } from './pagination';
import { LogTable } from './log-table';
import { LogToolbar } from './log-toolbar';
import type {
  FieldFilter,
  FieldFilterOp,
  LogNodesResponse,
  LogsResponse,
  TimeRangeValue,
} from './types';
import { buildLogQueryParams } from './utils';
import { TimeRangePicker } from './time-range-picker';

const LogCenter = () => {
  const { t } = useI18n();
  const { clusterUIConfig, globalReady } = useGlobal();
  const esEnabled = Boolean(clusterUIConfig?.es_enabled);
  const [logs, setLogs] = useState<LogsResponse['hits']>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [appliedSearch, setAppliedSearch] = useState('');
  const [selectedLevels, setSelectedLevels] = useState<string[]>([]);
  const [selectedLogType, setSelectedLogType] = useState('');
  const [selectedNode, setSelectedNode] = useState('');
  const [nodes, setNodes] = useState<string[]>([]);
  const [nodeField, setNodeField] = useState('node');
  const [pageFrom, setPageFrom] = useState(0);
  const [fieldFilters, setFieldFilters] = useState<FieldFilter[]>([]);
  const [timeRange, setTimeRange] = useState<TimeRangeValue>(DEFAULT_LOG_TIME_RANGE);
  const [refreshInterval, setRefreshInterval] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const refreshTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!esEnabled) return;

    const fetchNodes = async () => {
      try {
        const data = await request.get<LogNodesResponse>('/v1/cluster/logs/nodes');

        setNodes(Array.isArray(data.nodes) ? data.nodes : []);
        setNodeField(data.node_field || 'node');
      } catch {
        setNodes([]);
        setNodeField('node');
      }
    };

    fetchNodes();
  }, [esEnabled]);

  const fetchLogs = useCallback(async () => {
    if (!esEnabled) return;

    setLoading(true);

    try {
      const params = buildLogQueryParams({
        appliedSearch,
        selectedLevels,
        selectedLogType,
        selectedNode,
        nodeField,
        timeRange,
        pageFrom,
        fieldFilters,
        size: LOG_PAGE_SIZE,
      });
      const data = await request.get<LogsResponse>(`/v1/cluster/logs?${params.toString()}`);

      setLogs(data.hits || []);
      setTotal(data.total || 0);
    } catch {
      setLogs([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [
    appliedSearch,
    esEnabled,
    fieldFilters,
    nodeField,
    pageFrom,
    selectedLevels,
    selectedLogType,
    selectedNode,
    timeRange,
  ]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }

    if (refreshInterval > 0) {
      refreshTimerRef.current = setInterval(() => fetchLogs(), refreshInterval);
    }

    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current);
    };
  }, [fetchLogs, refreshInterval]);

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

  if (!globalReady) {
    return <PageContainer loading />;
  }
  if (!esEnabled) {
    return (
      <div className="flex h-[calc(100vh-8rem)] items-center justify-center text-center font-medium text-muted-foreground">
        {t('logCenter.notConfigured')}
      </div>
    );
  }

  return (
    <PageContainer
      title={t('menu.logCenter')}
      className="h-full gap-4"
      extraContent={
        <div className="flex gap-2">
          <TimeRangePicker
            value={timeRange}
            onChange={(value) => {
              setTimeRange(value);
              setPageFrom(0);
            }}
          />
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  aria-label={t('logCenter.refresh')}
                  onClick={fetchLogs}
                >
                  <RefreshCw className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{t('logCenter.refresh')}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Select
            value={refreshInterval}
            onChange={(value) => setRefreshInterval(Number(value || 0))}
            options={LOG_REFRESH_OPTIONS.map((option) => ({
              value: option.value,
              label: t(option.labelKey),
              prefix: <RefreshCw className="size-4" />,
            }))}
            allowClear={false}
            className="w-32"
          />
        </div>
      }
    >
      <div className="flex h-[calc(100vh-8rem)] min-h-[34rem] flex-col overflow-hidden rounded-md border bg-background">
        <LogToolbar
          nodes={nodes}
          selectedNode={selectedNode}
          onSelectedNodeChange={(value) => {
            setSelectedNode(value);
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
        />
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
          onFieldFilter={handleFieldFilter}
        />
        <LogPagination total={total} pageFrom={pageFrom} onPageFromChange={setPageFrom} />
      </div>
    </PageContainer>
  );
};

export default LogCenter;
