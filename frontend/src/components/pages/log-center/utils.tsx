import type { ReactNode } from 'react';
import { format } from 'date-fns';
import { CalendarDays, Hash, Type } from 'lucide-react';

import { LOG_LEVEL_BADGE_CLASSES, NODE_ROLE_BADGE_CLASSES } from '@/constants/logs';
import { cn } from '@/lib/utils';

import type { FieldFilter, LogNodeRole, LogRow } from './types';

export const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export const formatFieldValue = (value: unknown) => {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const firstNonEmptyLogValue = (...values: unknown[]) =>
  values.find((value) => value !== undefined && value !== null && value !== '');

export const getLogNodeName = (row: LogRow, nodeField = 'node') => {
  const configuredValue = firstNonEmptyLogValue(
    row[nodeField],
    nodeField === 'address.keyword' ? row.address : undefined,
    nodeField === 'node.keyword' ? row.node : undefined
  );
  const value = firstNonEmptyLogValue(row.address, row.node_name, configuredValue, row.node);
  return value === undefined ? '-' : String(value);
};

export const getLogNodeFilterValue = (row: LogRow, nodeField = 'node') => {
  const value = firstNonEmptyLogValue(
    row[nodeField],
    nodeField === 'address.keyword' ? row.address : undefined,
    nodeField === 'node.keyword' ? row.node : undefined
  );
  return value === undefined ? '' : String(value);
};

export const normalizeLogNodeRole = (value: unknown): LogNodeRole | '' => {
  const role = String(value || '').toLowerCase();
  if (role.includes('supervisor')) return 'supervisor';
  if (role.includes('worker')) return 'worker';
  if (role.includes('local')) return 'local';
  if (role.includes('unknown')) return 'unknown';
  return '';
};

export const getLogNodeRole = (row: LogRow): LogNodeRole | '' => {
  for (const value of [row.node_role, row.role, row.log_type]) {
    const role = normalizeLogNodeRole(value);
    if (role) return role;
  }
  return '';
};

export const inferNodeRoleFromName = (nodeName: string): LogNodeRole | '' => {
  const normalized = nodeName.toLowerCase();
  if (normalized.includes('supervisor')) return 'supervisor';
  if (normalized.includes('worker')) return 'worker';
  return '';
};

export const resolveHistoricalNodeRole = ({
  nodeName,
  historicalRole,
  resultRole,
  runtimeRole,
}: {
  nodeName: string;
  historicalRole?: unknown;
  resultRole?: unknown;
  runtimeRole?: unknown;
}): LogNodeRole =>
  normalizeLogNodeRole(historicalRole) ||
  normalizeLogNodeRole(resultRole) ||
  normalizeLogNodeRole(runtimeRole) ||
  inferNodeRoleFromName(nodeName) ||
  'unknown';

export function NodeRoleBadge({
  role,
  children,
  className,
}: {
  role: LogNodeRole | '';
  children: ReactNode;
  className?: string;
}) {
  const normalizedRole: LogNodeRole = role || 'unknown';
  return (
    <span
      className={cn(
        'inline-flex items-center rounded border px-1.5 py-0.5 font-medium',
        NODE_ROLE_BADGE_CLASSES[normalizedRole],
        className
      )}
    >
      {children}
    </span>
  );
}

export function LogLevelBadge({
  level,
  children,
  className,
}: {
  level?: string;
  children?: ReactNode;
  className?: string;
}) {
  const normalizedLevel = level === 'WARN' ? 'WARNING' : String(level || 'UNKNOWN').toUpperCase();
  return (
    <span
      className={cn(
        'inline-flex items-center rounded border px-1.5 py-0.5 font-mono font-semibold',
        LOG_LEVEL_BADGE_CLASSES[normalizedLevel] || LOG_LEVEL_BADGE_CLASSES.UNKNOWN,
        className
      )}
    >
      {children ?? normalizedLevel}
    </span>
  );
}

export const formatLogTime = (value?: string) => {
  if (!value) return '--';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return format(date, 'MM-dd HH:mm:ss.SSS');
};

export const formatLogTimeTitle = (value?: string) => {
  if (!value) return undefined;

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return `${format(date, 'yyyy-MM-dd HH:mm:ss.SSS xxx')} · ${value}`;
};

export const formatDateTime = (value: string) => {
  const timestamp = Number(value);
  const date = new Date(Number.isNaN(timestamp) ? value : timestamp);

  return Number.isNaN(date.getTime()) ? value : format(date, 'yyyy-MM-dd HH:mm');
};

export const toMilliseconds = (value: string) => String(new Date(value).getTime());

export const buildLogQueryParams = ({
  appliedSearch,
  selectedLevels,
  selectedLogType,
  selectedNode,
  nodeField,
  timeRange,
  pageFrom,
  fieldFilters,
  size,
}: {
  appliedSearch: string;
  selectedLevels: string[];
  selectedLogType: string;
  selectedNode: string;
  nodeField: string;
  timeRange: { from: string; to: string };
  pageFrom: number;
  fieldFilters: FieldFilter[];
  size: number;
}) => {
  const params = new URLSearchParams();

  if (appliedSearch) params.set('q', appliedSearch);
  if (selectedLevels.length) params.set('level', selectedLevels.join(','));
  if (selectedLogType) params.set('log_type', selectedLogType);
  if (selectedNode) params.set('node', selectedNode);
  if (nodeField !== 'node') params.set('node_field', nodeField);
  params.set('time_from', timeRange.from);
  params.set('time_to', timeRange.to);
  params.set('size', String(size));
  params.set('page_from', String(pageFrom));
  fieldFilters.forEach((filter) => {
    params.append('filters', `${filter.op}${filter.key}:${filter.value}`);
  });

  return params;
};

export const getLogSummary = (row: LogRow) => {
  if (row.message) return String(row.message);

  const eventType = String(row.event_type || '');
  const protocol = row.api_protocol
    ? String(row.api_protocol).charAt(0).toUpperCase() + String(row.api_protocol).slice(1)
    : '';
  const endpoint = String(row.endpoint || '');
  const model = row.model_uid ? ` model=${String(row.model_uid)}` : '';
  const stream = typeof row.stream === 'boolean' ? ` stream=${String(row.stream)}` : '';

  if (eventType === 'model_request_started') {
    return `[Request started] ${protocol} POST ${endpoint}${model}${stream}`.trim();
  }
  if (eventType === 'model_request_finished') {
    const elapsed = row.elapsed_ms === undefined ? '' : ` elapsed=${String(row.elapsed_ms)}ms`;
    return `[Request finished] ${endpoint} status=${String(row.status_code ?? '')}${elapsed}`.trim();
  }
  if (eventType === 'model_request_failed') {
    const errorType = row.error?.type ? ` error=${row.error.type}` : '';
    return `[Request failed] ${protocol} ${endpoint} status=${String(row.status_code ?? '')}${errorType}`.trim();
  }
  return eventType || endpoint || '-';
};

export const filterRowsByFields = (rows: LogRow[], filters: FieldFilter[]) => {
  if (!filters.length) return rows;

  const includeByKey = new Map<string, string[]>();
  const excludeFilters: FieldFilter[] = [];

  filters.forEach((filter) => {
    if (filter.op === '+') {
      includeByKey.set(filter.key, [...(includeByKey.get(filter.key) || []), filter.value]);
    } else {
      excludeFilters.push(filter);
    }
  });

  return rows.filter((row) => {
    for (const [key, values] of includeByKey.entries()) {
      if (!values.includes(String(row[key]))) return false;
    }

    return !excludeFilters.some((filter) => String(row[filter.key]) === filter.value);
  });
};

export function FieldTypeIcon({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  if (fieldKey === '@timestamp') {
    return <CalendarDays className="size-3.5 text-muted-foreground" />;
  }

  if (typeof value === 'number') {
    return <Hash className="size-3.5 text-muted-foreground" />;
  }

  return <Type className="size-3.5 text-muted-foreground" />;
}

export function HighlightText({
  text,
  keywords,
  className,
}: {
  text: unknown;
  keywords?: string | string[];
  className?: string;
}) {
  if (text === null || text === undefined) return null;

  const value = String(text);
  const validKeywords = (Array.isArray(keywords) ? keywords : [keywords]).filter(
    (keyword): keyword is string => Boolean(keyword && keyword.trim())
  );

  if (!validKeywords.length) return <>{value}</>;

  const pattern = validKeywords
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp)
    .join('|');
  const parts = value.split(new RegExp(`(${pattern})`, 'gi'));
  const lowerKeywords = new Set(validKeywords.map((keyword) => keyword.toLowerCase()));

  return (
    <>
      {parts.map((part, index) =>
        lowerKeywords.has(part.toLowerCase()) ? (
          <mark
            key={`${part}-${index}`}
            className={cn(
              'rounded-sm bg-amber-200 px-0 text-foreground dark:bg-amber-400/60',
              className
            )}
          >
            {part}
          </mark>
        ) : (
          part
        )
      )}
    </>
  );
}
