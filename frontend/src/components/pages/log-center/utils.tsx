import { format } from 'date-fns';
import { CalendarDays, Hash, Type } from 'lucide-react';

import { cn } from '@/lib/utils';

import type { FieldFilter, LogRow } from './types';

export const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export const formatFieldValue = (value: unknown) => {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

export const formatLogTime = (value?: string) => {
  if (!value) return '';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString([], {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
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
            className={cn('rounded-sm bg-amber-200 px-0 text-foreground dark:bg-amber-400/60', className)}
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
