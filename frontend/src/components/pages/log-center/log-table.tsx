'use client';

import { ChevronDown, Loader2 } from 'lucide-react';
import { Fragment, useMemo, useState } from 'react';

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { LOG_LEVEL_TEXT_CLASSES } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

import { ContextDialog } from './context-dialog';
import { LogDetail } from './log-detail';
import type { FieldFilter, FieldFilterOp, LogRow } from './types';
import { formatLogTime, HighlightText } from './utils';

interface LogTableProps {
  logs: LogRow[];
  loading: boolean;
  fieldFilters: FieldFilter[];
  appliedSearch: string;
  selectedLevels: string[];
  selectedLogType: string;
  nodeField: string;
  onFieldFilter: (key: string, value: unknown, op: FieldFilterOp) => void;
}

export function LogTable({
  logs,
  loading,
  fieldFilters,
  appliedSearch,
  selectedLevels,
  selectedLogType,
  nodeField,
  onFieldFilter,
}: LogTableProps) {
  const { t } = useI18n();
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [contextAnchorRow, setContextAnchorRow] = useState<LogRow | null>(null);

  const highlightValues = useMemo(() => {
    return {
      levels: [
        ...fieldFilters
          .filter((filter) => filter.op === '+' && filter.key === 'level')
          .map((filter) => filter.value),
        ...selectedLevels,
      ],
      nodes: fieldFilters
        .filter((filter) => filter.op === '+' && filter.key === 'node')
        .map((filter) => filter.value),
      messages: [
        appliedSearch,
        ...fieldFilters
          .filter((filter) => filter.op === '+' && filter.key === 'message')
          .map((filter) => filter.value),
      ],
    };
  }, [appliedSearch, fieldFilters, selectedLevels]);

  return (
    <>
      <div className="min-h-0 flex-1 overflow-auto">
        <Table size="small">
          <TableHeader className="sticky top-0 z-10">
            <TableRow>
              <TableHead className="w-8" />
              <TableHead className="w-40">{t('logCenter.time')}</TableHead>
              <TableHead className="w-24">{t('logCenter.level')}</TableHead>
              <TableHead className="w-44">{t('logCenter.node')}</TableHead>
              <TableHead>{t('logCenter.message')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {logs.map((row, index) => {
              const isExpanded = expandedRow === index;

              return (
                <Fragment key={`${row['@timestamp'] || index}-${index}`}>
                  <TableRow
                    className={cn('cursor-pointer', isExpanded && '[&>td]:border-b-0')}
                    onClick={() => setExpandedRow(isExpanded ? null : index)}
                  >
                    <TableCell className="w-8">
                      <ChevronDown className={cn('size-4 transition-transform', isExpanded && 'rotate-180')} />
                    </TableCell>
                    <TableCell className="w-40 whitespace-nowrap text-xs">
                      {formatLogTime(row['@timestamp'])}
                    </TableCell>
                    <TableCell className="w-24 text-xs">
                      <span className={cn('font-semibold', LOG_LEVEL_TEXT_CLASSES[String(row.level)] || 'text-foreground')}>
                        <HighlightText text={row.level || ''} keywords={highlightValues.levels} />
                      </span>
                    </TableCell>
                    <TableCell className="w-44 text-xs">
                      <HighlightText text={row.node || ''} keywords={highlightValues.nodes} />
                    </TableCell>
                    <TableCell className="max-w-0 truncate text-xs">
                      <HighlightText text={row.message || ''} keywords={highlightValues.messages} />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={5} className="p-0">
                      {isExpanded && (
                        <LogDetail
                          row={row}
                          onFilter={onFieldFilter}
                          fieldFilters={fieldFilters}
                          appliedSearch={appliedSearch}
                          selectedLevels={selectedLevels}
                          selectedLogType={selectedLogType}
                          nodeField={nodeField}
                          onViewContext={setContextAnchorRow}
                        />
                      )}
                    </TableCell>
                  </TableRow>
                </Fragment>
              );
            })}
            {loading && (
              <TableRow>
                <TableCell colSpan={5}>
                  <div className="flex items-center justify-center gap-2 py-8 text-muted-foreground">
                    <Loader2 className="size-5 animate-spin" />
                    <span>{t('logCenter.loading')}</span>
                  </div>
                </TableCell>
              </TableRow>
            )}
            {!loading && logs.length === 0 && (
              <TableRow>
                <TableCell colSpan={5}>
                  <div className="py-10 text-center text-muted-foreground">{t('logCenter.noLogs')}</div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      {contextAnchorRow && (
        <ContextDialog
          open={Boolean(contextAnchorRow)}
          onOpenChange={(open) => {
            if (!open) setContextAnchorRow(null);
          }}
          anchorRow={contextAnchorRow}
          nodeField={nodeField}
        />
      )}
    </>
  );
}
