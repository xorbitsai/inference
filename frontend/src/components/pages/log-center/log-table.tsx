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
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

import { ContextDialog } from './context-dialog';
import { LogDetail } from './log-detail';
import type { FieldFilter, FieldFilterOp, LogNodeRole, LogRow } from './types';
import {
  formatLogTime,
  formatLogTimeTitle,
  getLogNodeFilterValue,
  getLogNodeName,
  getLogNodeRole,
  getLogSummary,
  HighlightText,
  resolveHistoricalNodeRole,
  LogLevelBadge,
  NodeRoleBadge,
} from './utils';

interface LogTableProps {
  logs: LogRow[];
  loading: boolean;
  fieldFilters: FieldFilter[];
  appliedSearch: string;
  selectedLevels: string[];
  selectedLogType: string;
  nodeField: string;
  nodeRoles: Record<string, LogNodeRole>;
  onFieldFilter: (key: string, value: unknown, op: FieldFilterOp) => void;
  onViewRuntimeNode?: (nodeName: string) => void;
}

export function LogTable({
  logs,
  loading,
  fieldFilters,
  appliedSearch,
  selectedLevels,
  selectedLogType,
  nodeField,
  nodeRoles,
  onFieldFilter,
  onViewRuntimeNode,
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
        .filter((filter) => filter.op === '+' && filter.key === nodeField)
        .map((filter) => filter.value),
      messages: [
        appliedSearch,
        ...fieldFilters
          .filter((filter) => filter.op === '+' && filter.key === 'message')
          .map((filter) => filter.value),
      ],
    };
  }, [appliedSearch, fieldFilters, nodeField, selectedLevels]);

  const roleLabel = (role: string) => {
    if (role === 'supervisor') return t('clusterInfo.supervisor');
    if (role === 'worker') return t('clusterInfo.worker');
    if (role === 'local') return t('logCenter.local');
    return t('logCenter.unknownNodeType');
  };

  return (
    <>
      <div className="min-h-0 flex-1 overflow-auto">
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
            {logs.map((row, index) => {
              const isExpanded = expandedRow === index;
              const nodeName = getLogNodeName(row, nodeField);
              const nodeFilterValue = getLogNodeFilterValue(row, nodeField);
              const role = resolveHistoricalNodeRole({
                nodeName,
                historicalRole: nodeRoles[nodeFilterValue] || nodeRoles[nodeName],
                resultRole: getLogNodeRole(row),
              });

              return (
                <Fragment key={`${row['@timestamp'] || index}-${index}`}>
                  <TableRow
                    className={cn('cursor-pointer', isExpanded && '[&>td]:border-b-0')}
                    onClick={() => setExpandedRow(isExpanded ? null : index)}
                  >
                    <TableCell>
                      <ChevronDown
                        className={cn('size-4 transition-transform', isExpanded && 'rotate-180')}
                      />
                    </TableCell>
                    <TableCell
                      className="whitespace-nowrap font-mono text-xs text-muted-foreground"
                      title={formatLogTimeTitle(row['@timestamp'])}
                    >
                      {formatLogTime(row['@timestamp'])}
                    </TableCell>
                    <TableCell className="whitespace-nowrap text-xs">
                      <NodeRoleBadge role={role}>{roleLabel(role)}</NodeRoleBadge>
                    </TableCell>
                    <TableCell
                      className="min-w-[140px] max-w-[220px] whitespace-normal break-words [overflow-wrap:anywhere] font-mono text-xs"
                      title={nodeName}
                    >
                      <HighlightText text={nodeName} keywords={highlightValues.nodes} />
                    </TableCell>
                    <TableCell className="whitespace-nowrap text-xs">
                      <LogLevelBadge level={String(row.level || '')}>
                        <HighlightText
                          text={row.level || 'UNKNOWN'}
                          keywords={highlightValues.levels}
                        />
                      </LogLevelBadge>
                    </TableCell>
                    <TableCell className="whitespace-pre-wrap break-words [overflow-wrap:anywhere] text-xs">
                      <HighlightText
                        text={getLogSummary(row)}
                        keywords={highlightValues.messages}
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={6} className="p-0">
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
                          onViewRuntimeNode={onViewRuntimeNode}
                        />
                      )}
                    </TableCell>
                  </TableRow>
                </Fragment>
              );
            })}
            {loading && (
              <TableRow>
                <TableCell colSpan={6}>
                  <div className="flex items-center justify-center gap-2 py-8 text-muted-foreground">
                    <Loader2 className="size-5 animate-spin" />
                    <span>{t('logCenter.loading')}</span>
                  </div>
                </TableCell>
              </TableRow>
            )}
            {!loading && logs.length === 0 && (
              <TableRow>
                <TableCell colSpan={6}>
                  <div className="py-10 text-center text-muted-foreground">
                    {t('logCenter.noHistoricalLogs')}
                  </div>
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
