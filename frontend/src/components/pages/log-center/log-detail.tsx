'use client';

import { Check, Copy, FileJson, FileText, GitBranch, MinusCircle, PlusCircle } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { LOG_FONT_SIZE_CLASS } from '@/constants/logs';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import { cn, copyToClipboard } from '@/lib/utils';

import { getCorrelationId } from './correlation-utils';
import { CorrelatedDialog } from './correlated-dialog';
import { RequestBodyDialog } from './request-body-dialog';
import type { FieldFilter, FieldFilterOp, LogRow } from './types';
import { FieldTypeIcon, formatFieldValue, HighlightText } from './utils';

interface LogDetailProps {
  row: LogRow;
  onFilter: (key: string, value: unknown, op: FieldFilterOp) => void;
  fieldFilters: FieldFilter[];
  appliedSearch: string;
  selectedLevels: string[];
  selectedLogType: string;
  nodeField: string;
  onViewContext: (row: LogRow) => void;
}

export function LogDetail({
  row,
  onFilter,
  fieldFilters,
  appliedSearch,
  selectedLevels,
  selectedLogType,
  onViewContext,
}: LogDetailProps) {
  const { t } = useI18n();
  const { clusterAuth, clusterUIConfig } = useGlobal();
  const menuAuth = useMenuAuth();
  const [copied, setCopied] = useState(false);
  const [requestIdCopied, setRequestIdCopied] = useState(false);
  const [correlatedOpen, setCorrelatedOpen] = useState(false);
  const [requestBodyOpen, setRequestBodyOpen] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const copyRequestIdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (copyRequestIdTimerRef.current) clearTimeout(copyRequestIdTimerRef.current);
    };
  }, []);

  const fields = useMemo(
    () =>
      Object.entries(row).filter(
        ([, value]) => value !== undefined && value !== null && value !== ''
      ),
    [row]
  );

  const activeFilterMap = useMemo(() => {
    const map = new Map<string, Set<string>>();

    fieldFilters.forEach((filter) => {
      if (filter.op !== '+') return;

      map.set(filter.key, new Set([...(map.get(filter.key) || []), filter.value]));
    });

    if (selectedLevels.length) {
      map.set('level', new Set([...(map.get('level') || []), ...selectedLevels]));
    }

    if (selectedLogType) {
      map.set('log_type', new Set([...(map.get('log_type') || []), selectedLogType]));
    }

    return map;
  }, [fieldFilters, selectedLevels, selectedLogType]);

  const handleCopyJson = () => {
    copyToClipboard(JSON.stringify(row, null, 2));
    setCopied(true);
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    copyTimerRef.current = setTimeout(() => setCopied(false), 1500);
  };

  const handleViewContext = () => {
    onViewContext(row);
  };

  const requestId = typeof row.request_id === 'string' ? row.request_id.trim() : '';
  const correlationId = getCorrelationId(row);
  const eventType = String(row.event_type || row.event || '');
  const canReadRequestBody = Boolean(
    requestId &&
    eventType === 'model_request_started' &&
    clusterAuth?.auth !== false &&
    clusterUIConfig?.auth_advanced &&
    menuAuth.canReadModelRequestBody
  );

  const handleCopyRequestId = () => {
    if (!correlationId) return;
    copyToClipboard(correlationId);
    setRequestIdCopied(true);
    if (copyRequestIdTimerRef.current) clearTimeout(copyRequestIdTimerRef.current);
    copyRequestIdTimerRef.current = setTimeout(() => setRequestIdCopied(false), 1500);
  };

  return (
    <>
      <div className="px-4 pb-4">
        <Tabs defaultValue="table" className="gap-3">
          <div className="flex items-center justify-between gap-3 mt-1">
            <TabsList className="h-9">
              <TabsTrigger value="table" className="px-3 text-xs">
                {t('logCenter.detail.tableTab')}
              </TabsTrigger>
              <TabsTrigger value="json" className="px-3 text-xs">
                {t('logCenter.detail.jsonTab')}
              </TabsTrigger>
            </TabsList>
            <div className="flex flex-wrap items-center justify-end gap-1">
              {correlationId && (
                <Button variant="ghost" size="sm" className="text-xs" onClick={handleCopyRequestId}>
                  {requestIdCopied ? <Check className="size-4" /> : <Copy className="size-4" />}
                  {t('logCenter.detail.copyRequestId')}
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-primary hover:text-primary"
                onClick={handleViewContext}
              >
                <FileText className="size-4" />
                {t('logCenter.detail.viewContext')}
              </Button>
              {correlationId && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs text-primary hover:text-primary"
                  onClick={() => setCorrelatedOpen(true)}
                >
                  <GitBranch className="size-4" />
                  {t('logCenter.detail.viewCorrelated')}
                </Button>
              )}
              {canReadRequestBody && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs text-primary hover:text-primary"
                  onClick={() => setRequestBodyOpen(true)}
                >
                  <FileJson className="size-4" />
                  {t('logCenter.detail.viewRequestBody')}
                </Button>
              )}
            </div>
          </div>
          <TabsContent value="table">
            <Table size="small">
              <TableHeader>
                <TableRow>
                  <TableHead className="w-24">{t('logCenter.detail.action')}</TableHead>
                  <TableHead className="w-52">{t('logCenter.detail.field')}</TableHead>
                  <TableHead>{t('logCenter.detail.value')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {fields.map(([key, value]) => {
                  const isFilterMatch =
                    activeFilterMap.has(key) && activeFilterMap.get(key)?.has(String(value));
                  const valueKeywords = [
                    ...(appliedSearch ? [appliedSearch] : []),
                    ...(isFilterMatch ? [String(value)] : []),
                  ];

                  return (
                    <TableRow
                      key={key}
                      className={cn(isFilterMatch && 'bg-amber-50 dark:bg-amber-950/20')}
                    >
                      <TableCell className="whitespace-nowrap">
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-7"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onFilter(key, value, '+');
                                }}
                              >
                                <PlusCircle className="size-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>{t('logCenter.detail.filterFor')}</TooltipContent>
                          </Tooltip>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-7"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onFilter(key, value, '-');
                                }}
                              >
                                <MinusCircle className="size-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>{t('logCenter.detail.filterOut')}</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </TableCell>
                      <TableCell className={LOG_FONT_SIZE_CLASS}>
                        <span className="flex items-center gap-1.5">
                          <FieldTypeIcon fieldKey={key} value={value} />
                          <HighlightText text={key} keywords={appliedSearch} />
                        </span>
                      </TableCell>
                      <TableCell className={cn('break-all font-mono', LOG_FONT_SIZE_CLASS)}>
                        <HighlightText text={formatFieldValue(value)} keywords={valueKeywords} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TabsContent>
          <TabsContent value="json">
            <div className="relative max-h-96 overflow-auto rounded-md bg-muted p-3">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute right-2 top-2 size-8 bg-background/70"
                      onClick={handleCopyJson}
                    >
                      {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {copied ? t('logCenter.detail.copied') : t('logCenter.detail.copyJson')}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <pre className="m-0 whitespace-pre-wrap break-all pr-10 font-mono text-xs">
                {JSON.stringify(row, null, 2)}
              </pre>
            </div>
          </TabsContent>
        </Tabs>
      </div>
      <CorrelatedDialog
        requestId={correlationId}
        anchorTimestamp={row['@timestamp']}
        open={correlatedOpen}
        onOpenChange={setCorrelatedOpen}
      />
      <RequestBodyDialog
        requestId={requestId}
        open={requestBodyOpen}
        onOpenChange={setRequestBodyOpen}
      />
    </>
  );
}
