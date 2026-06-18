'use client';

import { ChevronDown, Loader2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
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
import request from '@/lib/request';
import { cn } from '@/lib/utils';

import { FilterChipBar } from './filter-chip-bar';
import { LogDetail } from './log-detail';
import type { FieldFilter, FieldFilterOp, LogContextResponse, LogRow } from './types';
import { filterRowsByFields, formatLogTime, HighlightText } from './utils';

interface ContextDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  anchorRow: LogRow;
  nodeField: string;
}

export function ContextDialog({ open, onOpenChange, anchorRow, nodeField }: ContextDialogProps) {
  const { t } = useI18n();
  const [currentAnchor, setCurrentAnchor] = useState(anchorRow);
  const [olderSize, setOlderSize] = useState(5);
  const [newerSize, setNewerSize] = useState(5);
  const [olderLoadCount, setOlderLoadCount] = useState(5);
  const [newerLoadCount, setNewerLoadCount] = useState(5);
  const [older, setOlder] = useState<LogRow[]>([]);
  const [newer, setNewer] = useState<LogRow[]>([]);
  const [hasMoreOlder, setHasMoreOlder] = useState(false);
  const [hasMoreNewer, setHasMoreNewer] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<'notFound' | 'fetchError' | null>(null);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [localFieldFilters, setLocalFieldFilters] = useState<FieldFilter[]>([]);
  const anchorRef = useRef<HTMLTableRowElement | null>(null);
  const scrolledAnchorRef = useRef<string | null>(null);

  useEffect(() => {
    if (open) setCurrentAnchor(anchorRow);
  }, [open, anchorRow]);

  const resetState = useCallback(() => {
    setOlderSize(5);
    setNewerSize(5);
    setOlderLoadCount(5);
    setNewerLoadCount(5);
    setOlder([]);
    setNewer([]);
    setError(null);
    setExpandedRow(null);
    setLocalFieldFilters([]);
    scrolledAnchorRef.current = null;
  }, []);

  const handleClose = () => {
    resetState();
    onOpenChange(false);
  };

  const timestamp = currentAnchor?.['@timestamp'];

  useEffect(() => {
    if (!open || !timestamp) return;

    const fetchContext = async () => {
      setLoading(true);
      setError(null);

      try {
        const params = new URLSearchParams();
        params.set('timestamp', timestamp);
        params.set('size', String(Math.max(olderSize, newerSize)));
        if (currentAnchor.node) params.set('node', String(currentAnchor.node));
        if (nodeField && nodeField !== 'node') params.set('node_field', nodeField);

        const data = await request.get<LogContextResponse>(
          `/v1/cluster/logs/context?${params.toString()}`
        );

        setOlder((data.older || []).slice(0, olderSize));
        setNewer((data.newer || []).slice(0, newerSize));
        setHasMoreOlder(data.has_more_older || false);
        setHasMoreNewer(data.has_more_newer || false);
      } catch (requestError: unknown) {
        const status = (requestError as { response?: { status?: number } })?.response?.status;
        setError(status === 404 ? 'notFound' : 'fetchError');
      } finally {
        setLoading(false);
      }
    };

    fetchContext();
  }, [open, timestamp, olderSize, newerSize, currentAnchor, nodeField]);

  useEffect(() => {
    if (!loading && timestamp && scrolledAnchorRef.current !== timestamp && anchorRef.current) {
      anchorRef.current.scrollIntoView({ block: 'center', behavior: 'smooth' });
      scrolledAnchorRef.current = timestamp;
    }
  }, [loading, timestamp]);

  const handleLocalFilter = useCallback((key: string, value: unknown, op: FieldFilterOp) => {
    const valueString = String(value);

    setLocalFieldFilters((current) => {
      const exists = current.find(
        (filter) => filter.key === key && filter.value === valueString && filter.op === op
      );

      if (exists) return current.filter((filter) => filter !== exists);
      return [...current, { key, value: valueString, op }];
    });
  }, []);

  const handleSwitchAnchor = (row: LogRow) => {
    resetState();
    setCurrentAnchor(row);
  };

  const clampLoadCount = (value: number | string) => Math.max(1, Math.min(500, Number(value) || 1));

  const renderLoadBar = (direction: 'newer' | 'older') => {
    const isNewer = direction === 'newer';
    const hasMore = isNewer ? hasMoreNewer : hasMoreOlder;
    const count = isNewer ? newerLoadCount : olderLoadCount;
    const setCount = isNewer ? setNewerLoadCount : setOlderLoadCount;
    const setSize = isNewer ? setNewerSize : setOlderSize;

    if (!hasMore) return null;

    return (
      <TableRow>
        <TableCell colSpan={5}>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="text-xs text-primary hover:text-primary"
              onClick={() => setSize((size) => size + clampLoadCount(count))}
            >
              {t(isNewer ? 'logCenter.detail.loadNewer' : 'logCenter.detail.loadOlder')}
            </Button>
            <Input
              type="number"
              min={1}
              max={500}
              value={count}
              onChange={(event) => setCount(clampLoadCount(event.target.value))}
              onBlur={(event) => setCount(clampLoadCount(event.target.value))}
              onKeyDown={(event) => {
                if (event.key === 'Enter') setSize((size) => size + clampLoadCount(count));
              }}
              className="h-8 w-16 text-center"
            />
            <span className="text-xs text-muted-foreground">
              {t(isNewer ? 'logCenter.detail.newerDocs' : 'logCenter.detail.olderDocs')}
            </span>
          </div>
        </TableCell>
      </TableRow>
    );
  };

  const renderContextRow = (row: LogRow, rowKey: string, isAnchor: boolean) => {
    const isExpanded = expandedRow === rowKey;

    return (
      <>
        <TableRow
          key={`${rowKey}-row`}
          ref={isAnchor ? anchorRef : undefined}
          className={cn(
            'cursor-pointer',
            isAnchor && 'bg-primary/10 hover:bg-primary/10',
            isExpanded && '[&>td]:border-b-0'
          )}
          onClick={() => setExpandedRow(isExpanded ? null : rowKey)}
        >
          <TableCell className="w-8">
            <ChevronDown
              className={cn('size-4 transition-transform', isExpanded && 'rotate-180')}
            />
          </TableCell>
          <TableCell className="w-40 whitespace-nowrap text-xs">
            {formatLogTime(row['@timestamp'])}
          </TableCell>
          <TableCell className="w-24 text-xs">
            <span
              className={cn(
                'font-semibold',
                LOG_LEVEL_TEXT_CLASSES[String(row.level)] || 'text-foreground'
              )}
            >
              {String(row.level || '')}
            </span>
          </TableCell>
          <TableCell className="w-44 text-xs">{String(row.node || '')}</TableCell>
          <TableCell className="max-w-0 truncate text-xs">
            <HighlightText text={row.message || ''} />
          </TableCell>
        </TableRow>
        <TableRow key={`${rowKey}-detail`}>
          <TableCell colSpan={5} className="p-0">
            {isExpanded && (
              <LogDetail
                row={row}
                onFilter={handleLocalFilter}
                fieldFilters={localFieldFilters}
                appliedSearch=""
                selectedLevels={[]}
                selectedLogType=""
                nodeField={nodeField}
                onViewContext={handleSwitchAnchor}
              />
            )}
          </TableCell>
        </TableRow>
      </>
    );
  };

  const newerDesc = filterRowsByFields([...newer].reverse(), localFieldFilters);
  const filteredOlder = filterRowsByFields(older, localFieldFilters);

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => (nextOpen ? onOpenChange(true) : handleClose())}
    >
      <DialogContent className="!max-w-3xl gap-0 p-0" maskClosable>
        <DialogHeader className="border-b px-5 py-4">
          <DialogTitle>{t('logCenter.detail.contextTitle')}</DialogTitle>
        </DialogHeader>
        <div className="min-h-0">
          {loading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="size-7 animate-spin text-muted-foreground" />
            </div>
          )}
          {error === 'notFound' && (
            <div className="py-16 text-center text-muted-foreground">
              {t('logCenter.detail.contextError')}
            </div>
          )}
          {error === 'fetchError' && (
            <div className="py-16 text-center text-destructive">
              {t('logCenter.detail.contextFetchError')}
            </div>
          )}
          {!loading && !error && (
            <>
              <FilterChipBar
                filters={localFieldFilters}
                clearLabel={t('logCenter.clearFilters')}
                onRemove={(index) =>
                  setLocalFieldFilters((current) =>
                    current.filter((_, filterIndex) => filterIndex !== index)
                  )
                }
                onClear={() => setLocalFieldFilters([])}
              />
              <div className="max-h-[65vh] overflow-auto">
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
                    {renderLoadBar('newer')}
                    {newerDesc.map((row, index) => renderContextRow(row, `newer-${index}`, false))}
                    {currentAnchor && renderContextRow(currentAnchor, 'anchor', true)}
                    {filteredOlder.map((row, index) =>
                      renderContextRow(row, `older-${index}`, false)
                    )}
                    {renderLoadBar('older')}
                  </TableBody>
                </Table>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
