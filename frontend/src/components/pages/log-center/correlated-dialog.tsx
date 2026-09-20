'use client';

import { Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
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

import type { CorrelatedLogsResponse, LogRow } from './types';
import { formatLogTime, getLogSummary } from './utils';

export function CorrelatedDialog({
  requestId,
  anchorTimestamp,
  open,
  onOpenChange,
}: {
  requestId: string;
  anchorTimestamp?: unknown;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const [rows, setRows] = useState<LogRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const [truncated, setTruncated] = useState(false);

  useEffect(() => {
    if (!open || !requestId) return;
    let active = true;
    setLoading(true);
    setFailed(false);
    const params = new URLSearchParams({ request_id: requestId });
    const anchor = new Date(String(anchorTimestamp || ''));
    if (!Number.isNaN(anchor.getTime())) {
      const windowMs = 24 * 60 * 60 * 1000;
      params.set('time_from', new Date(anchor.getTime() - windowMs).toISOString());
      params.set('time_to', new Date(anchor.getTime() + windowMs).toISOString());
    }
    request
      .get<CorrelatedLogsResponse>(`/v1/cluster/logs/correlated?${params.toString()}`)
      .then((data) => {
        if (!active) return;
        setRows(data.hits || []);
        setTruncated(Boolean(data.truncated));
      })
      .catch(() => active && setFailed(true))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [anchorTimestamp, open, requestId]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="!max-w-5xl gap-0 p-0" maskClosable>
        <DialogHeader className="border-b px-5 py-4">
          <DialogTitle>{t('logCenter.detail.correlatedTitle')}</DialogTitle>
          <div className="break-all font-mono text-xs text-muted-foreground">{requestId}</div>
        </DialogHeader>
        {loading && <Loader2 className="mx-auto my-16 size-7 animate-spin" />}
        {!loading && failed && (
          <div className="py-16 text-center text-destructive">
            {t('logCenter.detail.correlatedFetchError')}
          </div>
        )}
        {!loading && !failed && (
          <div className="max-h-[70vh] overflow-auto">
            {truncated && (
              <div className="border-b bg-amber-50 px-4 py-2 text-xs text-amber-800 dark:bg-amber-950/20 dark:text-amber-200">
                {t('logCenter.detail.correlatedTruncated')}
              </div>
            )}
            <Table size="small">
              <TableHeader className="sticky top-0 z-10">
                <TableRow>
                  <TableHead>{t('logCenter.time')}</TableHead>
                  <TableHead>{t('logCenter.type')}</TableHead>
                  <TableHead>{t('logCenter.role')}</TableHead>
                  <TableHead>{t('logCenter.node')}</TableHead>
                  <TableHead>{t('logCenter.message')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row, index) => (
                  <TableRow key={`${row['@timestamp'] || ''}-${index}`}>
                    <TableCell className="whitespace-nowrap text-xs">
                      {formatLogTime(row['@timestamp'])}
                    </TableCell>
                    <TableCell className="text-xs">{String(row.log_type || '')}</TableCell>
                    <TableCell className="text-xs">{String(row.role || '')}</TableCell>
                    <TableCell className="text-xs">{String(row.node || '')}</TableCell>
                    <TableCell className="text-xs">{getLogSummary(row)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
