'use client';

import { CheckCircle2, Download, LoaderCircle } from 'lucide-react';

import { CollapsiblePanel } from '@/components/ui/collapsible';
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useI18n } from '@/contexts/i18n-context';
import { formatFileSize } from '@/lib/utils';

export interface DownloadProgressFile {
  name: string;
  downloaded_bytes: number;
  total_bytes: number | null;
  progress: number | null;
  speed_bytes_per_second: number | null;
  elapsed_seconds: number;
  eta_seconds: number | null;
  status: string;
  replica_id?: number;
  replica_model_uid?: string;
}

interface DownloadProgressDetailsProps {
  files: DownloadProgressFile[];
}

function getProgressPercent(file: DownloadProgressFile): number {
  const progress = Number(file.progress);

  if (Number.isFinite(progress)) {
    const percent = progress <= 1 ? progress * 100 : progress;
    return Math.max(0, Math.min(100, percent));
  }

  if (file.total_bytes && file.total_bytes > 0) {
    return Math.max(0, Math.min(100, (file.downloaded_bytes / file.total_bytes) * 100));
  }

  return 0;
}

function formatSpeed(bytesPerSecond: number | null, completed: boolean): string {
  if (completed || bytesPerSecond === null || !Number.isFinite(bytesPerSecond)) {
    return '—';
  }

  return `${formatFileSize(Math.max(0, bytesPerSecond))}/s`;
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return '—';
  }

  const totalSeconds = Math.max(0, Math.ceil(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;

  return [hours, minutes, remainingSeconds]
    .map((value) => String(value).padStart(2, '0'))
    .join(':');
}

export default function DownloadProgressDetails({ files }: DownloadProgressDetailsProps) {
  const { t } = useI18n();
  const hasMultipleReplicas =
    new Set(files.map((file) => file.replica_model_uid).filter(Boolean)).size > 1;

  return (
    <CollapsiblePanel
      title={
        <span className="flex items-center gap-2">
          {t('launchModel.downloadDetails')}
          <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
            {files.length}
          </span>
        </span>
      }
      icon={<Download className="size-4" />}
      className="rounded-lg"
      contentClassName="p-0"
    >
      {files.length === 0 ? (
        <div className="flex min-h-24 items-center justify-center gap-2 px-4 py-6 text-sm text-muted-foreground">
          <LoaderCircle className="size-4 animate-spin text-primary" />
          {t('launchModel.waitingDownloadDetails')}
        </div>
      ) : (
        <div className="max-h-64 overflow-y-auto" aria-live="polite">
          <Table size="small" className="min-w-[760px] table-fixed">
            <TableHeader className="sticky top-0 z-10">
              <TableRow className="hover:bg-muted">
                <TableHead className="w-[34%]">{t('launchModel.downloadFileName')}</TableHead>
                <TableHead className="w-[25%]">{t('launchModel.downloadProgress')}</TableHead>
                <TableHead className="w-[15%]">{t('launchModel.downloadStatus')}</TableHead>
                <TableHead className="w-[13%]">{t('launchModel.downloadSpeed')}</TableHead>
                <TableHead className="w-[13%]">{t('launchModel.downloadEta')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {files.map((file, index) => {
                const progress = getProgressPercent(file);
                const completed = file.status === 'completed' || progress >= 100;

                return (
                  <TableRow key={`${file.replica_model_uid || 'model'}:${file.name}:${index}`}>
                    <TableCell className="min-w-0">
                      <div className="truncate font-medium" title={file.name || '-'}>
                        {file.name || '-'}
                      </div>
                      {hasMultipleReplicas && (
                        <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                          {t('launchModel.replica')} {file.replica_id}
                        </div>
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Progress value={progress} className="h-1.5 min-w-24 flex-1" />
                        <span className="w-9 shrink-0 text-right tabular-nums text-muted-foreground">
                          {Math.round(progress)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="flex items-center gap-1.5 whitespace-nowrap">
                        {completed ? (
                          <CheckCircle2 className="size-3.5 text-emerald-500" />
                        ) : (
                          <LoaderCircle className="size-3.5 animate-spin text-primary" />
                        )}
                        {t(
                          completed
                            ? 'launchModel.downloadCompleted'
                            : 'launchModel.downloadInProgress'
                        )}
                      </span>
                    </TableCell>
                    <TableCell className="whitespace-nowrap tabular-nums text-muted-foreground">
                      {formatSpeed(file.speed_bytes_per_second, completed)}
                    </TableCell>
                    <TableCell className="whitespace-nowrap tabular-nums text-muted-foreground">
                      {formatDuration(completed ? 0 : file.eta_seconds)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </CollapsiblePanel>
  );
}
