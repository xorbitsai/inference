import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

const STATUSES = new Set([
  'CREATING',
  'LOADING',
  'UPDATING',
  'READY',
  'ERROR',
  'TERMINATING',
  'TERMINATED',
]);

export function ModelStatusBadge({ status }: { status: string }) {
  const { t } = useI18n();
  return (
    <span
      className={cn(
        'mt-2 inline-flex rounded-md border px-2 py-0.5 text-xs',
        status === 'READY'
          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
          : status === 'ERROR'
            ? 'border-destructive/30 bg-destructive/10 text-destructive'
            : 'border-border bg-muted/40 text-muted-foreground'
      )}
    >
      {STATUSES.has(status) ? t(`runningModels.statuses.${status}`) : status}
    </span>
  );
}
