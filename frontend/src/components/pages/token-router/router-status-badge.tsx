import { Badge } from '@/components/ui/badge';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

const styles: Record<string, string> = {
  pending: 'border-blue-500/30 bg-blue-500/15 text-blue-700 dark:text-blue-300',
  assigned: 'border-blue-500/30 bg-blue-500/15 text-blue-700 dark:text-blue-300',
  starting: 'border-blue-500/30 bg-blue-500/15 text-blue-700 dark:text-blue-300',
  unavailable: 'border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-300',
  draining: 'border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-300',
  ready: 'border-emerald-500/30 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  syncing: 'border-blue-500/30 bg-blue-500/15 text-blue-700 dark:text-blue-300',
  degraded: 'border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-300',
  config_error: 'border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-300',
  heartbeat_timeout: 'border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-300',
  not_running: 'border-slate-500/30 bg-slate-500/15 text-slate-700 dark:text-slate-300',
  offline: 'border-slate-500/30 bg-slate-500/15 text-slate-700 dark:text-slate-300',
  error: 'border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-300',
  failed: 'border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-300',
  crash_loop: 'border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-300',
  port_conflict: 'border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-300',
  stopped: 'border-slate-500/30 bg-slate-500/15 text-slate-700 dark:text-slate-300',
  stale: 'border-slate-500/30 bg-slate-500/15 text-slate-700 dark:text-slate-300',
  disabled: 'border-zinc-500/30 bg-zinc-500/15 text-zinc-700 dark:text-zinc-300',
  draft: 'border-purple-500/30 bg-purple-500/15 text-purple-700 dark:text-purple-300',
};

const knownStatuses = new Set(Object.keys(styles));

export function RouterStatusBadge({ status }: { status: string }) {
  const { t } = useI18n();
  const label = knownStatuses.has(status) ? t(`tokenRouter.statuses.${status}`) : status;

  return <Badge className={cn(styles[status])}>{label}</Badge>;
}
