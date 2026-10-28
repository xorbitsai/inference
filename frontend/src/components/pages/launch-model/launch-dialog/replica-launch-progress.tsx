'use client';

import { LoaderCircle } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';
import type { LaunchProgressResponse, ReplicaItem } from '@/types/services';
import { normalizeProgress } from '../utils';
import DownloadProgressDetails from './download-progress-details';

function getLaunchStageKey(stage?: string, status?: string): string {
  if (status === 'TERMINATING' || status === 'TERMINATED')
    return `runningModels.statuses.${status}`;
  if (stage === 'downloading') return 'launchModel.stageDownloading';
  if (stage === 'waiting_for_dependencies') return 'launchModel.stageWaitingDependencies';
  if (stage === 'installing_dependencies') return 'launchModel.stageInstallingDependencies';
  if (stage === 'loading' || status === 'LOADING') return 'launchModel.stageLoading';
  return 'launchModel.stagePreparing';
}

export default function ReplicaLaunchProgress({
  replicaStatuses,
  progressDetails,
  constrainHeight = true,
}: {
  replicaStatuses: ReplicaItem[];
  progressDetails: LaunchProgressResponse | null;
  constrainHeight?: boolean;
}) {
  const { t } = useI18n();
  if (!replicaStatuses.length) {
    return <div className="text-sm text-muted-foreground">{t('launchModel.noReplicaStatus')}</div>;
  }

  const progressByReplicaId = new Map(
    (progressDetails?.replicas ?? []).map((replica) => [replica.replica_id, replica])
  );

  return (
    <div className="space-y-3 rounded-lg border bg-muted/20 p-4">
      <div className="text-sm font-medium">{t('launchModel.launchProgress')}</div>
      <div
        className={cn(
          'grid grid-cols-[repeat(auto-fit,minmax(min(100%,20rem),1fr))] gap-3 pr-1',
          constrainHeight && 'max-h-[min(40vh,24rem)] overflow-y-auto'
        )}
        aria-live="polite"
      >
        {replicaStatuses.map((replica) => {
          const replicaProgress = progressByReplicaId.get(replica.replica_id);
          const replicaReady = replica.status === 'READY';
          const replicaFailed = replica.status === 'ERROR';
          const replicaPercent = replicaReady
            ? 100
            : Math.min(normalizeProgress(replicaProgress?.progress), 99);
          const replicaStageKey = replicaReady
            ? 'launchModel.stageReady'
            : replicaFailed
              ? 'launchModel.stageFailed'
              : getLaunchStageKey(replicaProgress?.stage, replica.status);
          const statusClassName =
            replica.status === 'READY'
              ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
              : replica.status === 'ERROR'
                ? 'border-red-200 bg-red-50 text-red-700'
                : 'border-border bg-muted/40 text-muted-foreground';

          return (
            <article
              key={replica.replica_id}
              aria-label={`${t('launchModel.replica')} ${replica.replica_id}`}
              className="flex min-w-0 flex-col gap-3 rounded-md border bg-background/60 p-3"
            >
              <div className="flex min-w-0 items-start justify-between gap-3">
                <div className="flex min-w-0 flex-col gap-1 text-xs">
                  <span className="font-semibold">
                    {t('launchModel.replica')}&nbsp;{replica.replica_id}
                  </span>
                  <span
                    className="break-all font-mono text-muted-foreground"
                    title={replica.worker_address || '-'}
                  >
                    {replica.worker_address || '-'}
                  </span>
                  {replica.replica_uid && (
                    <span className="truncate text-muted-foreground" title={replica.replica_uid}>
                      {replica.replica_uid}
                    </span>
                  )}
                  <span className="text-muted-foreground">
                    {t('launchModel.GPUIdx')}:{' '}
                    {replica.gpu_idx && replica.gpu_idx.length > 0
                      ? replica.gpu_idx.join(', ')
                      : 'auto'}
                  </span>
                </div>
                <div
                  className={cn(
                    'shrink-0 rounded-md border px-2 py-1 text-xs font-medium leading-none',
                    statusClassName
                  )}
                >
                  {replica.status}
                </div>
              </div>
              <div className="space-y-1.5 text-xs text-muted-foreground">
                <div className="flex items-center justify-between gap-2">
                  <span className="flex min-w-0 items-center gap-1.5">
                    {!replicaReady &&
                      !replicaFailed &&
                      !['TERMINATING', 'TERMINATED'].includes(replica.status) && (
                        <LoaderCircle
                          className="size-3.5 shrink-0 animate-spin text-primary"
                          aria-hidden="true"
                        />
                      )}
                    <span>{t(replicaStageKey)}</span>
                  </span>
                  {(replicaProgress || replicaReady) && (
                    <span className="shrink-0 tabular-nums">{Math.round(replicaPercent)}%</span>
                  )}
                </div>
                {(replicaProgress || replicaReady) && (
                  <Progress
                    value={replicaPercent}
                    className="h-1.5"
                    aria-label={`${t('launchModel.replica')} ${replica.replica_id}: ${t(replicaStageKey)}`}
                  />
                )}
              </div>
              {replica.error_message && (
                <p className="whitespace-pre-wrap break-words text-xs text-destructive">
                  {replica.error_message}
                </p>
              )}
              {(replicaProgress?.stage === 'downloading' ||
                Boolean(replicaProgress?.download_files?.length)) && (
                <DownloadProgressDetails files={replicaProgress?.download_files ?? []} compact />
              )}
              {(replicaProgress?.stage === 'installing_dependencies' ||
                replicaProgress?.stage === 'loading') &&
                typeof replicaProgress.dependency_install_total === 'number' &&
                replicaProgress.dependency_install_total > 0 &&
                typeof replicaProgress.dependency_install_completed === 'number' && (
                  <div className="space-y-1.5 text-xs text-muted-foreground">
                    <div className="tabular-nums">
                      {t('launchModel.dependencyInstallCount', {
                        completed: replicaProgress.dependency_install_completed,
                        total: replicaProgress.dependency_install_total,
                      })}
                    </div>
                    {Boolean(replicaProgress.dependency_install_plan?.length) && (
                      <details>
                        <summary className="cursor-pointer">
                          {t('launchModel.dependencyInstallPlan')}
                        </summary>
                        <ul className="mt-1 max-h-32 space-y-0.5 overflow-y-auto pl-4 font-mono">
                          {replicaProgress.dependency_install_plan?.map((packageSpec) => (
                            <li key={packageSpec} className="break-all">
                              {packageSpec}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>
                )}
              {replicaProgress?.stage === 'loading' &&
                (replicaProgress.dependency_install_status === 'skipped' ||
                  replicaProgress.dependency_install_total === 0) && (
                  <div className="text-xs text-muted-foreground">
                    {t('launchModel.dependencyInstallSkipped')}
                  </div>
                )}
            </article>
          );
        })}
      </div>
    </div>
  );
}
