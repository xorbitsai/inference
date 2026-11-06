'use client';

import { Input } from '@/components/ui/input';
import { Select, type SelectOption } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';

export interface ReplicaConfigRow {
  replica_uid?: string;
  role?: 'hybrid' | 'prefill' | 'decode';
  worker_ip: string;
  gpu_idx: string;
  n_gpu?: number | 'auto';
}

interface ReplicaPlacementRowsProps {
  rows: ReplicaConfigRow[];
  workerOptions: SelectOption<string>[];
  onPatch: (index: number, patch: Partial<ReplicaConfigRow>) => void;
  disabled?: boolean;
  showRoles?: boolean;
  showSequence?: boolean;
  showGpu?: boolean;
}

/** Controlled editor shared by initial deployment and replica scale-up. */
export default function ReplicaPlacementRows({
  rows,
  workerOptions,
  onPatch,
  disabled = false,
  showRoles = false,
  showSequence = false,
  showGpu = true,
}: ReplicaPlacementRowsProps) {
  const { t } = useI18n();
  if (rows.length === 0) {
    return (
      <div className="w-full rounded-md border border-dashed border-border/70 p-3 text-xs text-muted-foreground">
        {t('launchModel.replicaPlacementEmpty')}
      </div>
    );
  }

  return (
    <div className="w-full space-y-2">
      {rows.map((row, index) => (
        <fieldset
          key={index}
          disabled={disabled}
          data-replica-index={index}
          className="min-w-0 rounded-md border border-border/70 p-3"
        >
          {showSequence && (
            <legend className="px-1 text-xs font-medium">
              {t('runningModels.addReplicaRow', { row: index + 1 })}
            </legend>
          )}
          <div
            className={`grid grid-cols-1 gap-2 ${showRoles ? 'sm:grid-cols-4' : 'sm:grid-cols-3'}`}
          >
            <label className="min-w-0 space-y-1 text-xs text-muted-foreground">
              <span>{t('launchModel.replicaUid')}</span>
              <Input
                data-field="replica_uid"
                value={row.replica_uid ?? ''}
                placeholder={t('launchModel.replicaUidPlaceholder')}
                disabled={disabled}
                onChange={(event) => onPatch(index, { replica_uid: event.target.value })}
              />
            </label>
            <div className="min-w-0 space-y-1 text-xs text-muted-foreground" data-field="worker_ip">
              <span>{t('launchModel.workerIp')}</span>
              <Select
                value={row.worker_ip}
                options={workerOptions}
                placeholder={t('launchModel.workerIpPlaceholder')}
                disabled={disabled}
                onChange={(value) => onPatch(index, { worker_ip: value ?? '' })}
              />
            </div>
            {showGpu && (
              <label className="min-w-0 space-y-1 text-xs text-muted-foreground">
                <span>{t('launchModel.GPUIdx')}</span>
                <Input
                  data-field="gpu_idx"
                  value={row.gpu_idx}
                  placeholder={t('launchModel.GPUIdxPlaceholder')}
                  disabled={disabled}
                  onChange={(event) =>
                    onPatch(index, { gpu_idx: event.target.value, n_gpu: undefined })
                  }
                />
              </label>
            )}
            {showRoles && (
              <div className="min-w-0 space-y-1 text-xs text-muted-foreground" data-field="role">
                <span>{t('launchModel.pdRole')}</span>
                <Select
                  value={row.role ?? 'hybrid'}
                  disabled={disabled}
                  options={[
                    { label: t('launchModel.pdHybrid'), value: 'hybrid' },
                    { label: 'Prefill', value: 'prefill' },
                    { label: 'Decode', value: 'decode' },
                  ]}
                  onChange={(value) => onPatch(index, { role: value as ReplicaConfigRow['role'] })}
                />
              </div>
            )}
          </div>
        </fieldset>
      ))}
    </div>
  );
}
