'use client';

import { FC } from 'react';

import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import { useWatch } from '@/hooks/use-form';
import { useFormContext } from '@/components/ui/form';
import type { FormInstance } from '@/types/form';
import type { WorkerOption } from '../types';

interface ReplicaConfigRow {
  replica_uid?: string;
  role?: 'hybrid' | 'prefill' | 'decode';
  worker_ip: string;
  gpu_idx: string;
  n_gpu?: number | 'auto';
}

interface ReplicaPlacementConfigProps {
  form: FormInstance;
  /** worker options carrying the FULL `ip:port` address (see extractWorkerItems). */
  workerOptions: WorkerOption[];
  modelUid?: string;
}

/**
 * Per-replica placement editor for `replica_config`.
 *
 * Renders one row per replica (row count is kept in sync with the `replica`
 * field by the launch dialog). Each row pins the replica to a single worker
 * (full `ip:port`) and optional GPU indexes. `n_gpu` is derived from
 * `gpu_idx` length (or "auto" when none) so it always stays consistent.
 *
 * Bound to form field `replica_config` (array of rows); `replica_placement_mode`
 * gates visibility at the field-config level.
 */
const ReplicaPlacementConfig: FC<ReplicaPlacementConfigProps> = ({ form, workerOptions }) => {
  const { t } = useI18n();
  const { notifyUserChange } = useFormContext();
  const engine = useWatch('model_engine', form);
  const supportsPD = String(engine).toLowerCase() === 'vllm';
  const columns = supportsPD ? 'grid-cols-[1fr_2fr_1fr_1fr]' : 'grid-cols-[1fr_2fr_1fr]';
  const rows = (useWatch('replica_config', form) as ReplicaConfigRow[] | undefined) ?? [];

  const patchRow = (index: number, patch: Partial<ReplicaConfigRow>) => {
    const next = rows.map((row, i) => (i === index ? { ...row, ...patch } : row));
    form.setFieldsValue({ replica_config: next });
    notifyUserChange();
  };

  if (rows.length === 0) {
    return (
      <div className="w-full rounded-md border border-dashed border-border/70 p-3 text-xs text-muted-foreground">
        {t('launchModel.replicaPlacementEmpty')}
      </div>
    );
  }

  return (
    <div className="w-full space-y-2">
      <div className={`grid ${columns} gap-2 px-1 text-xs font-medium text-muted-foreground`}>
        <span>{t('launchModel.replicaUid')}</span>
        <span>{t('launchModel.workerIp')}</span>
        <span>{t('launchModel.GPUIdx')}</span>
        {supportsPD && <span>{t('launchModel.pdRole')}</span>}
      </div>
      {rows.map((row, index) => (
        <div
          key={index}
          className={`grid ${columns} items-center gap-2 rounded-md border border-border/70 p-2`}
        >
          <Input
            value={row?.replica_uid ?? ''}
            placeholder={t('launchModel.replicaUidPlaceholder')}
            onChange={(e) => patchRow(index, { replica_uid: e.target.value })}
          />
          <Select
            value={row?.worker_ip ?? ''}
            options={workerOptions}
            placeholder={t('launchModel.workerIpPlaceholder')}
            onChange={(value) => patchRow(index, { worker_ip: (value as string) ?? '' })}
          />
          <Input
            value={row?.gpu_idx ?? ''}
            placeholder={t('launchModel.GPUIdxPlaceholder')}
            onChange={(e) =>
              patchRow(index, {
                gpu_idx: e.target.value,
                // Once the user edits GPU indexes, derive n_gpu from the new
                // value instead of reusing metadata retained during import.
                n_gpu: undefined,
              })
            }
          />
          {supportsPD && (
            <Select
              value={row.role ?? 'hybrid'}
              options={[
                { label: t('launchModel.pdHybrid'), value: 'hybrid' },
                { label: 'Prefill', value: 'prefill' },
                { label: 'Decode', value: 'decode' },
              ]}
              onChange={(value) => patchRow(index, { role: value as ReplicaConfigRow['role'] })}
            />
          )}
        </div>
      ))}
    </div>
  );
};

export default ReplicaPlacementConfig;
