'use client';

import { FC } from 'react';

import { useWatch } from '@/hooks/use-form';
import { useFormContext } from '@/components/ui/form';
import type { FormInstance } from '@/types/form';
import type { WorkerOption } from '../types';
import ReplicaPlacementRows, { type ReplicaConfigRow } from './replica-placement-rows';

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
  const { notifyUserChange } = useFormContext();
  const engine = useWatch('model_engine', form);
  const supportsPD = ['vllm', 'sglang', 'mlx'].includes(String(engine).toLowerCase());
  const rows = (useWatch('replica_config', form) as ReplicaConfigRow[] | undefined) ?? [];

  const patchRow = (index: number, patch: Partial<ReplicaConfigRow>) => {
    const next = rows.map((row, i) => (i === index ? { ...row, ...patch } : row));
    form.setFieldsValue({ replica_config: next });
    notifyUserChange();
  };

  return (
    <ReplicaPlacementRows
      rows={rows}
      workerOptions={workerOptions}
      onPatch={patchRow}
      showRoles={supportsPD}
    />
  );
};

export default ReplicaPlacementConfig;
