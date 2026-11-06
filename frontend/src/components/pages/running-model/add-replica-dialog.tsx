'use client';

import { FC, useEffect, useMemo, useState } from 'react';
import { Loader2, Plus } from 'lucide-react';
import { toast } from 'sonner';

import { GPU_IDX_PATTERN } from '@/components/pages/launch-model/utils';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { MultiSelect } from '@/components/ui/multi-select';
import { RadioGroup } from '@/components/ui/radio-group';
import { Select, type SelectOption } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import type { AddReplicaRequest, ModelEngine, ReplicaItem } from '@/types/services';
import ReplicaLaunchProgress from '../launch-model/launch-dialog/replica-launch-progress';
import type { AddReplicaOperation } from './use-add-replica';
import ReplicaPlacementRows, {
  type ReplicaConfigRow,
} from '../launch-model/launch-dialog/replica-placement-rows';
import {
  buildCustomReplicaConfigs,
  buildReplicaConfigs,
  filterWorkerOptions,
} from './add-replica-utils.mjs';
import { hasCompatibleEngineSpec } from './engine-compatibility.mjs';

interface WorkerOption {
  label: string;
  value: string;
  description?: string;
  gpuCount?: number;
}

type DeviceValue = 'auto' | 'GPU' | 'CPU';

interface AddReplicaDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (body: AddReplicaRequest) => void;
  loading: boolean;
  operation?: AddReplicaOperation | null;
  elapsedSeconds?: number;
  newReplicas?: ReplicaItem[];
  onRefresh?: () => void;
  onAcknowledge?: () => void;
  workerOptions: WorkerOption[];
  modelUid: string;
  modelName: string;
  modelType: string;
  modelEngine?: string;
  modelFormat?: string;
  modelSizeInBillions?: string | number;
  quantization?: string;
  currentReplicaCount: number;
  defaultDevice: DeviceValue;
}

const AddReplicaDialog: FC<AddReplicaDialogProps> = ({
  open,
  onOpenChange,
  onConfirm,
  loading,
  operation,
  elapsedSeconds = 0,
  newReplicas = [],
  onRefresh,
  onAcknowledge,
  workerOptions,
  modelUid,
  modelName,
  modelType,
  modelEngine,
  modelFormat,
  modelSizeInBillions,
  quantization,
  currentReplicaCount,
  defaultDevice,
}) => {
  const { t } = useI18n();
  const [selectedEngine, setSelectedEngine] = useState(modelEngine ?? '');
  const [replicaCount, setReplicaCount] = useState<number | ''>(1);
  const [device, setDevice] = useState<DeviceValue>(defaultDevice);
  const [selectedWorkers, setSelectedWorkers] = useState<string[]>([]);
  const [gpuIdx, setGpuIdx] = useState('');
  const [placementMode, setPlacementMode] = useState<'simple' | 'custom'>('simple');
  const [customRows, setCustomRows] = useState<ReplicaConfigRow[]>([]);
  const [placementError, setPlacementError] = useState<string>();
  const custom = placementMode === 'custom';
  const [engineMap, setEngineMap] = useState<ModelEngine>({});
  const unknown = operation?.phase === 'unknown';
  const locked = loading || unknown || operation?.phase === 'succeeded';
  const hasOperation = Boolean(operation);

  useEffect(() => {
    if (!open || hasOperation) return;

    setSelectedEngine(modelEngine ?? '');
    setReplicaCount(1);
    setDevice(defaultDevice);
    setSelectedWorkers([]);
    setGpuIdx('');
    setPlacementMode('simple');
    setCustomRows([{ worker_ip: '', gpu_idx: '' }]);
    setPlacementError(undefined);

    let active = true;
    const url =
      modelType.toLowerCase() === 'llm'
        ? `/v1/engines/${encodeURIComponent(modelName)}`
        : `/v1/engines/${encodeURIComponent(modelType)}/${encodeURIComponent(modelName)}`;
    request
      .get<ModelEngine>(url)
      .then((result) => {
        if (active) {
          setEngineMap(result || {});
        }
      })
      .catch(() => {
        if (active) {
          setEngineMap({});
        }
      });

    return () => {
      active = false;
    };
  }, [defaultDevice, hasOperation, modelEngine, modelName, modelType, open]);

  const engineOptions = useMemo<SelectOption<string>[]>(() => {
    const options: SelectOption<string>[] = Object.entries(engineMap)
      .filter(
        ([engine, metadata]) =>
          engine === modelEngine ||
          hasCompatibleEngineSpec(metadata, {
            modelFormat,
            modelSizeInBillions,
            quantization,
          })
      )
      .map(([engine, metadata]) => ({
        label: typeof metadata === 'string' ? `${engine} (${metadata})` : engine,
        value: engine,
        disabled: typeof metadata === 'string' && engine !== modelEngine,
      }));
    if (modelEngine && !options.some((option) => option.value === modelEngine)) {
      options.unshift({ label: modelEngine, value: modelEngine });
    }
    return options;
  }, [engineMap, modelEngine, modelFormat, modelSizeInBillions, quantization]);

  const deviceOptions: SelectOption<DeviceValue>[] = [
    { label: t('runningModels.addReplicaDeviceAuto'), value: 'auto' },
    { label: 'GPU', value: 'GPU' },
    { label: 'CPU', value: 'CPU' },
  ];
  const filteredWorkerOptions = useMemo(
    () => filterWorkerOptions(workerOptions, device, defaultDevice),
    [defaultDevice, device, workerOptions]
  );

  useEffect(() => {
    if (locked) return;
    const availableWorkers = new Set(filteredWorkerOptions.map((option) => option.value));
    setSelectedWorkers((workers) => {
      const nextWorkers = workers.filter((worker) => availableWorkers.has(worker));
      return nextWorkers.length === workers.length ? workers : nextWorkers;
    });
    setCustomRows((rows) => {
      let changed = false;
      const next = rows.map((row) => {
        if (!row.worker_ip || availableWorkers.has(row.worker_ip)) return row;
        changed = true;
        return { ...row, worker_ip: '', gpu_idx: '', n_gpu: undefined };
      });
      return changed ? next : rows;
    });
  }, [filteredWorkerOptions, locked]);

  useEffect(() => {
    if (locked) return;
    const count = Number(replicaCount);
    if (!Number.isInteger(count) || count < 1) return;
    setCustomRows((rows) =>
      rows.length === count
        ? rows
        : Array.from({ length: count }, (_, index) => rows[index] ?? { worker_ip: '', gpu_idx: '' })
    );
  }, [replicaCount, locked]);

  const patchCustomRow = (index: number, patch: Partial<ReplicaConfigRow>) => {
    if (locked) return;
    setPlacementError(undefined);
    setCustomRows((rows) =>
      rows.map((row, i) => {
        if (i !== index) return row;
        return {
          ...row,
          ...patch,
          ...(patch.worker_ip !== undefined && patch.worker_ip !== row.worker_ip
            ? { gpu_idx: '', n_gpu: undefined }
            : {}),
        };
      })
    );
  };

  const handleConfirm = () => {
    if (locked) return;
    const normalizedReplicaCount = Number(replicaCount);
    if (
      !Number.isInteger(normalizedReplicaCount) ||
      normalizedReplicaCount < (custom ? 1 : Math.max(1, selectedWorkers.length))
    ) {
      toast.error(t('runningModels.addReplicaInvalidCount'));
      return;
    }

    if (custom) {
      const result = buildCustomReplicaConfigs({
        replicaCount: normalizedReplicaCount,
        rows: customRows,
        workerAddresses: filteredWorkerOptions.map((worker) => worker.value),
        device,
      });
      if (result.error) {
        const message = t(`runningModels.addReplicaRowError_${result.error.code}`, {
          row: result.error.row,
        });
        setPlacementError(message);
        toast.error(message);
        return;
      }
      setPlacementError(undefined);
      onConfirm({
        replica: normalizedReplicaCount,
        ...(selectedEngine ? { model_engine: selectedEngine } : {}),
        replica_config: result.configs,
      });
      return;
    }

    const trimmedGpuIdx = gpuIdx.trim();
    const hasGpuIdx = trimmedGpuIdx !== '';
    if (hasGpuIdx && !GPU_IDX_PATTERN.test(trimmedGpuIdx)) {
      toast.error(t('runningModels.addReplicaInvalidGpuIdx'));
      return;
    }
    if (hasGpuIdx && selectedWorkers.length === 0) {
      toast.error(t('runningModels.addReplicaWorkerRequired'));
      return;
    }

    const gpuIndexes = hasGpuIdx
      ? trimmedGpuIdx.split(',').map((value) => Number.parseInt(value.trim(), 10))
      : [];
    if (gpuIndexes.length > 0 && gpuIndexes.length % normalizedReplicaCount !== 0) {
      toast.error(t('runningModels.addReplicaGpuCountMismatch'));
      return;
    }

    const body: AddReplicaRequest = { replica: normalizedReplicaCount };
    if (selectedEngine) {
      body.model_engine = selectedEngine;
    }

    if (selectedWorkers.length > 0) {
      const replicaConfigs = buildReplicaConfigs({
        replicaCount: normalizedReplicaCount,
        workerAddresses: selectedWorkers,
        device,
        gpuIndexes,
      });
      if (replicaConfigs) {
        body.replica_config = replicaConfigs;
      }
    } else if (device !== 'auto') {
      body.n_gpu = device === 'CPU' ? 0 : 1;
    }

    onConfirm(body);
  };

  const handleOpenChange = (nextOpen: boolean) => {
    if (!loading || nextOpen) {
      onOpenChange(nextOpen);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className="!max-w-[calc(100%-2rem)] sm:!max-w-3xl"
        maskClosable={!loading}
        showCloseButton={!loading}
      >
        <DialogHeader>
          <DialogTitle>{modelName || modelUid}</DialogTitle>
          <DialogDescription>{t('runningModels.addReplicaDescription')}</DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">{t('runningModels.addReplicaModelUid')}</label>
            <Input value={modelUid} disabled />
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">{t('runningModels.modelEngine')}</label>
            <Select
              value={selectedEngine}
              options={engineOptions}
              onChange={(value) => setSelectedEngine(value ?? '')}
              placeholder={modelEngine || t('runningModels.addReplicaEnginePlaceholder')}
              allowClear={false}
              disabled={locked || engineOptions.length === 0}
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">
              {t('runningModels.addReplicaCountLabel')}
              <span className="ml-1 text-destructive">*</span>
            </label>
            <Input
              type="number"
              min={custom ? 1 : Math.max(1, selectedWorkers.length)}
              aria-label={t('runningModels.addReplicaCountLabel')}
              step={1}
              value={replicaCount}
              onChange={(event) => {
                const value = event.target.value;
                setReplicaCount(value === '' ? '' : Number(value));
                setPlacementError(undefined);
              }}
              onBlur={() => {
                setReplicaCount((count) =>
                  Math.max(Number(count) || 0, custom ? 1 : selectedWorkers.length || 1)
                );
              }}
              disabled={locked}
            />
            <p className="text-xs text-muted-foreground">
              {t('runningModels.addReplicaCountPreview', {
                current: currentReplicaCount,
                total:
                  currentReplicaCount +
                  (typeof replicaCount === 'number' && Number.isFinite(replicaCount)
                    ? replicaCount
                    : 0),
              })}
            </p>
          </div>

          <div className="flex flex-col gap-2" data-field="device">
            <label className="text-sm font-medium">
              {t('runningModels.addReplicaDeviceLabel')}
            </label>
            <Select
              value={device}
              options={deviceOptions}
              onChange={(value) => {
                setDevice(value ?? 'auto');
                if (value !== 'GPU') setGpuIdx('');
                if (value === 'CPU') {
                  setCustomRows((rows) =>
                    rows.map((row) => ({ ...row, gpu_idx: '', n_gpu: undefined }))
                  );
                }
                setPlacementError(undefined);
              }}
              allowClear={false}
              disabled={locked}
            />
          </div>

          <div className="flex flex-col gap-2 sm:col-span-2">
            <span className="text-sm font-medium">{t('launchModel.replicaPlacementMode')}</span>
            <RadioGroup
              aria-label={t('launchModel.replicaPlacementMode')}
              value={placementMode}
              options={[
                { value: 'simple', label: t('runningModels.addReplicaSimplePlacement') },
                { value: 'custom', label: t('launchModel.placementCustom') },
              ]}
              onChange={(value) => {
                setPlacementMode(value === 'custom' ? 'custom' : 'simple');
                setPlacementError(undefined);
              }}
              disabled={locked}
            />
          </div>

          {!custom && device === 'GPU' && (
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">
                {t('runningModels.addReplicaGpuIdxLabel')}
              </label>
              <Input
                value={gpuIdx}
                onChange={(event) => setGpuIdx(event.target.value)}
                placeholder={t('runningModels.addReplicaGpuIdxPlaceholder')}
                disabled={locked}
              />
            </div>
          )}

          {!custom && (
            <div className="flex flex-col gap-2 sm:col-span-2">
              <label className="text-sm font-medium">
                {t('runningModels.addReplicaWorkerLabel')}
              </label>
              <MultiSelect
                value={selectedWorkers}
                options={filteredWorkerOptions}
                onChange={(workers) => {
                  setSelectedWorkers(workers);
                  setReplicaCount((count) => Math.max(Number(count) || 0, workers.length || 1));
                }}
                placeholder={t('runningModels.addReplicaAutoWorker')}
                searchable
                disabled={locked}
              />
              <p className="text-xs text-muted-foreground">
                {t('runningModels.addReplicaWorkerHint')}
              </p>
            </div>
          )}

          {custom && (
            <div className="space-y-3 sm:col-span-2">
              <p className="text-xs text-muted-foreground">
                {t('runningModels.addReplicaPlacementHint')}
              </p>
              <ReplicaPlacementRows
                rows={customRows}
                workerOptions={filteredWorkerOptions}
                onPatch={patchCustomRow}
                disabled={locked}
                showSequence
                showGpu={device !== 'CPU'}
              />
              {placementError && (
                <p role="alert" className="text-sm text-destructive">
                  {placementError}
                </p>
              )}
            </div>
          )}
        </div>

        {operation && (
          <div className="mt-6 space-y-3 border-t pt-4">
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-sm">
              <p role="status">{t(`runningModels.addReplicaState_${operation.phase}`)}</p>
              {(loading || unknown) && (
                <span className="tabular-nums text-muted-foreground" aria-live="off">
                  · {t('runningModels.addReplicaElapsed', { seconds: elapsedSeconds })}
                </span>
              )}
            </div>
            {(loading || unknown) && (
              <p className="text-xs text-muted-foreground">
                {t('runningModels.addReplicaWaitHint')}
              </p>
            )}
            {unknown && (
              <p className="text-sm text-amber-600" role="status">
                {t('runningModels.addReplicaUnknown')}
              </p>
            )}
            {operation.error && (
              <p className="break-words text-sm text-destructive" role="alert">
                {operation.error}
              </p>
            )}
            {operation.refreshFailed && (
              <p className="text-sm text-amber-600" role="status">
                {t(
                  operation.phase === 'succeeded'
                    ? 'runningModels.addReplicaSuccessRefreshFailed'
                    : 'runningModels.addReplicaRefreshFailed'
                )}
              </p>
            )}
            <ReplicaLaunchProgress
              replicaStatuses={newReplicas}
              progressDetails={operation.progress}
            />
            {(loading || unknown) && operation.count > newReplicas.length && (
              <p className="text-xs text-muted-foreground">
                {t('runningModels.addReplicaWaitingCount', {
                  count: operation.count - newReplicas.length,
                })}
              </p>
            )}
            <Button variant="outline" size="sm" onClick={onRefresh}>
              {t('common.refresh')}
            </Button>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={loading}>
            {t(operation && !loading ? 'common.close' : 'common.cancel')}
          </Button>
          {unknown ? (
            <Button variant="outline" onClick={onAcknowledge}>
              {t('runningModels.addReplicaAcknowledge')}
            </Button>
          ) : (
            operation?.phase !== 'succeeded' && (
              <Button onClick={handleConfirm} disabled={locked}>
                {loading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Plus className="mr-2 h-4 w-4" />
                )}
                {loading ? t('runningModels.addReplicaLoading') : t('runningModels.addReplica')}
              </Button>
            )
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default AddReplicaDialog;
