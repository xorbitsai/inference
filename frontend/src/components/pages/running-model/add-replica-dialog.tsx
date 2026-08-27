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
import { Select, type SelectOption } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import type { AddReplicaRequest, ModelEngine } from '@/types/services';
import { hasCompatibleEngineSpec } from './engine-compatibility.mjs';

interface WorkerOption {
  label: string;
  value: string;
  description?: string;
}

type DeviceValue = 'auto' | 'GPU' | 'CPU';

interface AddReplicaDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (body: AddReplicaRequest) => void;
  loading: boolean;
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
  const [replicaCount, setReplicaCount] = useState(1);
  const [device, setDevice] = useState<DeviceValue>(defaultDevice);
  const [selectedWorker, setSelectedWorker] = useState('');
  const [gpuIdx, setGpuIdx] = useState('');
  const [engineMap, setEngineMap] = useState<ModelEngine>({});

  useEffect(() => {
    if (!open) return;

    setSelectedEngine(modelEngine ?? '');
    setReplicaCount(1);
    setDevice(defaultDevice);
    setSelectedWorker('');
    setGpuIdx('');

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
  }, [defaultDevice, modelEngine, modelName, modelType, open]);

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

  const workerSelectOptions = [
    { label: t('runningModels.addReplicaAutoWorker'), value: '' },
    ...workerOptions,
  ];
  const deviceOptions: SelectOption<DeviceValue>[] = [
    { label: t('runningModels.addReplicaDeviceAuto'), value: 'auto' },
    { label: 'GPU', value: 'GPU' },
    { label: 'CPU', value: 'CPU' },
  ];

  const handleConfirm = () => {
    if (!Number.isInteger(replicaCount) || replicaCount < 1) {
      toast.error(t('runningModels.addReplicaInvalidCount'));
      return;
    }

    const trimmedGpuIdx = gpuIdx.trim();
    const hasGpuIdx = trimmedGpuIdx !== '';
    if (hasGpuIdx && !GPU_IDX_PATTERN.test(trimmedGpuIdx)) {
      toast.error(t('runningModels.addReplicaInvalidGpuIdx'));
      return;
    }
    if (hasGpuIdx && !selectedWorker) {
      toast.error(t('runningModels.addReplicaWorkerRequired'));
      return;
    }

    const gpuIndexes = hasGpuIdx
      ? trimmedGpuIdx.split(',').map((value) => Number.parseInt(value.trim(), 10))
      : [];
    if (gpuIndexes.length > 0 && gpuIndexes.length % replicaCount !== 0) {
      toast.error(t('runningModels.addReplicaGpuCountMismatch'));
      return;
    }

    const body: AddReplicaRequest = { replica: replicaCount };
    if (selectedEngine) {
      body.model_engine = selectedEngine;
    }

    if (selectedWorker) {
      const gpuCountPerReplica = gpuIndexes.length / replicaCount;
      body.replica_config = Array.from({ length: replicaCount }, (_, index) => {
        const assignedGpuIndexes = gpuIndexes.slice(
          index * gpuCountPerReplica,
          (index + 1) * gpuCountPerReplica
        );
        return {
          devices: [
            {
              worker_ip: selectedWorker,
              n_gpu:
                device === 'CPU' ? 0 : device === 'GPU' ? assignedGpuIndexes.length || 1 : 'auto',
              ...(assignedGpuIndexes.length > 0 ? { gpu_idx: assignedGpuIndexes } : {}),
            },
          ],
        };
      });
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
              disabled={loading || engineOptions.length === 0}
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">
              {t('runningModels.addReplicaCountLabel')}
              <span className="ml-1 text-destructive">*</span>
            </label>
            <Input
              type="number"
              min={1}
              step={1}
              value={replicaCount}
              onChange={(event) => setReplicaCount(Number(event.target.value))}
              disabled={loading}
            />
            <p className="text-xs text-muted-foreground">
              {t('runningModels.addReplicaCountPreview', {
                current: currentReplicaCount,
                total: currentReplicaCount + (Number.isFinite(replicaCount) ? replicaCount : 0),
              })}
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">
              {t('runningModels.addReplicaDeviceLabel')}
            </label>
            <Select
              value={device}
              options={deviceOptions}
              onChange={(value) => {
                setDevice(value ?? 'auto');
                if (value !== 'GPU') setGpuIdx('');
              }}
              allowClear={false}
              disabled={loading}
            />
          </div>

          {device === 'GPU' && (
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">
                {t('runningModels.addReplicaGpuIdxLabel')}
              </label>
              <Input
                value={gpuIdx}
                onChange={(event) => setGpuIdx(event.target.value)}
                placeholder={t('runningModels.addReplicaGpuIdxPlaceholder')}
                disabled={loading}
              />
            </div>
          )}

          <div className="flex flex-col gap-2 sm:col-span-2">
            <label className="text-sm font-medium">
              {t('runningModels.addReplicaWorkerLabel')}
            </label>
            <Select
              value={selectedWorker}
              options={workerSelectOptions}
              onChange={(value) => setSelectedWorker(value ?? '')}
              allowClear={false}
              showSearch
              disabled={loading}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={loading}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleConfirm} disabled={loading}>
            {loading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Plus className="mr-2 h-4 w-4" />
            )}
            {loading ? t('runningModels.addReplicaLoading') : t('runningModels.addReplica')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default AddReplicaDialog;
