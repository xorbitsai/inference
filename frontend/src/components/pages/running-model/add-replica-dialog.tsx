'use client';

import { FC, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';

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
import { Select } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import type { AddReplicaRequest } from '@/types/services';
import { GPU_IDX_PATTERN } from '@/components/pages/launch-model/utils';

interface WorkerOption {
  label: string;
  value: string;
}

interface AddReplicaDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (body: AddReplicaRequest) => void;
  loading: boolean;
  workerOptions: WorkerOption[];
  modelUid: string;
}

const AddReplicaDialog: FC<AddReplicaDialogProps> = ({
  open,
  onOpenChange,
  onConfirm,
  loading,
  workerOptions,
  modelUid,
}) => {
  const { t } = useI18n();
  const [selectedWorker, setSelectedWorker] = useState('');
  const [gpuIdx, setGpuIdx] = useState('');
  const [replicaUid, setReplicaUid] = useState('');

  const workerSelectOptions = [
    { label: t('runningModels.addReplicaAutoWorker'), value: '' },
    ...workerOptions,
  ];

  const handleConfirm = () => {
    const body: AddReplicaRequest = {};

    const hasWorker = selectedWorker !== '';
    const hasGpu = gpuIdx.trim() !== '';
    const hasUid = replicaUid.trim() !== '';

    if ((hasGpu || hasUid) && !hasWorker) {
      toast.error(t('runningModels.addReplicaWorkerRequired'));
      return;
    }
    if (hasGpu && !GPU_IDX_PATTERN.test(gpuIdx.trim())) {
      toast.error(t('runningModels.addReplicaInvalidGpuIdx'));
      return;
    }

    if (hasWorker) {
      body.replica_config = {
        devices: [],
      };
      if (hasUid) {
        body.replica_config.replica_uid = replicaUid.trim();
      }
      body.replica_config.devices = [
        {
          worker_ip: selectedWorker,
          ...(hasGpu
            ? {
                gpu_idx: gpuIdx
                  .split(',')
                  .map((s) => parseInt(s.trim(), 10))
                  .filter((n) => !isNaN(n)),
              }
            : {}),
        },
      ];
    }

    onConfirm(body);
  };

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      // Reset form on close
      setSelectedWorker('');
      setGpuIdx('');
      setReplicaUid('');
    }
    onOpenChange(nextOpen);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent maskClosable={!loading}>
        <DialogHeader>
          <DialogTitle>{t('runningModels.addReplicaTitle', { modelUid })}</DialogTitle>
          <DialogDescription>{t('runningModels.addReplicaWorkerLabel')}</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          {/* Worker selection */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">
              {t('runningModels.addReplicaWorkerLabel')}
            </label>
            <Select
              value={selectedWorker}
              options={workerSelectOptions}
              onChange={(value) => setSelectedWorker((value as string) ?? '')}
              disabled={loading}
            />
          </div>

          {/* GPU index */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">
              {t('runningModels.addReplicaGpuIdxLabel')}
            </label>
            <Input
              value={gpuIdx}
              onChange={(e) => setGpuIdx(e.target.value)}
              placeholder={t('runningModels.addReplicaGpuIdxPlaceholder')}
              disabled={loading}
            />
          </div>

          {/* Replica alias */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">{t('runningModels.addReplicaUidLabel')}</label>
            <Input
              value={replicaUid}
              onChange={(e) => setReplicaUid(e.target.value)}
              placeholder={t('runningModels.addReplicaUidPlaceholder')}
              disabled={loading}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={loading}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleConfirm} disabled={loading}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {loading ? t('runningModels.addReplicaLoading') : t('common.confirm')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default AddReplicaDialog;
