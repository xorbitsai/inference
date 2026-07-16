'use client';

import { FC, useState } from 'react';
import { Settings2, Copy, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import request from '@/lib/request';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useI18n } from '@/contexts/i18n-context';
import { copyToClipboard } from '@/lib/utils';
import type { ModelEnvItem } from '@/types/services';
import type { CatalogModel } from './types';

interface EnvManagementDialogProps {
  modelDetail: CatalogModel;
  onEnvDelete: () => void;
}

const EnvManagementDialog: FC<EnvManagementDialogProps> = ({ modelDetail, onEnvDelete }) => {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [dataSource, setDataSource] = useState<ModelEnvItem[]>([]);
  const [pendingDeleteItem, setPendingDeleteItem] = useState<ModelEnvItem>();
  const [deletingLoading, setDeletingLoading] = useState(false);

  const handleDelete = (data: ModelEnvItem) => {
    setPendingDeleteItem(data);
  };

  const fetchEnv = async () => {
    const res = await request.get(`/v1/virtualenvs?model_name=${modelDetail.model_name}`);
    const nextDataSource = Array.isArray(res?.list) ? res.list : [];

    setDataSource(nextDataSource);
    return nextDataSource;
  };

  const handleConfirmDelete = async () => {
    if (!pendingDeleteItem) return;
    setDeletingLoading(true);

    try {
      const params = new URLSearchParams();

      params.set('model_name', modelDetail.model_name);
      params.set('model_engine', pendingDeleteItem.model_engine);
      params.set('python_version', pendingDeleteItem.python_version);
      params.set('worker_ip', pendingDeleteItem.actor_ip_address);

      await request.delete(`/v1/virtualenvs?${params.toString()}`);
      toast.success(t('common.deleteSuccess'));
      setPendingDeleteItem(undefined);

      const nextDataSource = await fetchEnv();

      if (!nextDataSource.length) {
        setOpen(false);
        onEnvDelete();
      }
    } finally {
      setDeletingLoading(false);
    }
  };

  const onOpenChange = (open: boolean) => {
    setOpen(open);
    if (open) fetchEnv();
    if (!open) setPendingDeleteItem(undefined);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <button
          type="button"
          className="inline-flex shrink-0 items-center gap-1 rounded-full border border-sky-500/30 bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-700 transition-colors hover:bg-sky-500/20"
        >
          <Settings2 className="size-3.5" />
          {t('launchModel.manageVirtualEnvironments')}
        </button>
      </DialogTrigger>
      <DialogContent className="!max-w-5xl">
        <DialogHeader>
          <DialogTitle>{modelDetail.model_name}</DialogTitle>
        </DialogHeader>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t('launchModel.modelName')}</TableHead>
                <TableHead>{t('launchModel.envPath')}</TableHead>
                <TableHead>{t('launchModel.pythonVersion')}</TableHead>
                <TableHead>{t('launchModel.ipAddress')}</TableHead>
                <TableHead>{t('common.operation')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {dataSource.length ? (
                dataSource.map((item, index) => (
                  <TableRow key={`${item.model_name}_${index}`}>
                    <TableCell>{item.model_name}</TableCell>
                    <TableCell className="max-w-[220px]">
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate">{item?.path}</span>
                        <Copy
                          className="size-4 shrink-0 cursor-pointer text-muted-foreground hover:text-foreground"
                          onClick={() => copyToClipboard(item?.path)}
                        />
                      </div>
                    </TableCell>

                    <TableCell>{item.python_version}</TableCell>
                    <TableCell>{item.actor_ip_address}</TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="group hover:bg-destructive/10 rounded-full"
                        onClick={() => handleDelete(item)}
                      >
                        <Trash2 className="text-muted-foreground group-hover:text-destructive" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={7} className="h-40 text-center text-muted-foreground">
                    No virtual environments for now.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
        <ConfirmDialog
          isOpen={Boolean(pendingDeleteItem)}
          onOpenChange={(open) => {
            if (!open) setPendingDeleteItem(undefined);
          }}
          description={t('launchModel.confirmDeleteVirtualEnv')}
          onConfirm={handleConfirmDelete}
          isLoading={Boolean(deletingLoading)}
        />
      </DialogContent>
    </Dialog>
  );
};
export default EnvManagementDialog;
