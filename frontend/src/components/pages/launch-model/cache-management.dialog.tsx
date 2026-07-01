'use client';

import { FC, useState } from 'react';
import { Box, Copy, Trash2 } from 'lucide-react';
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
import { copyText } from '@/lib/utils';
import type { ModelCachedItem } from '@/types/services';
import type { CatalogModel } from './types';

interface CacheManagementDialogProps {
  modelDetail: CatalogModel;
  onCacheDelete: () => void;
}

const CacheManagementDialog: FC<CacheManagementDialogProps> = ({
  modelDetail,
  onCacheDelete,
}) => {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [dataSource, setDataSource] = useState<ModelCachedItem[]>([]);
  const [pendingDeleteItem, setPendingDeleteItem] = useState<ModelCachedItem>();
  const [deletingVersion, setDeletingVersion] = useState('');

  const handleDelete = (data: ModelCachedItem) => {
    setPendingDeleteItem(data);
  };

  const fetchCache = async () => {
    const res = await request.get(`/v1/cache/models?model_name=${modelDetail.model_name}`);
    const nextDataSource = Array.isArray(res?.list) ? res.list : [];

    setDataSource(nextDataSource);
    return nextDataSource;
  };

  const handleConfirmDelete = async () => {
    if (!pendingDeleteItem?.model_version) return;

    setDeletingVersion(pendingDeleteItem.model_version);

    try {
      const params = new URLSearchParams();

      params.set('model_version', pendingDeleteItem.model_version);
      await request.delete(`/v1/cache/models?${params.toString()}`);
      toast.success(t('common.deleteSuccess'));
      setPendingDeleteItem(undefined);

      const nextDataSource = await fetchCache();

      if (!nextDataSource.length) {
        setOpen(false);
        onCacheDelete();
      }
    } finally {
      setDeletingVersion('');
    }
  };

  const onOpenChange = (open: boolean) => {
    setOpen(open);
    if (open) fetchCache();
    if (!open) setPendingDeleteItem(undefined);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <button
          type="button"
          className="inline-flex shrink-0 items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-700 transition-colors hover:bg-emerald-500/20"
        >
          <Box className="size-3.5" />
          {t('launchModel.manageCachedModels')}
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
                <TableHead>{t('launchModel.model_format')}</TableHead>
                <TableHead>{t('launchModel.model_size_in_billions')}</TableHead>
                <TableHead>{t('launchModel.quantizations')}</TableHead>
                <TableHead>{t('launchModel.real_path')}</TableHead>
                <TableHead>{t('launchModel.path')}</TableHead>
                <TableHead>{t('launchModel.ipAddress')}</TableHead>
                <TableHead>{t('common.operation')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {dataSource.length ? (
                dataSource.map((item) => (
                  <TableRow key={item.model_version}>
                    <TableCell>{item.model_format || '-'}</TableCell>
                    <TableCell>{item.model_size_in_billions || '-'}</TableCell>
                    <TableCell>{item.quantization || '-'}</TableCell>
                    <TableCell className="max-w-[220px]">
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate">{item?.real_path}</span>
                        <Copy
                          className="size-4 shrink-0 cursor-pointer text-muted-foreground hover:text-foreground"
                          onClick={() => copyText(item?.real_path)}
                        />
                      </div>
                    </TableCell>
                    <TableCell className="max-w-[220px]">
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate">{item?.path}</span>
                        <Copy
                          className="size-4 shrink-0 cursor-pointer text-muted-foreground hover:text-foreground"
                          onClick={() => copyText(item?.path)}
                        />
                      </div>
                    </TableCell>
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
                    No cache for now.
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
          description={t('launchModel.confirmDeleteCacheFiles')}
          onConfirm={handleConfirmDelete}
          isLoading={Boolean(deletingVersion)}
        />
      </DialogContent>
    </Dialog>
  );
};
export default CacheManagementDialog;
