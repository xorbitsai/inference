'use client';

import { FC, useState, useMemo } from 'react';
import { SquarePen, Copy } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useI18n } from '@/contexts/i18n-context';
import type { CatalogModel, RequestModelType } from './types';
import { copyToClipboard } from '@/lib/utils';

interface EnvManagementDialogProps {
  model: CatalogModel;
  modelType: RequestModelType;
}
const CustomEditDialog: FC<EnvManagementDialogProps> = ({ model, modelType }) => {
  const { t } = useI18n();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const value = useMemo(() => JSON.stringify(model, null, 4), [model]);
  const handleEdit = () => {
    // set model detail
    sessionStorage.setItem('customJsonData', JSON.stringify(model));
    router.push(`/register-model/${modelType}/${model.model_name}`);
  };
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button
          type="button"
          aria-label="Delete model"
          className="rounded-full p-1 text-muted-foreground transition-colors hover:text-foreground"
        >
          <SquarePen className="size-5" />
        </button>
      </DialogTrigger>
      <DialogContent className="!max-w-2xl" showCloseButton={false}>
        <DialogHeader className="flex !flex-row items-center justify-between">
          <DialogTitle className="truncate">{model.model_name}</DialogTitle>
          <button
            className="shrink-0 p-1 text-muted-foreground transition-colors hover:text-foreground"
            onClick={() => copyToClipboard(value)}
          >
            <Copy className="size-4" />
          </button>
        </DialogHeader>
        <Textarea value={value} className="min-h-96" disabled />
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleEdit}>{t('common.edit')}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
export default CustomEditDialog;
