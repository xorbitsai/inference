'use client';

import { useEffect, useId, useRef, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';
import request from '@/lib/request';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import { LAUNCH_MODEL_UPDATE_OPTIONS } from '@/constants/launch';
import type { RequestModelType } from './types';

interface CatalogRefreshProps {
  modelType: RequestModelType;
  onModelTypeChange: (modelType: RequestModelType) => void;
  onRefreshed: (modelType: RequestModelType) => void;
}

function RefreshControls({ modelType, onModelTypeChange, onRefreshed }: CatalogRefreshProps) {
  const { t } = useI18n();
  const inputId = useId();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [pending, setPending] = useState(false);
  const busy = useRef(false);
  const active = useRef(true);

  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);

  const changeOpen = (next: boolean) => {
    if (busy.current) return;
    setOpen(next);
    if (!next) setName('');
  };

  const refresh = async (single: boolean) => {
    const modelName = name.trim();
    if (busy.current || (single && !modelName)) return;
    busy.current = true;
    setPending(true);
    try {
      await request.post(single ? '/v1/models/update_spec' : '/v1/models/update_type', {
        model_type: modelType.toLowerCase(),
        ...(single ? { model_name: modelName } : {}),
      });
      if (!active.current) return;
      setOpen(false);
      setName('');
      toast.success(t('launchModel.catalogUpdated'));
      onRefreshed(modelType);
    } catch {
      // The shared request client presents auth, License and catalog errors.
      // Keep the single-model dialog/input available for an explicit retry.
    } finally {
      busy.current = false;
      if (active.current) setPending(false);
    }
  };

  return (
    <div className="relative z-20 mb-1 flex shrink-0 flex-wrap items-center justify-end gap-2">
      <div className="flex">
        <Select
          value={modelType}
          onChange={(value) => onModelTypeChange(value as RequestModelType)}
          options={LAUNCH_MODEL_UPDATE_OPTIONS}
          allowClear={false}
          disabled={pending}
          className="w-32 rounded-r-none"
          dropdownAutoWidth
        />
        <Button onClick={() => void refresh(false)} disabled={pending} className="rounded-l-none">
          {pending && !open ? <Loader2 className="animate-spin" /> : <RefreshCw />}
          {t('common.update')}
        </Button>
      </div>
      <Dialog open={open} onOpenChange={changeOpen}>
        <DialogTrigger asChild>
          <Button variant="outline" disabled={pending}>
            {t('launchModel.updateOneModel')}
          </Button>
        </DialogTrigger>
        <DialogContent showCloseButton={!pending} maskClosable={!pending}>
          <DialogHeader>
            <DialogTitle>{t('launchModel.updateOneModel')}</DialogTitle>
            <DialogDescription>{t('launchModel.updateOneModelDescription')}</DialogDescription>
          </DialogHeader>
          <form
            id={`${inputId}-form`}
            className="space-y-4"
            onSubmit={(event) => {
              event.preventDefault();
              void refresh(true);
            }}
          >
            <p className="text-sm">
              {t('launchModel.modelType')}:{' '}
              {LAUNCH_MODEL_UPDATE_OPTIONS.find((option) => option.value === modelType)?.label}
            </p>
            <div className="space-y-2">
              <label htmlFor={inputId} className="text-sm font-medium">
                {t('launchModel.modelName')}
              </label>
              <Input
                id={inputId}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder={t('launchModel.catalogModelNamePlaceholder')}
                maxLength={512}
                disabled={pending}
                autoComplete="off"
              />
            </div>
          </form>
          <DialogFooter>
            <Button variant="outline" disabled={pending} onClick={() => changeOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button type="submit" form={`${inputId}-form`} disabled={pending || !name.trim()}>
              {pending && <Loader2 className="animate-spin" />}
              {t('common.update')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default function CatalogRefresh(props: CatalogRefreshProps) {
  const { canRefreshModelCatalog } = useMenuAuth();
  return canRefreshModelCatalog ? <RefreshControls {...props} /> : null;
}
