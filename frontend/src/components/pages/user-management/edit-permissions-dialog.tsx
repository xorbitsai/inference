'use client';

import { ReactNode, useEffect, useState } from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import type { UserItem } from '@/types/services';
import { PermissionSelector } from './permissions';

interface EditPermissionsDialogProps {
  trigger: ReactNode;
  userDetail: UserItem;
  onSuccess: () => void | Promise<void>;
}

export function EditPermissionsDialog({
  trigger,
  userDetail,
  onSuccess,
}: EditPermissionsDialogProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [permissions, setPermissions] = useState<string[]>(userDetail.permissions || []);

  useEffect(() => {
    if (open) {
      setPermissions([...(userDetail.permissions || [])]);
    }
  }, [open, userDetail.permissions]);

  const handleSave = async () => {
    setLoading(true);
    try {
      await request.put(`/v1/admin/users/${userDetail.id}`, {
        permissions,
      });
      toast.success(t('userManagement.updateSuccess'));
      setOpen(false);
      await onSuccess();
    } catch {
      // handled by interceptor
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {t('userManagement.editPermissionsTitle', { username: userDetail.username })}
          </DialogTitle>
        </DialogHeader>
        <PermissionSelector selected={permissions} onChange={setPermissions} />
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleSave} loading={loading}>
            {t('common.confirm')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
