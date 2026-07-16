'use client';

import { ReactNode, useState } from 'react';
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
import { Form } from '@/components/ui/form';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { useI18n } from '@/contexts/i18n-context';
import { useForm } from '@/hooks/use-form';
import request from '@/lib/request';
import type { UserItem } from '@/types/services';

interface ChangePasswordDialogProps {
  trigger: ReactNode;
  userDetail: UserItem;
  onSuccess?: () => void | Promise<void>;
}

export function ChangePasswordDialog({
  trigger,
  userDetail,
  onSuccess,
}: ChangePasswordDialogProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [form] = useForm();

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (nextOpen) {
      form.resetFields();
    }
  };

  const handleChangePassword = async (values: Record<string, unknown>) => {
    setLoading(true);
    try {
      await request.put(`/v1/admin/users/${userDetail.id}/password`, {
        new_password: values.new_password,
      });
      toast.success(t('userManagement.passwordChanged'));
      setOpen(false);
      await onSuccess?.();
    } catch {
      // handled by interceptor
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>
            {t('userManagement.changePasswordTitle', { username: userDetail.username })}
          </DialogTitle>
        </DialogHeader>
        <Form form={form} onFinish={handleChangePassword}>
          <FormField
            name="new_password"
            label={t('userManagement.newPassword')}
            placeholder={t('userManagement.passwordPlaceholder')}
            rules={[{ required: true, message: t('userManagement.passwordRequired') }]}
          >
            <Input type="password" />
          </FormField>

          <FormField
            name="confirm_password"
            label={t('userManagement.confirmPassword')}
            placeholder={t('userManagement.confirmPasswordPlaceholder')}
            rules={[
              { required: true, message: t('userManagement.passwordRequired') },
              {
                validator: (val: unknown) => val === form.getFieldValue('new_password'),
                message: t('userManagement.passwordMismatch'),
              },
            ]}
          >
            <Input type="password" />
          </FormField>

          <DialogFooter>
            <Button variant="outline" type="button" onClick={() => setOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button type="submit" loading={loading}>
              {t('common.confirm')}
            </Button>
          </DialogFooter>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
