'use client';

import { useEffect, useState } from 'react';
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
import { Form } from '@/components/ui/form';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { useI18n } from '@/contexts/i18n-context';
import { useForm } from '@/hooks/use-form';
import request from '@/lib/request';

interface ChangePasswordDialogProps {
  open: boolean;
  userId: number | null;
  onChanged: () => void;
}

export function ChangePasswordDialog({ open, userId, onChanged }: ChangePasswordDialogProps) {
  const { t } = useI18n();
  const [form] = useForm();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open) {
      form.resetFields();
    }
  }, [form, open]);

  const handleChangePassword = async (values: Record<string, unknown>) => {
    if (userId == null) return;

    setLoading(true);
    try {
      await request.put(`/v1/admin/users/${userId}/password`, {
        new_password: values.new_password,
      });
      toast.success(t('userManagement.passwordChanged'));
      onChanged();
    } catch {
      // handled by interceptor
    } finally {
      setLoading(false);
    }
  };
  return (
    <Dialog open={open} onOpenChange={() => {}}>
      <DialogContent
        className="!w-lg"
        showCloseButton={false}
        maskClosable={false}
        onEscapeKeyDown={(event) => event.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>{t('userManagement.changePassword')}</DialogTitle>
          <DialogDescription>{t('login.mustChangePasswordDesc')}</DialogDescription>
        </DialogHeader>
        <Form form={form} onFinish={handleChangePassword}>
          <FormField
            name="new_password"
            label={t('userManagement.newPassword')}
            placeholder={t('userManagement.passwordPlaceholder')}
            rules={[
              { required: true, message: t('userManagement.passwordRequired') },
              {
                validator: (val: unknown) => typeof val === 'string' && val.length > 8,
                message: t('userManagement.passwordTooShort'),
              },
            ]}
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
            <Button block type="submit" className="!mt-2" loading={loading}>
              {t('userManagement.changePassword')}
            </Button>
          </DialogFooter>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
