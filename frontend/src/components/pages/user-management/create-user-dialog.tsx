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
import { PermissionSelector } from './permissions';

interface CreateUserDialogProps {
  trigger?: ReactNode;
  children?: ReactNode;
  onSuccess: () => void | Promise<void>;
}

export function CreateUserDialog({ trigger, children, onSuccess }: CreateUserDialogProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [permissions, setPermissions] = useState<string[]>([]);
  const [form] = useForm();

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (nextOpen) {
      form.resetFields();
      setPermissions([]);
    }
  };

  const handleCreate = async (values: Record<string, unknown>) => {
    setLoading(true);
    try {
      await request.post('/v1/admin/users', {
        username: values.username,
        password: values.password,
        permissions,
      });
      toast.success(t('userManagement.createSuccess'));
      setOpen(false);
      await onSuccess();
    } catch {
      // handled by interceptor
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>{trigger || children}</DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t('userManagement.createUser')}</DialogTitle>
        </DialogHeader>
        <Form form={form} onFinish={handleCreate}>
          <FormField
            name="username"
            label={t('userManagement.username')}
            placeholder={t('userManagement.usernamePlaceholder')}
            rules={[{ required: true, message: t('userManagement.usernameRequired') }]}
          >
            <Input />
          </FormField>

          <FormField
            name="password"
            label={t('userManagement.password')}
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
                validator: (val: unknown) => val === form.getFieldValue('password'),
                message: t('userManagement.passwordMismatch'),
              },
            ]}
          >
            <Input type="password" />
          </FormField>

          <div className="mt-4">
            <PermissionSelector selected={permissions} onChange={setPermissions} />
          </div>

          <DialogFooter className="mt-4">
            <Button variant="outline" type="button" onClick={() => setOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button type="submit" loading={loading}>
              {t('userManagement.create')}
            </Button>
          </DialogFooter>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
