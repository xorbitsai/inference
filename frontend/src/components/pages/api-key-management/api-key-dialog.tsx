'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Copy, Eye, EyeOff } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { CheckboxGroup } from '@/components/ui/checkbox-group';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Form } from '@/components/ui/form';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { MultiSelect } from '@/components/ui/multi-select';
import { Select, type SelectOption } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { DateTimePicker } from '@/components/ui/date-time-picker';
import { useI18n } from '@/contexts/i18n-context';
import { useForm, useWatch } from '@/hooks/use-form';
import request from '@/lib/request';
import { copyToClipboard } from '@/lib/utils';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import {
  MODEL_TYPE_OPTIONS,
  getPermissionType,
  getPermissionValue,
  type ApiKey,
  type ApiKeyUser,
  type KeyFormValues,
  type ModelPermission,
  type PermissionMode,
  type RunningModel,
} from './utils';

interface ApiKeyDialogProps {
  open: boolean;
  apiKey?: ApiKey | null;
  users: ApiKeyUser[];
  onOpenChange: (open: boolean) => void;
  onSuccess: () => Promise<void> | void;
}

export function ApiKeyDialog({ open, apiKey, users, onOpenChange, onSuccess }: ApiKeyDialogProps) {
  const { t } = useI18n();
  const { isAdmin } = useMenuAuth();
  const [form] = useForm();
  const permissionMode = useWatch('permission_mode', form);
  const [submitLoading, setSubmitLoading] = useState(false);
  const [models, setModels] = useState<RunningModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);
  const [newKeyVisible, setNewKeyVisible] = useState(false);
  const isEdit = Boolean(apiKey);

  const permissionModeOptions: SelectOption<PermissionMode>[] = useMemo(() => {
    const options: SelectOption<PermissionMode>[] = [
      { value: 'all', label: t('apiKey.permissionAll') },
      { value: 'model_type', label: t('apiKey.permissionByType') },
      { value: 'model_id', label: t('apiKey.permissionById') },
    ];

    if (isEdit) {
      options.push({ value: 'mixed', label: t('apiKey.permissionMixed') });
    }

    return options;
  }, [isEdit, t]);
  const modelIdOptions = useMemo(
    () =>
      models
        .map((model) => {
          const value = model.id || model.model_uid || model.model_name;

          if (!value) return null;

          return {
            value,
            label: `${model.model_name || value}(${model.model_type || '-'}) - ${value}`,
          };
        })
        .filter(Boolean) as Array<{ value: string; label: string }>,
    [models]
  );
  const userOptions: SelectOption<string>[] = useMemo(
    () =>
      users.map((user) => ({
        value: String(user.id),
        label: user.username,
      })),
    [users]
  );

  const inferPermissions = useCallback(
    (
      permissions: ModelPermission[] = []
    ): Pick<KeyFormValues, 'permission_mode' | 'model_types' | 'model_ids'> => {
      if (permissions.length === 0 || !Array.isArray(permissions)) {
        return { permission_mode: 'all' };
      }

      const modelTypes: string[] = [];
      const modelIds: string[] = [];
      let hasAllPermission = false;

      permissions.forEach((permission) => {
        const type = getPermissionType(permission);
        const value = getPermissionValue(permission);

        if (type === 'all') {
          hasAllPermission = true;
          return;
        }

        if (type === 'model_type' && value) {
          modelTypes.push(value);
          return;
        }

        if (type === 'model_id' && value) {
          modelIds.push(value);
        }
      });

      if (hasAllPermission || (!modelTypes.length && !modelIds.length)) {
        return { permission_mode: 'all' };
      }

      if (modelTypes.length && modelIds.length) {
        return {
          permission_mode: 'mixed',
          model_types: modelTypes,
          model_ids: modelIds,
        };
      }

      if (modelIds.length) {
        return {
          permission_mode: 'model_id',
          model_ids: modelIds,
        };
      }

      if (modelTypes.length) {
        return {
          permission_mode: 'model_type',
          model_types: modelTypes,
        };
      }

      return { permission_mode: 'all' };
    },
    []
  );

  const resetDialogState = useCallback(() => {
    form.resetFields();
    setSubmitLoading(false);
    setNewKeyValue(null);
    setNewKeyVisible(false);
  }, [form]);

  const hydrateForm = useCallback(() => {
    form.resetFields();

    if (!apiKey) {
      form.setFieldsValue({
        permission_mode: 'all',
        model_types: [],
        model_ids: [],
      });
      return;
    }

    const permissions = inferPermissions(apiKey.model_permissions || []);

    form.setFieldsValue({
      name: apiKey.name || '',
      description: apiKey.description || '',
      ...permissions,
    });
  }, [apiKey, form, inferPermissions]);

  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const data = await request.get<{ data?: RunningModel[] } | RunningModel[]>('/v1/models');
      const list = Array.isArray(data) ? data : data?.data || [];
      setModels(Array.isArray(list) ? list : []);
    } catch {
      setModels([]);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;

    hydrateForm();
    fetchModels();
  }, [fetchModels, hydrateForm, open]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      resetDialogState();
    }

    onOpenChange(nextOpen);
  };

  const handleClose = () => {
    resetDialogState();
    onOpenChange(false);
  };

  const handlePermissionModeChange = () => {
    form.setFieldsValue({
      model_types: [],
      model_ids: [],
    });
  };

  const buildPermissionPayload = (values: KeyFormValues) => {
    if (values.permission_mode === 'model_type' || values.permission_mode === 'mixed') {
      const typePermissions = (values.model_types || []).map((value) => ({
        permission_type: 'model_type',
        permission_value: value,
      }));

      if (values.permission_mode === 'mixed') {
        return [
          ...typePermissions,
          ...(values.model_ids || []).map((value) => ({
            permission_type: 'model_id',
            permission_value: value,
          })),
        ];
      }

      return typePermissions;
    }

    if (values.permission_mode === 'model_id') {
      return (values.model_ids || []).map((value) => ({
        permission_type: 'model_id',
        permission_value: value,
      }));
    }

    return [
      {
        permission_type: 'all',
        permission_value: null,
      },
    ];
  };

  const formatExpiresAt = (value?: string) => {
    if (!value) return undefined;

    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
      return `${value}T23:59:59`;
    }

    return value;
  };

  const buildCreatePayload = (values: KeyFormValues) => {
    const body: Record<string, unknown> = {
      model_permissions: buildPermissionPayload(values),
    };

    if (values.name) body.name = values.name;
    if (values.description) body.description = values.description;
    if (values.user_id) {
      body.owner = /^\d+$/.test(values.user_id) ? Number(values.user_id) : values.user_id;
    }

    const expiresAt = formatExpiresAt(values.expires_at);
    if (expiresAt) body.expires_at = expiresAt;

    return body;
  };

  const buildEditPayload = (values: KeyFormValues) => ({
    name: values.name || null,
    description: values.description || null,
    model_permissions: buildPermissionPayload(values),
  });

  const handleSubmit = async (values: KeyFormValues) => {
    setSubmitLoading(true);
    try {
      if (apiKey) {
        await request.put(`/v1/admin/keys/${apiKey.id}`, buildEditPayload(values));
        toast.success(t('apiKey.updateSuccess'));
        handleClose();
      } else {
        const result = await request.post<{ key: string }>(
          '/v1/admin/keys',
          buildCreatePayload(values)
        );
        setNewKeyValue(result.key);
        setNewKeyVisible(false);
      }

      await onSuccess();
    } catch {
      // handled by interceptor
    } finally {
      setSubmitLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {newKeyValue
              ? t('apiKey.keyCreated')
              : isEdit
                ? `${t('common.edit')} ${t('apiKey.key')} ${apiKey?.id || ''}`
                : t('apiKey.createKey')}
          </DialogTitle>
        </DialogHeader>

        {newKeyValue ? (
          <div className="flex flex-col gap-4">
            <p className="text-sm text-muted-foreground">{t('apiKey.saveKeyWarning')}</p>
            <div className="flex items-center gap-2 rounded-lg border bg-muted/50 px-3 py-2">
              <span className="min-w-0 flex-1 break-all font-mono text-sm">
                {newKeyVisible ? newKeyValue : '•'.repeat(Math.min(newKeyValue.length, 40))}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => setNewKeyVisible((visible) => !visible)}
              >
                {newKeyVisible ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => copyToClipboard(newKeyValue)}
              >
                <Copy className="size-4" />
              </Button>
            </div>
            <DialogFooter>
              <Button onClick={handleClose}>{t('common.confirm')}</Button>
            </DialogFooter>
          </div>
        ) : (
          <Form form={form} onFinish={handleSubmit} className="space-y-4">
            <FormField
              name="name"
              label={t('apiKey.name')}
              placeholder={t('apiKey.namePlaceholder')}
            >
              <Input />
            </FormField>

            <FormField
              name="description"
              label={t('apiKey.description')}
              placeholder={t('apiKey.descriptionPlaceholder')}
            >
              <Textarea />
            </FormField>
            {isAdmin && !apiKey && (
              <FormField
                name="user_id"
                label={t('apiKey.owner')}
                placeholder={t('apiKey.ownerPlaceholder')}
                extra={t('apiKey.ownerHint')}
              >
                <Select options={userOptions} showSearch />
              </FormField>
            )}

            {!apiKey && (
              <FormField
                name="expires_at"
                label={t('apiKey.expiresAt')}
                extra={t('apiKey.expiresAtHint')}
              >
                <DateTimePicker
                  showTime={false}
                  showSelectedTime={false}
                  inputClassName="h-9"
                />
              </FormField>
            )}

            <FormField name="permission_mode" label={t('apiKey.permissionType')}>
              <Select
                options={permissionModeOptions}
                allowClear={false}
                onChange={handlePermissionModeChange}
              />
            </FormField>

            {(permissionMode === 'model_type' || permissionMode === 'mixed') && (
              <FormField name="model_types" rules={[{ required: true }]}>
                <CheckboxGroup options={MODEL_TYPE_OPTIONS} />
              </FormField>
            )}

            {(permissionMode === 'model_id' || permissionMode === 'mixed') && (
              <FormField
                name="model_ids"
                extra={modelsLoading ? t('apiKey.modelsLoading') : undefined}
                rules={[{ required: true }]}
              >
                <MultiSelect
                  searchable={false}
                  options={modelIdOptions}
                  placeholder={t('apiKey.modelIdsPlaceholder')}
                  disabled={modelsLoading}
                />
              </FormField>
            )}

            <DialogFooter>
              <Button variant="outline" type="button" onClick={handleClose}>
                {t('common.cancel')}
              </Button>
              <Button type="submit" loading={submitLoading}>
                {apiKey
                  ? submitLoading
                    ? t('apiKey.saving')
                    : t('common.confirm')
                  : submitLoading
                    ? t('apiKey.creating')
                    : t('apiKey.create')}
              </Button>
            </DialogFooter>
          </Form>
        )}
      </DialogContent>
    </Dialog>
  );
}
