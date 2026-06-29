'use client';

import { useCallback, useEffect, useState } from 'react';
import { Copy, Eye, EyeOff, KeyRound, Plus, RotateCw, Trash2 } from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
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
import PageContainer from '@/components/ui/page-container';
import { Select } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useI18n } from '@/contexts/i18n-context';
import { useForm, useWatch } from '@/hooks/use-form';
import { decodeJwtScopes } from '@/lib/utils';
import request from '@/lib/request';
import Cookies from 'js-cookie';
import { NO_AUTH } from '@/constants';

interface ApiKey {
  id: number;
  user_id: number;
  key_prefix: string;
  name: string | null;
  description: string | null;
  enabled: boolean;
  expires_at: string | null;
  model_permissions: string[];
  created_at: string | null;
  rotated_at: string | null;
  token_budget: number | null;
  token_usage: number;
  token_remaining: number | null;
  token_budget_exhausted: boolean;
  token_renewal: 'none' | 'daily' | 'monthly' | 'custom';
  token_renewal_interval_days: number | null;
  token_renewal_next_at: string | null;
  request_rate_limit_enabled: boolean;
  request_rate_limit_requests: number | null;
  request_rate_limit_window_seconds: number | null;
  request_rate_limit_count: number;
  request_rate_limit_remaining: number | null;
  request_rate_limit_reset_at: string | null;
}

const optionalNumber = (value: unknown) => {
  if (value === undefined || value === null || value === '') return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const formatInteger = (value: unknown) => Number(value || 0).toLocaleString();

export default function ApiKeyManagement() {
  const { t } = useI18n();

  const token = Cookies.get('token');
  const jwtScopes = decodeJwtScopes(token === NO_AUTH ? undefined : token);
  const isAdmin = jwtScopes.includes('admin');
  const canManageKeys = isAdmin || jwtScopes.includes('keys:manage');

  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);

  // create
  const [createOpen, setCreateOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [form] = useForm();
  const tokenRenewal = useWatch('token_renewal', form) || 'none';
  const requestRateLimitEnabled = Boolean(useWatch('request_rate_limit_enabled', form));
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);
  const [newKeyVisible, setNewKeyVisible] = useState(false);

  // delete
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  // toggle enabled
  const [togglingId, setTogglingId] = useState<number | null>(null);

  // rotate
  const [rotateId, setRotateId] = useState<number | null>(null);
  const [rotateLoading, setRotateLoading] = useState(false);
  const [rotatedKeyValue, setRotatedKeyValue] = useState<string | null>(null);
  const [rotatedKeyVisible, setRotatedKeyVisible] = useState(false);

  const fetchKeys = useCallback(async () => {
    setLoading(true);
    try {
      const data = await request.get<ApiKey[]>('/v1/admin/keys');
      setKeys(Array.isArray(data) ? data : []);
    } catch {
      setKeys([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const openCreate = () => {
    form.resetFields();
    setNewKeyValue(null);
    setNewKeyVisible(false);
    setCreateOpen(true);
  };

  const handleCreate = async (values: Record<string, unknown>) => {
    setCreateLoading(true);
    try {
      const body: Record<string, unknown> = { name: values.name };
      if (values.description) body.description = values.description;
      if (values.expires_at) body.expires_at = values.expires_at;
      const tokenBudget = optionalNumber(values.token_budget);
      if (tokenBudget !== undefined) body.token_budget = tokenBudget;
      body.token_renewal = values.token_renewal || 'none';
      if (values.token_renewal === 'custom') {
        body.token_renewal_interval_days = optionalNumber(values.token_renewal_interval_days);
      }
      body.request_rate_limit_enabled = Boolean(values.request_rate_limit_enabled);
      if (values.request_rate_limit_enabled) {
        body.request_rate_limit_requests = optionalNumber(values.request_rate_limit_requests);
        body.request_rate_limit_window_seconds = optionalNumber(
          values.request_rate_limit_window_seconds
        );
      }
      const result = await request.post<{ key: string }>('/v1/admin/keys', body);
      setNewKeyValue(result.key);
      setNewKeyVisible(false);
      await fetchKeys();
    } catch {
      // handled by interceptor
    } finally {
      setCreateLoading(false);
    }
  };

  const handleRotate = async () => {
    if (rotateId == null) return;
    setRotateLoading(true);
    try {
      const result = await request.post<{ key: string }>(`/v1/admin/keys/${rotateId}/rotate`, {});
      setRotatedKeyValue(result.key);
      setRotatedKeyVisible(false);
      await fetchKeys();
    } catch {
      // handled by interceptor
    } finally {
      setRotateLoading(false);
    }
  };

  const handleDelete = async () => {
    if (deleteId == null) return;
    setDeleteLoading(true);
    try {
      await request.delete(`/v1/admin/keys/${deleteId}`);
      toast.success(t('common.deleteSuccess'));
      setDeleteId(null);
      await fetchKeys();
    } catch {
      // handled by interceptor
    } finally {
      setDeleteLoading(false);
    }
  };

  const handleToggleEnabled = async (key: ApiKey) => {
    setTogglingId(key.id);
    try {
      await request.put(`/v1/admin/keys/${key.id}`, { enabled: !key.enabled });
      setKeys((prev) => prev.map((k) => (k.id === key.id ? { ...k, enabled: !key.enabled } : k)));
    } catch {
      // handled by interceptor
    } finally {
      setTogglingId(null);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        toast.success(t('common.copySuccess'));
      })
      .catch((err) => {
        console.error('Failed to copy text: ', err);
      });
  };

  const maskKey = (prefix: string) => `${prefix}${'•'.repeat(16)}`;

  const formatDate = (iso: string | null) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString();
  };

  const formatDateTime = (iso: string | null) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleString();
  };

  const isExpired = (expiresAt: string | null) => {
    if (!expiresAt) return false;
    return new Date(expiresAt) < new Date();
  };

  const getStatusBadge = (key: ApiKey) => {
    if (isExpired(key.expires_at)) {
      return <Badge variant="destructive">{t('apiKey.expired')}</Badge>;
    }
    if (!key.enabled) {
      return <Badge variant="secondary">{t('apiKey.disabled')}</Badge>;
    }
    if (key.token_budget_exhausted) {
      return <Badge variant="destructive">{t('apiKey.tokenExhausted')}</Badge>;
    }
    if (key.request_rate_limit_enabled && key.request_rate_limit_remaining === 0) {
      return <Badge variant="secondary">{t('apiKey.rateLimited')}</Badge>;
    }
    return (
      <Badge className="bg-green-500/15 text-green-600 border-green-500/30 dark:text-green-400">
        {t('apiKey.active')}
      </Badge>
    );
  };

  const renderTokenUsage = (key: ApiKey) => {
    if (key.token_budget == null) {
      return t('apiKey.usedTokensUnlimited', {
        used: formatInteger(key.token_usage),
      });
    }
    return t('apiKey.usedTokens', {
      used: formatInteger(key.token_usage),
      total: formatInteger(key.token_budget),
      remaining: formatInteger(key.token_remaining),
    });
  };

  const renderRenewal = (key: ApiKey) => {
    const label = t(`apiKey.tokenRenewal_${key.token_renewal || 'none'}`);
    if (!key.token_renewal_next_at) return label;
    return `${label} · ${formatDateTime(key.token_renewal_next_at)}`;
  };

  const renderRateLimit = (key: ApiKey) => {
    if (!key.request_rate_limit_enabled) return t('apiKey.noLimit');
    return t('apiKey.rateLimitUsage', {
      used: formatInteger(key.request_rate_limit_count),
      total: formatInteger(key.request_rate_limit_requests),
      remaining: formatInteger(key.request_rate_limit_remaining),
    });
  };

  return (
    <PageContainer
      title={t('menu.apiKeyManagement')}
      subTitle={t('apiKey.pageDescription')}
      extraContent={
        <Button onClick={openCreate}>
          <Plus className="h-4 w-4 mr-2" />
          {t('apiKey.createKey')}
        </Button>
      }
    >
      <div className="rounded-xl border border-border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t('apiKey.name')}</TableHead>
              <TableHead>{t('apiKey.key')}</TableHead>
              <TableHead>{t('apiKey.description')}</TableHead>
              <TableHead>{t('apiKey.createdAt')}</TableHead>
              <TableHead>{t('apiKey.expiresAt')}</TableHead>
              <TableHead>{t('apiKey.tokenUsage')}</TableHead>
              <TableHead>{t('apiKey.tokenRenewal')}</TableHead>
              <TableHead>{t('apiKey.rateLimit')}</TableHead>
              <TableHead>{t('apiKey.lastRotated')}</TableHead>
              <TableHead>{t('apiKey.status')}</TableHead>
              <TableHead>{t('apiKey.enabled')}</TableHead>
              <TableHead className="text-right">{t('common.operation')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={12} className="text-center py-16 text-muted-foreground">
                  {t('apiKey.loading')}
                </TableCell>
              </TableRow>
            ) : keys.length === 0 ? (
              <TableRow>
                <TableCell colSpan={12} className="py-16">
                  <div className="flex flex-col items-center gap-3 text-muted-foreground">
                    <KeyRound className="h-10 w-10 opacity-30" />
                    <p className="text-sm">{t('apiKey.noKeys')}</p>
                    <Button variant="outline" size="sm" onClick={openCreate}>
                      <Plus className="h-4 w-4 mr-1" />
                      {t('apiKey.createKey')}
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              keys.map((key) => (
                <TableRow key={key.id}>
                  <TableCell className="font-medium">{key.name || '—'}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1.5 font-mono text-xs">
                      <span className="text-muted-foreground">{maskKey(key.key_prefix)}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground max-w-[180px] truncate">
                    {key.description || '—'}
                  </TableCell>
                  <TableCell
                    className="text-muted-foreground whitespace-nowrap"
                    suppressHydrationWarning
                  >
                    {formatDate(key.created_at)}
                  </TableCell>
                  <TableCell
                    className="text-muted-foreground whitespace-nowrap"
                    suppressHydrationWarning
                  >
                    {formatDate(key.expires_at)}
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {renderTokenUsage(key)}
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {renderRenewal(key)}
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {renderRateLimit(key)}
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {formatDateTime(key.rotated_at)}
                  </TableCell>
                  <TableCell>{getStatusBadge(key)}</TableCell>
                  <TableCell>
                    <Switch
                      checked={key.enabled}
                      disabled={
                        !canManageKeys || togglingId === key.id || isExpired(key.expires_at)
                      }
                      onChange={() => handleToggleEnabled(key)}
                    />
                  </TableCell>
                  <TableCell className="text-right">
                    {canManageKeys && (
                      <div className="flex justify-end gap-1">
                        <Button variant="ghost" size="sm" onClick={() => setRotateId(key.id)}>
                          <RotateCw className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={() => setDeleteId(key.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* ── Create Dialog ─────────────────────────────── */}
      <Dialog
        open={createOpen}
        onOpenChange={(open) => {
          if (!open) setCreateOpen(false);
          else setCreateOpen(true);
        }}
      >
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {newKeyValue ? t('apiKey.keyCreated') : t('apiKey.createKey')}
            </DialogTitle>
          </DialogHeader>

          {newKeyValue ? (
            <div className="flex flex-col gap-4">
              <p className="text-sm text-muted-foreground">{t('apiKey.saveKeyWarning')}</p>
              <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3 py-2">
                <span className="flex-1 font-mono text-sm break-all">
                  {newKeyVisible ? newKeyValue : '•'.repeat(Math.min(newKeyValue.length, 40))}
                </span>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => setNewKeyVisible((v) => !v)}
                >
                  {newKeyVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => copyToClipboard(newKeyValue)}
                >
                  <Copy className="h-4 w-4" />
                </button>
              </div>
              <DialogFooter>
                <Button onClick={() => setCreateOpen(false)}>{t('common.confirm')}</Button>
              </DialogFooter>
            </div>
          ) : (
            <Form
              form={form}
              initialValues={{
                token_renewal: 'none',
                request_rate_limit_enabled: false,
              }}
              onFinish={handleCreate}
            >
              <FormField
                name="name"
                label={t('apiKey.name')}
                placeholder={t('apiKey.namePlaceholder')}
                rules={[{ required: true, message: t('apiKey.nameRequired') }]}
              >
                <Input />
              </FormField>

              <FormField
                name="description"
                label={t('apiKey.description')}
                placeholder={t('apiKey.descriptionPlaceholder')}
              >
                <Input />
              </FormField>

              <FormField
                name="expires_at"
                label={t('apiKey.expiresAt')}
                extra={t('apiKey.expiresAtHint')}
              >
                <Input
                  type="date"
                  min={new Date().toISOString().split('T')[0]}
                  suppressHydrationWarning
                />
              </FormField>

              <div className="rounded-lg border border-border bg-muted/20 p-3 space-y-3">
                <div>
                  <p className="text-sm font-medium">{t('apiKey.advancedSettings')}</p>
                  <p className="text-xs text-muted-foreground">
                    {t('apiKey.advancedSettingsHint')}
                  </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <FormField
                    name="token_budget"
                    label={t('apiKey.tokenBudget')}
                    extra={t('apiKey.tokenBudgetHint')}
                  >
                    <Input type="number" min={1} />
                  </FormField>

                  <FormField name="token_renewal" label={t('apiKey.tokenRenewal')}>
                    <Select
                      allowClear={false}
                      options={[
                        { value: 'none', label: t('apiKey.tokenRenewal_none') },
                        { value: 'daily', label: t('apiKey.tokenRenewal_daily') },
                        { value: 'monthly', label: t('apiKey.tokenRenewal_monthly') },
                        { value: 'custom', label: t('apiKey.tokenRenewal_custom') },
                      ]}
                    />
                  </FormField>
                </div>

                {tokenRenewal === 'custom' && (
                  <FormField
                    name="token_renewal_interval_days"
                    label={t('apiKey.tokenRenewalIntervalDays')}
                    rules={[
                      {
                        required: true,
                        message: t('apiKey.tokenRenewalIntervalRequired'),
                      },
                    ]}
                  >
                    <Input type="number" min={1} />
                  </FormField>
                )}

                <FormField
                  name="request_rate_limit_enabled"
                  label={t('apiKey.requestRateLimitEnabled')}
                  valuePropName="checked"
                  layout="horizontal"
                >
                  <Switch />
                </FormField>

                {requestRateLimitEnabled && (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <FormField
                      name="request_rate_limit_requests"
                      label={t('apiKey.rateLimitRequests')}
                      rules={[
                        {
                          required: true,
                          message: t('apiKey.rateLimitRequestsRequired'),
                        },
                      ]}
                    >
                      <Input type="number" min={1} />
                    </FormField>
                    <FormField
                      name="request_rate_limit_window_seconds"
                      label={t('apiKey.rateLimitWindowSeconds')}
                      rules={[
                        {
                          required: true,
                          message: t('apiKey.rateLimitWindowRequired'),
                        },
                      ]}
                    >
                      <Input type="number" min={1} />
                    </FormField>
                  </div>
                )}
              </div>

              <DialogFooter>
                <Button variant="outline" type="button" onClick={() => setCreateOpen(false)}>
                  {t('common.cancel')}
                </Button>
                <Button type="submit" disabled={createLoading}>
                  {createLoading ? t('apiKey.creating') : t('apiKey.create')}
                </Button>
              </DialogFooter>
            </Form>
          )}
        </DialogContent>
      </Dialog>

      {/* ── Delete Confirm ────────────────────────────── */}
      <ConfirmDialog
        isOpen={deleteId != null}
        onOpenChange={(open) => {
          if (!open) setDeleteId(null);
        }}
        onConfirm={handleDelete}
        title={t('apiKey.deleteTitle')}
        description={t('apiKey.deleteDescription')}
        confirmText={t('common.delete')}
        confirmClassName="bg-destructive text-white hover:bg-destructive/90"
        isLoading={deleteLoading}
      />

      {/* ── Rotate Key Dialog ─────────────────────────── */}
      <Dialog
        open={rotateId != null}
        onOpenChange={(open) => {
          if (!open) {
            setRotateId(null);
            setRotatedKeyValue(null);
            setRotatedKeyVisible(false);
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              {rotatedKeyValue ? t('apiKey.keyRotated') : t('apiKey.rotateKey')}
            </DialogTitle>
          </DialogHeader>
          {rotatedKeyValue ? (
            <div className="flex flex-col gap-4">
              <p className="text-sm text-muted-foreground">{t('apiKey.rotateKeyWarning')}</p>
              <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3 py-2">
                <span className="flex-1 font-mono text-sm break-all">
                  {rotatedKeyVisible
                    ? rotatedKeyValue
                    : '•'.repeat(Math.min(rotatedKeyValue.length, 40))}
                </span>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => setRotatedKeyVisible((v) => !v)}
                >
                  {rotatedKeyVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => copyToClipboard(rotatedKeyValue)}
                >
                  <Copy className="h-4 w-4" />
                </button>
              </div>
              <DialogFooter>
                <Button
                  onClick={() => {
                    setRotateId(null);
                    setRotatedKeyValue(null);
                    setRotatedKeyVisible(false);
                  }}
                >
                  {t('common.confirm')}
                </Button>
              </DialogFooter>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">{t('apiKey.rotateKeyConfirm')}</p>
              <DialogFooter>
                <Button variant="outline" onClick={() => setRotateId(null)}>
                  {t('common.cancel')}
                </Button>
                <Button onClick={handleRotate} disabled={rotateLoading}>
                  {rotateLoading ? t('apiKey.rotating') : t('apiKey.rotate')}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </PageContainer>
  );
}
