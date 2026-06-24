'use client';

import { useCallback, useEffect, useState } from 'react';
import { Copy, Eye, EyeOff, KeyRound, Plus, Trash2 } from 'lucide-react';
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
import { useForm } from '@/hooks/use-form';
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
}

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
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);
  const [newKeyVisible, setNewKeyVisible] = useState(false);

  // delete
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  // reveal
  const [revealedKeys, setRevealedKeys] = useState<Record<number, string>>({});
  const [revealingId, setRevealingId] = useState<number | null>(null);

  // toggle enabled
  const [togglingId, setTogglingId] = useState<number | null>(null);

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
      setKeys((prev) =>
        prev.map((k) => (k.id === key.id ? { ...k, enabled: !key.enabled } : k))
      );
    } catch {
      // handled by interceptor
    } finally {
      setTogglingId(null);
    }
  };

  const handleReveal = async (keyId: number) => {
    if (revealedKeys[keyId]) {
      setRevealedKeys((prev) => {
        const next = { ...prev };
        delete next[keyId];
        return next;
      });
      return;
    }
    setRevealingId(keyId);
    try {
      const data = await request.get<{ key: string }>(`/v1/admin/keys/${keyId}/reveal`);
      setRevealedKeys((prev) => ({ ...prev, [keyId]: data.key }));
    } catch {
      // handled by interceptor
    } finally {
      setRevealingId(null);
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
    return (
      <Badge className="bg-green-500/15 text-green-600 border-green-500/30 dark:text-green-400">
        {t('apiKey.active')}
      </Badge>
    );
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
              <TableHead>{t('apiKey.status')}</TableHead>
              <TableHead>{t('apiKey.enabled')}</TableHead>
              <TableHead className="text-right">{t('common.operation')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center py-16 text-muted-foreground">
                  {t('apiKey.loading')}
                </TableCell>
              </TableRow>
            ) : keys.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="py-16">
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
                      <span className="text-muted-foreground">
                        {revealedKeys[key.id] ? revealedKeys[key.id] : maskKey(key.key_prefix)}
                      </span>
                      {isAdmin && (
                        <button
                          type="button"
                          title={revealedKeys[key.id] ? t('apiKey.hideKey') : t('apiKey.revealKey')}
                          className="text-muted-foreground hover:text-foreground transition-colors"
                          onClick={() => handleReveal(key.id)}
                          disabled={revealingId === key.id}
                        >
                          {revealedKeys[key.id] ? (
                            <EyeOff className="h-3.5 w-3.5" />
                          ) : (
                            <Eye className="h-3.5 w-3.5" />
                          )}
                        </button>
                      )}
                      <button
                        type="button"
                        title={t('common.copySuccess')}
                        disabled={!revealedKeys[key.id]}
                        className="text-muted-foreground hover:text-foreground transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                        onClick={() => revealedKeys[key.id] && copyToClipboard(revealedKeys[key.id])}
                      >
                        <Copy className="h-3.5 w-3.5" />
                      </button>
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
                  <TableCell>{getStatusBadge(key)}</TableCell>
                  <TableCell>
                    <Switch
                      checked={key.enabled}
                      disabled={!canManageKeys || togglingId === key.id || isExpired(key.expires_at)}
                      onChange={() => handleToggleEnabled(key)}
                    />
                  </TableCell>
                  <TableCell className="text-right">
                    {canManageKeys && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => setDeleteId(key.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
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
        <DialogContent className="sm:max-w-md">
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
                  {newKeyVisible
                    ? newKeyValue
                    : '•'.repeat(Math.min(newKeyValue.length, 40))}
                </span>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => setNewKeyVisible((v) => !v)}
                >
                  {newKeyVisible ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
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
            <Form form={form} onFinish={handleCreate}>
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
    </PageContainer>
  );
}
