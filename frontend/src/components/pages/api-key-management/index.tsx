'use client';

import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  ChevronDown,
  Copy,
  Edit3,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  Plus,
  ShieldBan,
  Trash2,
} from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import PageContainer from '@/components/ui/page-container';
import { Switch } from '@/components/ui/switch';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';
import { cn } from '@/lib/utils';
import { ApiKeyDialog } from './api-key-dialog';
import {
  getBannedCount,
  getPermissionLabel,
  getPermissionType,
  getPermissionValue,
  type ApiKey,
  type ApiKeyUser,
  type ModelPermission,
} from './utils';

const toDash = (value: unknown) => {
  if (value === undefined || value === null || value === '') {
    return '-';
  }

  return String(value);
};

const padDatePart = (value: number) => String(value).padStart(2, '0');

const permissionTypeValues = new Set(['LLM', 'embedding', 'rerank', 'image', 'video', 'audio']);

export default function ApiKeyManagement() {
  const { t } = useI18n();
  const { isAdmin, canCreateKeys, canManageKeys } = useMenuAuth();

  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [users, setUsers] = useState<ApiKeyUser[]>([]);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<ApiKey | null>(null);

  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  const [revealedKeys, setRevealedKeys] = useState<Record<number, string>>({});
  const [revealingId, setRevealingId] = useState<number | null>(null);
  const [expandedIds, setExpandedIds] = useState<Record<number, boolean>>({});

  const [togglingId, setTogglingId] = useState<number | null>(null);

  const [bannedKey, setBannedKey] = useState<ApiKey | null>(null);
  const [bannedLoading, setBannedLoading] = useState(false);
  const [bannedList, setBannedList] = useState<unknown[]>([]);

  const userNameMap = useMemo(
    () => new Map(users.map((user) => [String(user.id), user.username])),
    [users]
  );
  const fetchKeys = useCallback(async (expandedId?: number) => {
    setLoading(true);
    try {
      const data = await request.get<ApiKey[]>('/v1/admin/keys');
      const list = Array.isArray(data) ? data : [];
      setKeys(list);
      setExpandedIds(
        list.some((item) => item.id === expandedId)
          ? { [String(expandedId)]: true }
          : list[0]
            ? { [list[0].id]: true }
            : {}
      );
    } catch {
      setKeys([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchUsers = useCallback(async () => {
    try {
      const data = await request.get<ApiKeyUser[]>('/v1/admin/users');
      setUsers(Array.isArray(data) ? data : []);
    } catch {
      setUsers([]);
    }
  }, []);

  useEffect(() => {
    fetchKeys();
    if (isAdmin) fetchUsers();
  }, [fetchKeys, fetchUsers, isAdmin]);

  const openCreate = () => {
    setEditingKey(null);
    setDialogOpen(true);
  };

  const openEdit = (key: ApiKey) => {
    setEditingKey(key);
    setDialogOpen(true);
  };

  const closeDialog = () => {
    setDialogOpen(false);
    setEditingKey(null);
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
        prev.map((item) => (item.id === key.id ? { ...item, enabled: !key.enabled } : item))
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

  const openBannedDialog = async (key: ApiKey) => {
    setBannedKey(key);
    setBannedLoading(true);
    setBannedList([]);
    try {
      const data = await request.get<unknown>(`/v1/admin/keys/${key.id}/banned`);
      const list = Array.isArray(data)
        ? data
        : Array.isArray((data as { data?: unknown[] })?.data)
          ? (data as { data: unknown[] }).data
          : Array.isArray((data as { items?: unknown[] })?.items)
            ? (data as { items: unknown[] }).items
            : [];
      setBannedList(list);
    } catch {
      setBannedList([]);
    } finally {
      setBannedLoading(false);
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

  const formatDate = (iso: string | null) => {
    if (!iso) return '-';

    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '-';

    const year = date.getFullYear();
    const month = padDatePart(date.getMonth() + 1);
    const day = padDatePart(date.getDate());
    const hours = padDatePart(date.getHours());
    const minutes = padDatePart(date.getMinutes());
    const seconds = padDatePart(date.getSeconds());

    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  };

  const isExpired = (expiresAt: string | null) => {
    if (!expiresAt) return false;
    return new Date(expiresAt) < new Date();
  };

  const getStatusBadge = (key: ApiKey) => {
    if (!key.enabled) {
      return (
        <Badge className="border-red-500/30 bg-red-500/15 text-red-600 dark:text-red-400">
          {t('apiKey.disabled')}
        </Badge>
      );
    }
    return (
      <Badge className="border-green-500/30 bg-green-500/15 text-green-600 dark:text-green-400">
        {t('apiKey.enabled')}
      </Badge>
    );
  };

  const renderPermissions = (permissions: ModelPermission[]) => {
    if (!permissions?.length) {
      return (
        <Badge className="border-blue-500/30 bg-blue-500/15 text-blue-700 dark:text-blue-300">
          {t('apiKey.permissionAll')}
        </Badge>
      );
    }

    return (
      <div className="flex flex-wrap gap-1.5">
        {permissions.map((permission, index) => {
          const type = getPermissionType(permission);
          const value = getPermissionValue(permission);
          const isAllPermission = type === 'all';
          const isModelType = type === 'model_type' || permissionTypeValues.has(value);

          return (
            <Badge
              key={`${type}-${value}-${index}`}
              variant={isAllPermission || isModelType ? 'default' : 'outline'}
              className={cn(
                'font-normal',
                isAllPermission &&
                  'border-blue-500/30 bg-blue-500/15 text-blue-700 hover:bg-blue-500/20 dark:text-blue-300',
                isModelType &&
                  !isAllPermission &&
                  'border-emerald-500/30 bg-emerald-500/15 text-emerald-700 hover:bg-emerald-500/20 dark:text-emerald-300'
              )}
            >
              {isAllPermission ? t('apiKey.permissionAll') : getPermissionLabel(permission)}
            </Badge>
          );
        })}
      </div>
    );
  };

  const renderKeyValue = (key: ApiKey) => {
    const revealedValue = revealedKeys[key.id];
    const displayValue = revealedValue || `${key.key_prefix}${'*'.repeat(44)}`;
    return (
      <div className="flex min-w-0 items-center gap-2">
        <span className="min-w-0 truncate font-mono text-xs text-muted-foreground">
          {displayValue}
        </span>
        {canManageKeys && (
          <>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              title={revealedKeys[key.id] ? t('apiKey.hideKey') : t('apiKey.revealKey')}
              className="size-7"
              onClick={() => handleReveal(key.id)}
              disabled={revealingId === key.id}
            >
              {revealingId === key.id ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : revealedKeys[key.id] ? (
                <EyeOff className="size-3.5" />
              ) : (
                <Eye className="size-3.5" />
              )}
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              title={t('common.copySuccess')}
              className="size-7"
              disabled={!revealedValue}
              onClick={() => copyToClipboard(revealedValue)}
            >
              <Copy className="size-3.5" />
            </Button>
          </>
        )}
      </div>
    );
  };

  const renderField = (label: string, value: ReactNode) => (
    <div className="rounded-lg bg-muted/40 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1.5 min-w-0 text-sm font-medium text-foreground">{value}</div>
    </div>
  );

  const renderOwner = (key: ApiKey) => {
    if (key.owner_username) return key.owner_username;
    if (key.user_id == null) return '-';

    return userNameMap.get(String(key.user_id)) || key.user_id;
  };

  return (
    <PageContainer
      title={t('menu.apiKeyManagement')}
      subTitle={t('apiKey.pageDescription')}
      extraContent={
        canCreateKeys && (
          <Button onClick={openCreate}>
            <Plus className="size-4" />
            {t('apiKey.createKey')}
          </Button>
        )
      }
    >
      {loading ? (
        <div className="flex min-h-[58vh] flex-col items-center justify-center gap-3 text-muted-foreground">
          <Loader2 className="size-8 animate-spin" />
          <p className="text-sm">{t('apiKey.loading')}</p>
        </div>
      ) : keys.length === 0 ? (
        <div className="flex min-h-[58vh] flex-col items-center justify-center gap-4 text-center">
          <div className="flex size-16 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <KeyRound className="size-8" />
          </div>
          <p className="text-sm font-medium text-muted-foreground">{t('apiKey.noKeys')}</p>
        </div>
      ) : (
        <div className="space-y-4">
          {keys.map((key) => {
            const expanded = expandedIds[key.id] ?? false;

            return (
              <div
                key={key.id}
                className="overflow-hidden rounded-xl border bg-card text-card-foreground shadow-sm"
              >
                <div className="flex items-center gap-3 px-4 py-3">
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-3 text-left"
                    onClick={() =>
                      setExpandedIds((prev) => ({
                        ...prev,
                        [key.id]: !expanded,
                      }))
                    }
                  >
                    <KeyRound className="size-5 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 flex-1 truncate text-sm font-semibold">
                      {t('apiKey.key')} {key.id}
                    </span>
                    <ChevronDown
                      className={cn(
                        'size-4 shrink-0 text-muted-foreground transition-transform',
                        expanded && 'rotate-180'
                      )}
                    />
                  </button>

                  {canManageKeys && (
                    <div className="flex shrink-0 items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        title={t('common.edit')}
                        className="size-8"
                        onClick={() => openEdit(key)}
                      >
                        <Edit3 className="size-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        title={t('common.delete')}
                        className="size-8 text-destructive hover:bg-destructive/10 hover:text-destructive"
                        onClick={() => setDeleteId(key.id)}
                      >
                        <Trash2 className="size-4" />
                      </Button>
                    </div>
                  )}
                </div>

                {expanded && (
                  <div className="border-t px-4 py-4">
                    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                      {renderField(t('apiKey.name'), toDash(key.name))}
                      {renderField(t('apiKey.key'), renderKeyValue(key))}
                      {renderField(
                        t('apiKey.status'),
                        <div className="flex items-center gap-3">
                          {getStatusBadge(key)}
                          {canManageKeys && (
                            <Switch
                              checked={key.enabled}
                              disabled={togglingId === key.id || isExpired(key.expires_at)}
                              onChange={() => handleToggleEnabled(key)}
                            />
                          )}
                        </div>
                      )}
                      {renderField(t('apiKey.owner'), renderOwner(key))}
                      {renderField(t('apiKey.createdAt'), formatDate(key.created_at))}
                      {renderField(t('apiKey.expiresAt'), formatDate(key.expires_at))}
                      {renderField(
                        t('apiKey.modelPermissions'),
                        renderPermissions(key.model_permissions)
                      )}
                      {renderField(
                        t('apiKey.bannedCount'),
                        <div className="flex items-center gap-3">
                          <span>{getBannedCount(key)}</span>
                          {isAdmin && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => openBannedDialog(key)}
                            >
                              <ShieldBan className="size-3.5" />
                              {t('apiKey.view')}
                            </Button>
                          )}
                        </div>
                      )}
                      {renderField(t('apiKey.description'), toDash(key.description))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <ApiKeyDialog
        open={dialogOpen}
        apiKey={editingKey}
        users={users}
        onOpenChange={(open) => {
          if (!open) {
            closeDialog();
          } else setDialogOpen(true);
        }}
        onSuccess={() => fetchKeys(Boolean(editingKey) ? editingKey?.id : undefined)}
      />

      <Dialog
        open={!!bannedKey}
        onOpenChange={(open) => {
          if (!open) {
            setBannedKey(null);
            setBannedList([]);
          }
        }}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {t('apiKey.bannedList')} - {t('apiKey.key')} {bannedKey?.id}
            </DialogTitle>
          </DialogHeader>

          {bannedLoading ? (
            <div className="flex min-h-40 items-center justify-center text-muted-foreground">
              <Loader2 className="size-6 animate-spin" />
            </div>
          ) : bannedList.length === 0 ? (
            <div className="flex min-h-40 flex-col items-center justify-center gap-3 text-muted-foreground">
              <ShieldBan className="size-8 opacity-50" />
              <p className="text-sm">{t('apiKey.noBanned')}</p>
            </div>
          ) : (
            <div className="space-y-2">
              {bannedList.map((item, index) => (
                <div key={index} className="rounded-lg border bg-muted/30 p-3 text-sm">
                  <pre className="whitespace-pre-wrap break-words font-mono text-xs">
                    {typeof item === 'string' ? item : JSON.stringify(item, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </DialogContent>
      </Dialog>

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
