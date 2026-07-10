'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { KeyRound, Pencil, Plus, Trash2, UserIcon, Users, CircleQuestionMark } from 'lucide-react';
import { toast } from 'sonner';

import { InfoTooltip } from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
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
import { getAccessToken, getRefreshToken } from '@/lib/auth-storage';
import request from '@/lib/request';
import { cn, decodeJwtPayload } from '@/lib/utils';
import type { UserItem } from '@/types/services';
import { ChangePasswordDialog } from './change-password-dialog';
import { CreateUserDialog } from './create-user-dialog';
import { EditPermissionsDialog } from './edit-permissions-dialog';
import { PermissionBadges } from './permissions';

export default function UserManagement() {
  const { t } = useI18n();
  const [users, setUsers] = useState<UserItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [togglingId, setTogglingId] = useState<number | null>(null);
  const [currentUserId, setCurrentUserId] = useState<number | null>(null);
  const [expandedPermissionIds, setExpandedPermissionIds] = useState<number[]>([]);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await request.get<UserItem[]>('/v1/admin/users');
      setUsers(Array.isArray(data) ? data : []);
    } catch {
      setUsers([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const payload = decodeJwtPayload(getAccessToken()) || decodeJwtPayload(getRefreshToken());
    const userId = Number(payload?.user_id);
    setCurrentUserId(Number.isFinite(userId) ? userId : null);
  }, []);

  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  const deleteUser = useMemo(
    () => users.find((user) => user.id === deleteId) || null,
    [deleteId, users]
  );

  const handleToggle = async (user: UserItem) => {
    if (user.id === currentUserId) {
      return;
    }

    setTogglingId(user.id);
    try {
      await request.put(`/v1/admin/users/${user.id}`, { enabled: !user.enabled });
      setUsers((prev) =>
        prev.map((item) => (item.id === user.id ? { ...item, enabled: !user.enabled } : item))
      );
    } catch {
      // handled by interceptor
    } finally {
      setTogglingId(null);
    }
  };

  const handleDelete = async () => {
    if (deleteId == null) return;

    setDeleteLoading(true);
    try {
      await request.delete(`/v1/admin/users/${deleteId}`);
      toast.success(t('common.deleteSuccess'));
      setDeleteId(null);
      await fetchUsers();
    } catch {
      // handled by interceptor
    } finally {
      setDeleteLoading(false);
    }
  };

  const togglePermissionExpanded = (userId: number) => {
    setExpandedPermissionIds((prev) =>
      prev.includes(userId) ? prev.filter((item) => item !== userId) : [...prev, userId]
    );
  };

  return (
    <PageContainer
      title={t('menu.userManagement')}
      subTitle={t('userManagement.pageDescription')}
      extraContent={
        <CreateUserDialog
          trigger={
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              {t('userManagement.createUser')}
            </Button>
          }
          onSuccess={fetchUsers}
        />
      }
    >
      <div className="overflow-hidden rounded-xl border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t('userManagement.username')}</TableHead>
              <TableHead>{t('userManagement.source')}</TableHead>
              <TableHead>{t('userManagement.permissions')}</TableHead>
              <TableHead>{t('userManagement.createdAt')}</TableHead>
              <TableHead>{t('userManagement.enabled')}</TableHead>
              <TableHead className="text-right">{t('common.operation')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={6} className="py-16 text-center text-muted-foreground">
                  {t('userManagement.loading')}
                </TableCell>
              </TableRow>
            ) : users.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="py-16">
                  <div className="flex flex-col items-center gap-3 text-muted-foreground">
                    <Users className="h-10 w-10 opacity-30" />
                    <p className="text-sm">{t('userManagement.noUsers')}</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              users.map((user) => {
                const isCurrentUser = user.id === currentUserId;

                return (
                  <TableRow key={user.id}>
                    <TableCell>
                      <div className="flex items-center gap-2 font-medium">
                        <UserIcon className="h-4 w-4 text-muted-foreground" />
                        {user.username}
                        {isCurrentUser && (
                          <Badge variant="secondary">{t('userManagement.currentUser')}</Badge>
                        )}
                        {user.must_change_password && (
                          <Badge
                            variant="outline"
                            className="border-amber-400 text-xs text-amber-500"
                          >
                            {t('userManagement.mustChangePwd')}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">{user.source}</Badge>
                    </TableCell>
                    <TableCell>
                      <PermissionBadges
                        permissions={user.permissions}
                        expanded={expandedPermissionIds.includes(user.id)}
                        onToggleExpanded={() => togglePermissionExpanded(user.id)}
                      />
                    </TableCell>
                    <TableCell className="whitespace-nowrap text-muted-foreground">
                      {user.created_at || '-'}
                    </TableCell>
                    <TableCell>
                      {isCurrentUser ? (
                        <InfoTooltip content={t('userManagement.currentUserToggleHint')}>
                          <Badge
                            className={cn(
                              user.enabled
                                ? 'border-green-500/30 bg-green-500/15 text-green-600 dark:text-green-400'
                                : 'border-red-500/30 bg-red-500/15 text-red-600 dark:text-red-400'
                            )}
                          >
                            {user.enabled
                              ? t('userManagement.enabledStatus')
                              : t('userManagement.disabledStatus')}
                            <CircleQuestionMark />
                          </Badge>
                        </InfoTooltip>
                      ) : (
                        <Switch
                          checked={user.enabled}
                          disabled={togglingId === user.id}
                          onChange={() => handleToggle(user)}
                        />
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        {user.source === 'local' && (
                          <ChangePasswordDialog
                            trigger={
                              <Button
                                variant="ghost"
                                size="sm"
                                title={t('userManagement.changePassword')}
                              >
                                <KeyRound className="h-4 w-4" />
                              </Button>
                            }
                            userDetail={user}
                            onSuccess={fetchUsers}
                          />
                        )}
                        <EditPermissionsDialog
                          trigger={
                            <Button variant="ghost" size="sm" title={t('common.edit')}>
                              <Pencil className="h-4 w-4" />
                            </Button>
                          }
                          userDetail={user}
                          onSuccess={fetchUsers}
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                          title={t('common.delete')}
                          onClick={() => setDeleteId(user.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>

      <ConfirmDialog
        isOpen={deleteId != null}
        onOpenChange={(open) => {
          if (!open) setDeleteId(null);
        }}
        onConfirm={handleDelete}
        title={t('userManagement.deleteTitle')}
        description={t('userManagement.deleteDescription', {
          username: deleteUser?.username || '',
        })}
        confirmText={t('common.delete')}
        confirmClassName="bg-destructive text-white hover:bg-destructive/90"
        isLoading={deleteLoading}
      />
    </PageContainer>
  );
}
