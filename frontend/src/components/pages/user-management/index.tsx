'use client';

import { useCallback, useEffect, useState } from 'react';
import { KeyRound, Pencil, Plus, Trash2, UserIcon, Users } from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox-group';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
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
import request from '@/lib/request';

interface User {
  id: number;
  username: string;
  source: string;
  enabled: boolean;
  must_change_password: boolean;
  permissions: string[];
  created_at: string | null;
}

const ALL_PERMISSIONS = [
  'admin',
  'models:list',
  'models:read',
  'models:write',
  'keys:create',
  'keys:manage',
  'users:manage',
  'cache:list',
  'cache:delete',
  'virtualenv:list',
  'virtualenv:delete',
];

interface CreateForm {
  username: string;
  password: string;
  confirmPassword: string;
  permissions: string[];
}

interface EditForm {
  permissions: string[];
  enabled: boolean;
}

interface PasswordForm {
  newPassword: string;
  confirmPassword: string;
}

const EMPTY_CREATE: CreateForm = {
  username: '',
  password: '',
  confirmPassword: '',
  permissions: [],
};

export default function UserManagement() {
  const { t } = useI18n();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);

  // create
  const [createOpen, setCreateOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [createForm, setCreateForm] = useState<CreateForm>(EMPTY_CREATE);
  const [createErrors, setCreateErrors] = useState<Partial<Record<keyof CreateForm, string>>>({});

  // edit
  const [editUser, setEditUser] = useState<User | null>(null);
  const [editForm, setEditForm] = useState<EditForm>({ permissions: [], enabled: true });
  const [editLoading, setEditLoading] = useState(false);

  // change password
  const [pwdUser, setPwdUser] = useState<User | null>(null);
  const [pwdForm, setPwdForm] = useState<PasswordForm>({ newPassword: '', confirmPassword: '' });
  const [pwdErrors, setPwdErrors] = useState<Partial<Record<keyof PasswordForm, string>>>({});
  const [pwdLoading, setPwdLoading] = useState(false);

  // delete
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  // toggle enabled
  const [togglingId, setTogglingId] = useState<number | null>(null);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await request.get<User[]>('/v1/admin/users');
      setUsers(Array.isArray(data) ? data : []);
    } catch {
      setUsers([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  // ── Create ────────────────────────────────────────────────
  const validateCreate = () => {
    const errs: Partial<Record<keyof CreateForm, string>> = {};
    if (!createForm.username.trim()) errs.username = t('userManagement.usernameRequired');
    if (!createForm.password) errs.password = t('userManagement.passwordRequired');
    if (createForm.password !== createForm.confirmPassword)
      errs.confirmPassword = t('userManagement.passwordMismatch');
    setCreateErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleCreate = async () => {
    if (!validateCreate()) return;
    setCreateLoading(true);
    try {
      await request.post('/v1/admin/users', {
        username: createForm.username.trim(),
        password: createForm.password,
        permissions: createForm.permissions,
      });
      toast.success(t('userManagement.createSuccess'));
      setCreateOpen(false);
      setCreateForm(EMPTY_CREATE);
      await fetchUsers();
    } catch {
      // handled by interceptor
    } finally {
      setCreateLoading(false);
    }
  };

  // ── Edit ──────────────────────────────────────────────────
  const openEdit = (user: User) => {
    setEditUser(user);
    setEditForm({ permissions: [...(user.permissions || [])], enabled: user.enabled });
  };

  const handleEdit = async () => {
    if (!editUser) return;
    setEditLoading(true);
    try {
      await request.put(`/v1/admin/users/${editUser.id}`, {
        enabled: editForm.enabled,
        permissions: editForm.permissions,
      });
      toast.success(t('userManagement.updateSuccess'));
      setEditUser(null);
      await fetchUsers();
    } catch {
      // handled by interceptor
    } finally {
      setEditLoading(false);
    }
  };

  // ── Toggle enabled (inline) ───────────────────────────────
  const handleToggle = async (user: User) => {
    setTogglingId(user.id);
    try {
      await request.put(`/v1/admin/users/${user.id}`, { enabled: !user.enabled });
      setUsers((prev) =>
        prev.map((u) => (u.id === user.id ? { ...u, enabled: !user.enabled } : u))
      );
    } catch {
      // handled by interceptor
    } finally {
      setTogglingId(null);
    }
  };

  // ── Change Password ───────────────────────────────────────
  const validatePwd = () => {
    const errs: Partial<Record<keyof PasswordForm, string>> = {};
    if (!pwdForm.newPassword) errs.newPassword = t('userManagement.passwordRequired');
    if (pwdForm.newPassword !== pwdForm.confirmPassword)
      errs.confirmPassword = t('userManagement.passwordMismatch');
    setPwdErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleChangePassword = async () => {
    if (!pwdUser || !validatePwd()) return;
    setPwdLoading(true);
    try {
      await request.put(`/v1/admin/users/${pwdUser.id}/password`, {
        new_password: pwdForm.newPassword,
      });
      toast.success(t('userManagement.passwordChanged'));
      setPwdUser(null);
      setPwdForm({ newPassword: '', confirmPassword: '' });
    } catch {
      // handled by interceptor
    } finally {
      setPwdLoading(false);
    }
  };

  // ── Delete ────────────────────────────────────────────────
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

  const formatDate = (iso: string | null) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString();
  };

  return (
    <PageContainer
      title={t('menu.userManagement')}
      subTitle={t('userManagement.pageDescription')}
      extraContent={
        <Button
          onClick={() => {
            setCreateForm(EMPTY_CREATE);
            setCreateErrors({});
            setCreateOpen(true);
          }}
        >
          <Plus className="h-4 w-4 mr-2" />
          {t('userManagement.createUser')}
        </Button>
      }
    >
      <div className="rounded-xl border border-border overflow-hidden">
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
                <TableCell colSpan={6} className="text-center py-16 text-muted-foreground">
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
              users.map((user) => (
                <TableRow key={user.id}>
                  <TableCell>
                    <div className="flex items-center gap-2 font-medium">
                      <UserIcon className="h-4 w-4 text-muted-foreground" />
                      {user.username}
                      {user.must_change_password && (
                        <Badge variant="outline" className="text-xs text-amber-500 border-amber-400">
                          {t('userManagement.mustChangePwd')}
                        </Badge>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">{user.source}</Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1 max-w-sm">
                      {user.permissions.length === 0 ? (
                        <span className="text-muted-foreground text-xs">—</span>
                      ) : user.permissions.includes('admin') ? (
                        <Badge className="bg-primary/15 text-primary border-primary/30">admin</Badge>
                      ) : (
                        user.permissions.slice(0, 3).map((p) => (
                          <Badge key={p} variant="outline" className="text-xs">
                            {p}
                          </Badge>
                        ))
                      )}
                      {!user.permissions.includes('admin') && user.permissions.length > 3 && (
                        <Badge variant="outline" className="text-xs text-muted-foreground">
                          +{user.permissions.length - 3}
                        </Badge>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap" suppressHydrationWarning>
                    {formatDate(user.created_at)}
                  </TableCell>
                  <TableCell>
                    <Switch
                      checked={user.enabled}
                      disabled={togglingId === user.id}
                      onChange={() => handleToggle(user)}
                    />
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-1">
                      {user.source === 'local' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          title={t('userManagement.changePassword')}
                          onClick={() => {
                            setPwdUser(user);
                            setPwdForm({ newPassword: '', confirmPassword: '' });
                            setPwdErrors({});
                          }}
                        >
                          <KeyRound className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        title={t('common.edit')}
                        onClick={() => openEdit(user)}
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-destructive hover:text-destructive hover:bg-destructive/10"
                        title={t('common.delete')}
                        onClick={() => setDeleteId(user.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* ── Create Dialog ───────────────────────────────────── */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t('userManagement.createUser')}</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="create-username">
                {t('userManagement.username')} <span className="text-destructive">*</span>
              </Label>
              <Input
                id="create-username"
                value={createForm.username}
                onChange={(e) => {
                  setCreateForm((f) => ({ ...f, username: e.target.value }));
                  setCreateErrors((e2) => ({ ...e2, username: undefined }));
                }}
                placeholder={t('userManagement.usernamePlaceholder')}
                error={!!createErrors.username}
              />
              {createErrors.username && (
                <p className="text-xs text-destructive">{createErrors.username}</p>
              )}
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="create-password">
                {t('userManagement.password')} <span className="text-destructive">*</span>
              </Label>
              <Input
                id="create-password"
                type="password"
                value={createForm.password}
                onChange={(e) => {
                  setCreateForm((f) => ({ ...f, password: e.target.value }));
                  setCreateErrors((e2) => ({ ...e2, password: undefined }));
                }}
                placeholder={t('userManagement.passwordPlaceholder')}
                error={!!createErrors.password}
              />
              {createErrors.password && (
                <p className="text-xs text-destructive">{createErrors.password}</p>
              )}
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="create-confirm">
                {t('userManagement.confirmPassword')} <span className="text-destructive">*</span>
              </Label>
              <Input
                id="create-confirm"
                type="password"
                value={createForm.confirmPassword}
                onChange={(e) => {
                  setCreateForm((f) => ({ ...f, confirmPassword: e.target.value }));
                  setCreateErrors((e2) => ({ ...e2, confirmPassword: undefined }));
                }}
                placeholder={t('userManagement.confirmPasswordPlaceholder')}
                error={!!createErrors.confirmPassword}
              />
              {createErrors.confirmPassword && (
                <p className="text-xs text-destructive">{createErrors.confirmPassword}</p>
              )}
            </div>

            <PermissionSelector
              label={t('userManagement.permissions')}
              selected={createForm.permissions}
              onChange={(perms) => setCreateForm((f) => ({ ...f, permissions: perms }))}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleCreate} disabled={createLoading}>
              {createLoading ? t('userManagement.creating') : t('userManagement.create')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Edit Dialog ─────────────────────────────────────── */}
      <Dialog open={!!editUser} onOpenChange={(open) => { if (!open) setEditUser(null); }}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t('userManagement.editUser')} — {editUser?.username}
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between rounded-lg border border-border px-4 py-3">
              <Label>{t('userManagement.enabled')}</Label>
              <Switch
                checked={editForm.enabled}
                onChange={(v) => setEditForm((f) => ({ ...f, enabled: v }))}
              />
            </div>
            <PermissionSelector
              label={t('userManagement.permissions')}
              selected={editForm.permissions}
              onChange={(perms) => setEditForm((f) => ({ ...f, permissions: perms }))}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditUser(null)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleEdit} disabled={editLoading}>
              {editLoading ? t('userManagement.saving') : t('common.confirm')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Change Password Dialog ───────────────────────────── */}
      <Dialog open={!!pwdUser} onOpenChange={(open) => { if (!open) setPwdUser(null); }}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>
              {t('userManagement.changePassword')} — {pwdUser?.username}
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="new-password">
                {t('userManagement.newPassword')} <span className="text-destructive">*</span>
              </Label>
              <Input
                id="new-password"
                type="password"
                value={pwdForm.newPassword}
                onChange={(e) => {
                  setPwdForm((f) => ({ ...f, newPassword: e.target.value }));
                  setPwdErrors((e2) => ({ ...e2, newPassword: undefined }));
                }}
                placeholder={t('userManagement.passwordPlaceholder')}
                error={!!pwdErrors.newPassword}
              />
              {pwdErrors.newPassword && (
                <p className="text-xs text-destructive">{pwdErrors.newPassword}</p>
              )}
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="confirm-new-password">
                {t('userManagement.confirmPassword')} <span className="text-destructive">*</span>
              </Label>
              <Input
                id="confirm-new-password"
                type="password"
                value={pwdForm.confirmPassword}
                onChange={(e) => {
                  setPwdForm((f) => ({ ...f, confirmPassword: e.target.value }));
                  setPwdErrors((e2) => ({ ...e2, confirmPassword: undefined }));
                }}
                placeholder={t('userManagement.confirmPasswordPlaceholder')}
                error={!!pwdErrors.confirmPassword}
              />
              {pwdErrors.confirmPassword && (
                <p className="text-xs text-destructive">{pwdErrors.confirmPassword}</p>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPwdUser(null)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleChangePassword} disabled={pwdLoading}>
              {pwdLoading ? t('userManagement.saving') : t('common.confirm')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Delete Confirm ───────────────────────────────────── */}
      <ConfirmDialog
        isOpen={deleteId != null}
        onOpenChange={(open) => { if (!open) setDeleteId(null); }}
        onConfirm={handleDelete}
        title={t('userManagement.deleteTitle')}
        description={t('userManagement.deleteDescription')}
        confirmText={t('common.delete')}
        confirmClassName="bg-destructive text-white hover:bg-destructive/90"
        isLoading={deleteLoading}
      />
    </PageContainer>
  );
}

// ── Permission Selector ──────────────────────────────────────
function PermissionSelector({
  label,
  selected,
  onChange,
}: {
  label: string;
  selected: string[];
  onChange: (perms: string[]) => void;
}) {
  const toggle = (perm: string) => {
    onChange(
      selected.includes(perm) ? selected.filter((p) => p !== perm) : [...selected, perm]
    );
  };

  return (
    <div className="flex flex-col gap-2">
      <Label>{label}</Label>
      <div className="rounded-lg border border-border p-3 grid grid-cols-2 gap-2">
        {ALL_PERMISSIONS.map((perm) => (
          <label
            key={perm}
            className="flex items-center gap-2 cursor-pointer select-none"
          >
            <Checkbox
              checked={selected.includes(perm)}
              onCheckedChange={() => toggle(perm)}
            />
            <span className="text-sm font-mono">{perm}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
