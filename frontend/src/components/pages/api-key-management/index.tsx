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
import { Textarea } from '@/components/ui/textarea';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';

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

interface CreateForm {
  name: string;
  description: string;
  expires_at: string;
}

const EMPTY_FORM: CreateForm = { name: '', description: '', expires_at: '' };

export default function ApiKeyManagement() {
  const { t } = useI18n();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [form, setForm] = useState<CreateForm>(EMPTY_FORM);
  const [formError, setFormError] = useState('');
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);
  const [newKeyVisible, setNewKeyVisible] = useState(false);
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [revealedKeys, setRevealedKeys] = useState<Record<number, string>>({});
  const [revealingId, setRevealingId] = useState<number | null>(null);
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

  const handleCreate = async () => {
    if (!form.name.trim()) {
      setFormError(t('apiKey.nameRequired'));
      return;
    }
    setCreateLoading(true);
    setFormError('');
    try {
      const body: Record<string, string> = { name: form.name.trim() };
      if (form.description.trim()) body.description = form.description.trim();
      if (form.expires_at) body.expires_at = form.expires_at;
      const result = await request.post<{ key: string }>('/v1/admin/keys', body);
      setNewKeyValue(result.key);
      setNewKeyVisible(false);
      await fetchKeys();
    } catch {
      // error handled by request interceptor
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
      // error handled by request interceptor
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
      // error handled by request interceptor
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
      // error handled by request interceptor
    } finally {
      setRevealingId(null);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast.success(t('common.copySuccess'));
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
        <Button onClick={() => { setCreateOpen(true); setForm(EMPTY_FORM); setFormError(''); setNewKeyValue(null); }}>
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
                    <Button variant="outline" size="sm" onClick={() => { setCreateOpen(true); setForm(EMPTY_FORM); setFormError(''); setNewKeyValue(null); }}>
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
                      <button
                        type="button"
                        title={revealedKeys[key.id] ? t('apiKey.hideKey') : t('apiKey.revealKey')}
                        className="text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => handleReveal(key.id)}
                        disabled={revealingId === key.id}
                      >
                        {revealedKeys[key.id] ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                      <button
                        type="button"
                        title={t('common.copySuccess')}
                        className="text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => copyToClipboard(revealedKeys[key.id] || maskKey(key.key_prefix))}
                      >
                        <Copy className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground max-w-[180px] truncate">
                    {key.description || '—'}
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {formatDate(key.created_at)}
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {formatDate(key.expires_at)}
                  </TableCell>
                  <TableCell>{getStatusBadge(key)}</TableCell>
                  <TableCell>
                    <Switch
                      checked={key.enabled}
                      disabled={togglingId === key.id || isExpired(key.expires_at)}
                      onChange={() => handleToggleEnabled(key)}
                    />
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-destructive hover:text-destructive hover:bg-destructive/10"
                      onClick={() => setDeleteId(key.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Create Dialog */}
      <Dialog
        open={createOpen}
        onOpenChange={(open) => {
          if (!open) { setCreateOpen(false); setNewKeyValue(null); }
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
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="key-name">
                  {t('apiKey.name')} <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="key-name"
                  value={form.name}
                  onChange={(e) => { setForm((f) => ({ ...f, name: e.target.value })); setFormError(''); }}
                  placeholder={t('apiKey.namePlaceholder')}
                  error={!!formError}
                />
                {formError && <p className="text-xs text-destructive">{formError}</p>}
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="key-desc">{t('apiKey.description')}</Label>
                <Textarea
                  id="key-desc"
                  value={form.description}
                  onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
                  placeholder={t('apiKey.descriptionPlaceholder')}
                  rows={3}
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="key-expires">{t('apiKey.expiresAt')}</Label>
                <Input
                  id="key-expires"
                  type="date"
                  value={form.expires_at}
                  onChange={(e) => setForm((f) => ({ ...f, expires_at: e.target.value }))}
                  min={new Date().toISOString().split('T')[0]}
                />
                <p className="text-xs text-muted-foreground">{t('apiKey.expiresAtHint')}</p>
              </div>

              <DialogFooter>
                <Button variant="outline" onClick={() => setCreateOpen(false)}>
                  {t('common.cancel')}
                </Button>
                <Button onClick={handleCreate} disabled={createLoading}>
                  {createLoading ? t('apiKey.creating') : t('apiKey.create')}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirm */}
      <ConfirmDialog
        isOpen={deleteId != null}
        onOpenChange={(open) => { if (!open) setDeleteId(null); }}
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
