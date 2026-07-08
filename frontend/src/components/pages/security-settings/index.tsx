'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Ban,
  KeyRound,
  Loader2,
  Network,
  RotateCcw,
  Save,
  ShieldCheck,
  Trash2,
  Unlock,
} from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import PageContainer from '@/components/ui/page-container';
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
import { cn } from '@/lib/utils';

type RateLimitValue = number | '';

interface RateLimitLayerConfig {
  max_failures?: RateLimitValue;
  window_seconds?: RateLimitValue;
  ban_seconds?: RateLimitValue;
}

interface RateLimitConfig {
  ip: RateLimitLayerConfig;
  key: RateLimitLayerConfig;
}

interface BannedIp {
  ip: string;
  remaining_seconds?: number;
}

interface BannedKey {
  ip: string;
  key_id: string | number;
  remaining_seconds?: number;
}

type Layer = 'ip' | 'key';
type Field = keyof RateLimitLayerConfig;
type ClearTarget = 'ips' | 'keys' | null;

const defaultConfig: RateLimitConfig = {
  ip: {},
  key: {},
};

function normalizeConfig(value: Partial<RateLimitConfig> | null | undefined): RateLimitConfig {
  return {
    ip: value?.ip || {},
    key: value?.key || {},
  };
}

function sanitizeLayer(layer: RateLimitLayerConfig) {
  return Object.fromEntries(
    Object.entries(layer).filter(
      ([, value]) => value !== '' && value !== null && value !== undefined
    )
  );
}

function formatSeconds(
  value: number | undefined,
  t: (key: string, vars?: Record<string, number>) => string
) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '-';
  }

  if (value < 60) {
    return t('securitySettings.seconds', { count: Math.max(0, Math.round(value)) });
  }

  if (value < 3600) {
    return t('securitySettings.minutes', { count: Math.ceil(value / 60) });
  }

  return t('securitySettings.hours', { count: Math.ceil(value / 3600) });
}

function displayConfigValue(value: RateLimitValue | undefined) {
  return value === '' || value === undefined ? '-' : value;
}

export default function SecuritySettings() {
  const { t } = useI18n();
  const [config, setConfig] = useState<RateLimitConfig>(defaultConfig);
  const [bannedIps, setBannedIps] = useState<BannedIp[]>([]);
  const [bannedKeys, setBannedKeys] = useState<BannedKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [unbanLoadingKey, setUnbanLoadingKey] = useState<string | null>(null);
  const [clearTarget, setClearTarget] = useState<ClearTarget>(null);
  const [clearing, setClearing] = useState(false);

  const fetchConfig = useCallback(async () => {
    const data = await request.get<Partial<RateLimitConfig>>('/v1/admin/security/rate-limit');
    setConfig(normalizeConfig(data));
  }, []);

  const fetchBannedIps = useCallback(async () => {
    const data = await request.get<BannedIp[]>('/v1/admin/security/banned-ips');
    setBannedIps(Array.isArray(data) ? data : []);
  }, []);

  const fetchBannedKeys = useCallback(async () => {
    const data = await request.get<BannedKey[]>('/v1/admin/security/banned-keys');
    setBannedKeys(Array.isArray(data) ? data : []);
  }, []);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    try {
      const [configResult, ipsResult, keysResult] = await Promise.allSettled([
        fetchConfig(),
        fetchBannedIps(),
        fetchBannedKeys(),
      ]);

      if (configResult.status === 'rejected') setConfig(defaultConfig);
      if (ipsResult.status === 'rejected') setBannedIps([]);
      if (keysResult.status === 'rejected') setBannedKeys([]);
    } finally {
      setLoading(false);
    }
  }, [fetchBannedIps, fetchBannedKeys, fetchConfig]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const stats = useMemo(
    () => [
      {
        label: t('securitySettings.ipLayer'),
        value: displayConfigValue(config.ip.max_failures),
        detail: t('securitySettings.failuresAllowed'),
        Icon: Network,
        tone: 'text-sky-600 bg-sky-500/10',
      },
      {
        label: t('securitySettings.keyLayer'),
        value: displayConfigValue(config.key.max_failures),
        detail: t('securitySettings.failuresAllowed'),
        Icon: KeyRound,
        tone: 'text-violet-600 bg-violet-500/10',
      },
      {
        label: t('securitySettings.bannedIps'),
        value: bannedIps.length,
        detail: t('securitySettings.currentBlocks'),
        Icon: Ban,
        tone: 'text-rose-600 bg-rose-500/10',
      },
      {
        label: t('securitySettings.bannedKeys'),
        value: bannedKeys.length,
        detail: t('securitySettings.currentBlocks'),
        Icon: ShieldCheck,
        tone: 'text-emerald-600 bg-emerald-500/10',
      },
    ],
    [bannedIps.length, bannedKeys.length, config.ip.max_failures, config.key.max_failures, t]
  );

  const updateConfig = (layer: Layer, field: Field, value: string) => {
    setConfig((prev) => ({
      ...prev,
      [layer]: {
        ...prev[layer],
        [field]: value === '' ? '' : Math.max(0, Number.parseInt(value, 10) || 0),
      },
    }));
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      await request.put('/v1/admin/security/rate-limit', {
        ip: sanitizeLayer(config.ip),
        key: sanitizeLayer(config.key),
      });
      toast.success(t('securitySettings.saveSuccess'));
      await fetchConfig();
    } catch {
      // handled by interceptor
    } finally {
      setSaving(false);
    }
  };

  const unbanIp = async (ip: string) => {
    setUnbanLoadingKey(`ip:${ip}`);
    try {
      await request.post('/v1/admin/security/unban-ip', { ip });
      toast.success(t('securitySettings.unbanSuccess'));
      await fetchBannedIps();
    } catch {
      // handled by interceptor
    } finally {
      setUnbanLoadingKey(null);
    }
  };

  const unbanKey = async (ip: string, keyId: string | number) => {
    setUnbanLoadingKey(`key:${ip}:${keyId}`);
    try {
      await request.post('/v1/admin/security/unban-key', { ip, key_id: keyId });
      toast.success(t('securitySettings.unbanSuccess'));
      await fetchBannedKeys();
    } catch {
      // handled by interceptor
    } finally {
      setUnbanLoadingKey(null);
    }
  };

  const clearAll = async () => {
    if (!clearTarget) return;

    setClearing(true);
    try {
      if (clearTarget === 'ips') {
        await request.post('/v1/admin/security/unban-all-ips');
        await fetchBannedIps();
      } else {
        await request.post('/v1/admin/security/unban-all-keys');
        await fetchBannedKeys();
      }
      toast.success(t('securitySettings.unbanSuccess'));
      setClearTarget(null);
    } catch {
      // handled by interceptor
    } finally {
      setClearing(false);
    }
  };

  const renderNumberInput = (layer: Layer, field: Field) => (
    <div className="space-y-2">
      <Label htmlFor={`${layer}-${field}`}>{t(`securitySettings.${field}`)}</Label>
      <Input
        id={`${layer}-${field}`}
        type="number"
        min={0}
        value={config[layer][field] ?? ''}
        placeholder={t('securitySettings.notConfigured')}
        onChange={(event) => updateConfig(layer, field, event.target.value)}
      />
    </div>
  );

  return (
    <PageContainer
      title={t('menu.securitySettings')}
      subTitle={t('securitySettings.pageDescription')}
      loading={loading}
      extraContent={
        <Button variant="outline" onClick={refreshAll} disabled={loading || saving}>
          <RotateCcw className="mr-2 h-4 w-4" />
          {t('securitySettings.refresh')}
        </Button>
      }
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {stats.map(({ label, value, detail, Icon, tone }) => (
          <Card key={label} className="rounded-lg py-5 shadow-none">
            <CardContent className="flex items-center gap-4">
              <div className={cn('flex h-11 w-11 items-center justify-center rounded-lg', tone)}>
                <Icon className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="text-sm text-muted-foreground">{label}</div>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-2xl font-semibold">{value}</span>
                  <span className="text-xs text-muted-foreground">{detail}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="mt-6 rounded-lg shadow-none">
        <CardHeader className="border-b">
          <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
            <div>
              <CardTitle>{t('securitySettings.rateLimitConfig')}</CardTitle>
              <CardDescription className="mt-2">
                {t('securitySettings.rateLimitDesc')}
              </CardDescription>
            </div>
            <Button onClick={saveConfig} disabled={saving} className="w-fit">
              {saving ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Save className="mr-2 h-4 w-4" />
              )}
              {t('securitySettings.saveConfig')}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-6 lg:grid-cols-2">
          {(['ip', 'key'] as Layer[]).map((layer) => (
            <section key={layer} className="rounded-lg border border-border p-5">
              <div className="mb-5 flex items-center gap-3">
                <div
                  className={cn(
                    'flex h-9 w-9 items-center justify-center rounded-lg',
                    layer === 'ip'
                      ? 'bg-sky-500/10 text-sky-600'
                      : 'bg-violet-500/10 text-violet-600'
                  )}
                >
                  {layer === 'ip' ? (
                    <Network className="h-4 w-4" />
                  ) : (
                    <KeyRound className="h-4 w-4" />
                  )}
                </div>
                <div>
                  <h2 className="font-semibold">{t(`securitySettings.${layer}Layer`)}</h2>
                  <p className="text-sm text-muted-foreground">
                    {t(`securitySettings.${layer}LayerDesc`)}
                  </p>
                </div>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                {renderNumberInput(layer, 'max_failures')}
                {renderNumberInput(layer, 'window_seconds')}
                {renderNumberInput(layer, 'ban_seconds')}
              </div>
            </section>
          ))}
        </CardContent>
      </Card>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <BlockTable
          title={t('securitySettings.bannedIps')}
          description={t('securitySettings.bannedIpsDesc')}
          clearDisabled={bannedIps.length === 0}
          onClear={() => setClearTarget('ips')}
        >
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t('securitySettings.ip')}</TableHead>
                <TableHead>{t('securitySettings.remaining')}</TableHead>
                <TableHead className="text-right">{t('common.operation')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {bannedIps.length === 0 ? (
                <EmptyRow colSpan={3} text={t('securitySettings.noBannedIps')} />
              ) : (
                bannedIps.map((row) => {
                  const loadingKey = `ip:${row.ip}`;

                  return (
                    <TableRow key={row.ip}>
                      <TableCell className="font-medium">{row.ip}</TableCell>
                      <TableCell>{formatSeconds(row.remaining_seconds, t)}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          title={t('securitySettings.unban')}
                          onClick={() => unbanIp(row.ip)}
                          disabled={unbanLoadingKey === loadingKey}
                        >
                          {unbanLoadingKey === loadingKey ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Unlock className="h-4 w-4" />
                          )}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </BlockTable>

        <BlockTable
          title={t('securitySettings.bannedKeys')}
          description={t('securitySettings.bannedKeysDesc')}
          clearDisabled={bannedKeys.length === 0}
          onClear={() => setClearTarget('keys')}
        >
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t('securitySettings.ip')}</TableHead>
                <TableHead>{t('securitySettings.keyId')}</TableHead>
                <TableHead>{t('securitySettings.remaining')}</TableHead>
                <TableHead className="text-right">{t('common.operation')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {bannedKeys.length === 0 ? (
                <EmptyRow colSpan={4} text={t('securitySettings.noBannedKeys')} />
              ) : (
                bannedKeys.map((row) => {
                  const loadingKey = `key:${row.ip}:${row.key_id}`;

                  return (
                    <TableRow key={`${row.ip}-${row.key_id}`}>
                      <TableCell className="font-medium">{row.ip}</TableCell>
                      <TableCell>{row.key_id}</TableCell>
                      <TableCell>{formatSeconds(row.remaining_seconds, t)}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          title={t('securitySettings.unban')}
                          onClick={() => unbanKey(row.ip, row.key_id)}
                          disabled={unbanLoadingKey === loadingKey}
                        >
                          {unbanLoadingKey === loadingKey ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Unlock className="h-4 w-4" />
                          )}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </BlockTable>
      </div>

      <ConfirmDialog
        isOpen={clearTarget !== null}
        onOpenChange={(open) => {
          if (!open) setClearTarget(null);
        }}
        onConfirm={clearAll}
        isLoading={clearing}
        title={t('securitySettings.clearTitle')}
        description={
          clearTarget === 'ips'
            ? t('securitySettings.clearIpsDescription')
            : t('securitySettings.clearKeysDescription')
        }
        confirmText={t('securitySettings.unbanAll')}
        confirmClassName="bg-destructive text-destructive-foreground hover:bg-destructive/90"
      />
    </PageContainer>
  );
}

interface BlockTableProps {
  title: string;
  description: string;
  clearDisabled: boolean;
  onClear: () => void;
  children: React.ReactNode;
}

function BlockTable({ title, description, clearDisabled, onClear, children }: BlockTableProps) {
  const { t } = useI18n();

  return (
    <Card className="rounded-lg shadow-none">
      <CardHeader className="border-b">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription className="mt-2">{description}</CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="shrink-0 text-destructive hover:bg-destructive/10 hover:text-destructive"
            disabled={clearDisabled}
            onClick={onClear}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            {t('securitySettings.unbanAll')}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-hidden rounded-lg border border-border">{children}</div>
      </CardContent>
    </Card>
  );
}

function EmptyRow({ colSpan, text }: { colSpan: number; text: string }) {
  return (
    <TableRow>
      <TableCell colSpan={colSpan} className="py-12 text-center text-muted-foreground">
        {text}
      </TableCell>
    </TableRow>
  );
}
