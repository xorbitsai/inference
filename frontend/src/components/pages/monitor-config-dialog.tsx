'use client';

import { Database, Loader2 } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MONITOR_DASHBOARD_TABS } from '@/constants/monitor';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { cn } from '@/lib/utils';

interface MonitorConfigResponse {
  grafana_url: string;
  grafana_datasource: string;
  grafana_alert_datasource: string;
  cluster_name: string;
  grafana_dashboards: Record<string, string>;
  grafana_dashboards_configured: string[];
  sources: Record<string, string>;
}

interface MonitorConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved?: () => void;
}

type FormState = {
  grafana_url: string;
  grafana_datasource: string;
  grafana_alert_datasource: string;
  cluster_name: string;
};

const EMPTY_FORM: FormState = {
  grafana_url: '',
  grafana_datasource: '',
  grafana_alert_datasource: '',
  cluster_name: '',
};

export function MonitorConfigDialog({
  open,
  onOpenChange,
  onSaved,
}: MonitorConfigDialogProps) {
  const { t } = useI18n();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [form, setForm] = useState<FormState>({ ...EMPTY_FORM });
  const [dashboards, setDashboards] = useState<Record<string, string>>({});
  const [sources, setSources] = useState<Record<string, string>>({});
  const [test, setTest] = useState<{ status: 'idle' | 'testing' | 'ok' | 'fail'; msg?: string }>({
    status: 'idle',
  });
  const [confirmReset, setConfirmReset] = useState(false);

  useEffect(() => {
    if (!open) return;
    setTest({ status: 'idle' });
    setConfirmReset(false);
    void load();
  }, [open]);

  async function load() {
    setLoading(true);
    try {
      const data = await request.get<MonitorConfigResponse>('/v1/cluster/monitor_config');
      setForm({
        grafana_url: data.grafana_url ?? '',
        grafana_datasource: data.grafana_datasource ?? '',
        grafana_alert_datasource: data.grafana_alert_datasource ?? '',
        cluster_name: data.cluster_name ?? '',
      });
      // Only pre-fill dashboard UIDs for tabs that are explicitly configured
      // (DB or env). Non-configured tabs must stay empty so that resolved
      // fallback UIDs are never written back as real configuration, which would
      // silently mark every tab as enabled (see GET→PUT round-trip regression).
      const configured = new Set(data.grafana_dashboards_configured ?? []);
      const nextDashboards: Record<string, string> = {};
      for (const tab of MONITOR_DASHBOARD_TABS) {
        nextDashboards[tab.key] = configured.has(tab.key)
          ? data.grafana_dashboards?.[tab.key] ?? ''
          : '';
      }
      setDashboards(nextDashboards);
      setSources(data.sources ?? {});
    } finally {
      setLoading(false);
    }
  }

  async function handleTest() {
    if (!form.grafana_url) return;
    setTest({ status: 'testing' });
    try {
      const res = await request.post<{ ok: boolean; error?: string }>(
        '/v1/cluster/monitor_config/check-grafana',
        { grafana_url: form.grafana_url }
      );
      if (res.ok) {
        setTest({ status: 'ok' });
      } else {
        setTest({ status: 'fail', msg: res.error });
      }
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data
        ?.detail;
      setTest({ status: 'fail', msg: detail });
    }
  }

  async function handleSave() {
    setSaving(true);
    try {
      // Submit every dashboard key explicitly: enabled tabs send their UID,
      // disabled tabs send an empty string so the backend overwrites any stale
      // DB value and the tab falls back to "default" (not configured). This
      // keeps the configured set stable across an unchanged save.
      const payloadDashboards: Record<string, string> = {};
      for (const tab of MONITOR_DASHBOARD_TABS) {
        payloadDashboards[tab.key] = dashboards[tab.key]?.trim() ?? '';
      }
      await request.put('/v1/cluster/monitor_config', {
        ...form,
        grafana_dashboards: payloadDashboards,
      });
      onSaved?.();
      onOpenChange(false);
    } finally {
      setSaving(false);
    }
  }

  async function handleReset() {
    if (!confirmReset) {
      setConfirmReset(true);
      return;
    }
    setResetting(true);
    try {
      await request.post('/v1/cluster/monitor_config/reset');
      await load();
      onSaved?.();
      setConfirmReset(false);
    } finally {
      setResetting(false);
    }
  }

  const SourceBadge = ({ field }: { field: string }) => {
    const src = sources[field] || 'default';
    return (
      <span className="ml-2 inline-flex items-center gap-1 rounded-md bg-emerald-500/10 px-1.5 py-0.5 text-xs font-medium text-emerald-600">
        <Database className="size-3" />
        {t(`monitorCenter.config.source.${src}`)}
      </span>
    );
  };

  const renderField = (
    labelKey: string,
    field: keyof FormState,
    sourceField: string,
    trailing?: ReactNode
  ) => (
    <div className="space-y-1.5">
      <Label className="flex items-center">
        {t(labelKey)}
        <SourceBadge field={sourceField} />
      </Label>
      <div className="flex gap-2">
        <Input
          value={form[field]}
          onChange={(e) => setForm((prev) => ({ ...prev, [field]: e.target.value }))}
        />
        {trailing}
      </div>
    </div>
  );

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) setConfirmReset(false);
        onOpenChange(next);
      }}
    >
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t('monitorCenter.config.title')}</DialogTitle>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="size-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="space-y-4">
            {renderField(
              'monitorCenter.config.grafanaUrl',
              'grafana_url',
              'grafana_url',
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="shrink-0"
                onClick={handleTest}
                disabled={test.status === 'testing' || !form.grafana_url}
              >
                {test.status === 'testing' && <Loader2 className="size-3.5 animate-spin" />}
                {t('monitorCenter.config.testConnection')}
              </Button>
            )}
            {test.status === 'ok' && (
              <p className="-mt-2 text-xs text-emerald-600">
                {t('monitorCenter.config.connectionOk')}
              </p>
            )}
            {test.status === 'fail' && (
              <p className="-mt-2 text-xs text-destructive">
                {t('monitorCenter.config.connectionFailed')}
                {test.msg ? `: ${test.msg}` : ''}
              </p>
            )}

            {renderField(
              'monitorCenter.config.datasourceName',
              'grafana_datasource',
              'grafana_datasource'
            )}
            {renderField(
              'monitorCenter.config.alertDatasourceName',
              'grafana_alert_datasource',
              'grafana_alert_datasource'
            )}
            {renderField(
              'monitorCenter.config.clusterName',
              'cluster_name',
              'cluster_name'
            )}

            <div className="space-y-2">
              <Label className="text-sm font-semibold">
                {t('monitorCenter.config.dashboards')}
              </Label>
              <div className="space-y-2">
                {MONITOR_DASHBOARD_TABS.map((tab) => (
                  <div key={tab.key} className="flex items-center gap-2">
                    <Label className="w-24 shrink-0 text-xs text-muted-foreground">
                      {t(tab.labelKey)}
                    </Label>
                    <Input
                      className="flex-1"
                      value={dashboards[tab.key] ?? ''}
                      onChange={(e) =>
                        setDashboards((prev) => ({ ...prev, [tab.key]: e.target.value }))
                      }
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        <DialogFooter className="gap-2 sm:justify-between">
          <div className="flex flex-col gap-1">
            {confirmReset && (
              <span className="text-xs text-destructive">
                {t('monitorCenter.config.resetConfirm')}
              </span>
            )}
            <Button
              type="button"
              variant="link"
              className={cn(
                'px-0',
                confirmReset ? 'text-destructive' : 'text-orange-600 hover:text-orange-700'
              )}
              onClick={handleReset}
              disabled={loading || saving || resetting}
            >
              {resetting && <Loader2 className="size-3.5 animate-spin" />}
              {t('monitorCenter.config.reset')}
            </Button>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              {t('monitorCenter.config.cancel')}
            </Button>
            <Button onClick={handleSave} disabled={loading || saving || resetting}>
              {saving && <Loader2 className="size-3.5 animate-spin" />}
              {t('monitorCenter.config.save')}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default MonitorConfigDialog;
