'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CircleAlert,
  Copy,
  FileSearch,
  Filter,
  KeyRound,
  Loader2,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  UserRound,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { JSONSyntaxHighlighter } from '@/components/ui/json-syntax-highlighter';
import { Label } from '@/components/ui/label';
import { MultiSelect } from '@/components/ui/multi-select';
import PageContainer from '@/components/ui/page-container';
import { Select } from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { DEFAULT_LOG_TIME_RANGE } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { cn, copyToClipboard } from '@/lib/utils';

import { TimeRangePicker } from '../log-center/time-range-picker';
import type { TimeRangeValue } from '../log-center/types';

const AUDIT_PAGE_SIZE = 50;

interface AuditRecord {
  '@timestamp'?: string;
  event_type?: string;
  category?: string;
  auth_type?: string;
  user?: string;
  api_key_name?: string;
  api_key_prefix?: string;
  model_id?: string;
  model_name?: string;
  model_type?: string;
  endpoint?: string;
  status?: string;
  latency_ms?: number;
  client_ip?: string;
  node?: string;
  address?: string;
  [key: string]: unknown;
}

interface AuditSearchResponse {
  hits?: AuditRecord[];
  total?: number;
}

interface AuditFilters {
  user: string;
  apiKeyName: string;
  modelId: string;
  modelName: string;
  modelType: string[];
  category: string[];
  authType: string;
  status: string[];
  clientIp: string;
}

type AuditFilterKey = keyof AuditFilters;
type AuditTextFilterKey = 'user' | 'apiKeyName' | 'modelId' | 'modelName' | 'clientIp';
type AuditMultiFilterKey = 'modelType' | 'category' | 'status';

const defaultFilters: AuditFilters = {
  user: '',
  apiKeyName: '',
  modelId: '',
  modelName: '',
  modelType: [],
  category: [],
  authType: '',
  status: [],
  clientIp: '',
};

const defaultTextFilters = {
  user: '',
  apiKeyName: '',
  modelId: '',
  modelName: '',
  clientIp: '',
};

const filterParamMap: Record<AuditFilterKey, string> = {
  user: 'user',
  apiKeyName: 'api_key_name',
  modelId: 'model_id',
  modelName: 'model_name',
  modelType: 'model_type',
  category: 'category',
  authType: 'auth_type',
  status: 'status',
  clientIp: 'client_ip',
};

const categoryValues = ['inference', 'admin', 'auth'];
const authTypeValues = ['api_key', 'jwt', 'none'];
const statusValues = [
  'success',
  'error',
  'denied',
  'login_failed',
  'model_not_found',
  'ip_banned',
  'key_banned',
  'key_expired',
  'key_disabled',
  'invalid_key',
  'invalid_token',
  'insufficient_scope',
  'user_disabled',
  'no_credentials',
];
const modelTypeValues = ['LLM', 'embedding', 'rerank', 'image', 'video', 'audio'];

const toDash = (value: unknown) => {
  if (value === undefined || value === null || value === '') return '-';
  return String(value);
};

const padDatePart = (value: number) => String(value).padStart(2, '0');

const formatAuditTime = (value?: string) => {
  if (!value) return '-';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return [
    `${date.getFullYear()}-${padDatePart(date.getMonth() + 1)}-${padDatePart(date.getDate())}`,
    `${padDatePart(date.getHours())}:${padDatePart(date.getMinutes())}:${padDatePart(
      date.getSeconds()
    )}`,
  ].join(' ');
};

function statusTone(status?: string) {
  switch (status) {
    case 'success':
      return 'border-emerald-500/30 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300';
    case 'error':
    case 'denied':
    case 'invalid_key':
    case 'invalid_token':
    case 'insufficient_scope':
    case 'user_disabled':
    case 'no_credentials':
      return 'border-rose-500/30 bg-rose-500/15 text-rose-700 dark:text-rose-300';
    case 'model_not_found':
    case 'ip_banned':
    case 'key_banned':
      return 'border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-300';
    default:
      return 'border-slate-500/30 bg-slate-500/10 text-slate-700 dark:text-slate-300';
  }
}

function categoryTone(category?: string) {
  switch (category) {
    case 'inference':
      return 'border-blue-500/30 bg-blue-500/15 text-blue-700 dark:text-blue-300';
    case 'auth':
      return 'border-violet-500/30 bg-violet-500/15 text-violet-700 dark:text-violet-300';
    case 'admin':
      return 'border-cyan-500/30 bg-cyan-500/15 text-cyan-700 dark:text-cyan-300';
    default:
      return 'border-slate-500/30 bg-slate-500/10 text-slate-700 dark:text-slate-300';
  }
}

export default function AuditCenter() {
  const { t } = useI18n();
  const [records, setRecords] = useState<AuditRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState<TimeRangeValue>(DEFAULT_LOG_TIME_RANGE);
  const [filters, setFilters] = useState<AuditFilters>(defaultFilters);
  const [draftFilters, setDraftFilters] = useState(defaultTextFilters);
  const [pageFrom, setPageFrom] = useState(0);
  const [selectedRecord, setSelectedRecord] = useState<AuditRecord | null>(null);
  const requestSeqRef = useRef(0);

  const queryParams = useMemo(() => {
    const params = new URLSearchParams();
    params.set('time_from', timeRange.from);
    params.set('time_to', timeRange.to);
    params.set('page_from', String(pageFrom));
    params.set('size', String(AUDIT_PAGE_SIZE));

    Object.entries(filters).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        if (value.length) params.set(filterParamMap[key as AuditFilterKey], value.join(','));
        return;
      }
      if (value) params.set(filterParamMap[key as AuditFilterKey], value);
    });

    return params;
  }, [filters, pageFrom, timeRange]);

  const fetchAuditRecords = useCallback(async () => {
    const seq = ++requestSeqRef.current;
    setLoading(true);
    try {
      const data = await request.get<AuditSearchResponse>(
        '/v1/audit/search?' + queryParams.toString()
      );

      if (seq === requestSeqRef.current) {
        setRecords(Array.isArray(data.hits) ? data.hits : []);
        setTotal(data.total || 0);
      }
    } catch {
      if (seq === requestSeqRef.current) {
        setRecords([]);
        setTotal(0);
      }
    } finally {
      if (seq === requestSeqRef.current) {
        setLoading(false);
      }
    }
  }, [queryParams]);

  useEffect(() => {
    fetchAuditRecords();
  }, [fetchAuditRecords]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setFilters((current) => {
        const hasChanged = (Object.keys(defaultTextFilters) as AuditTextFilterKey[]).some(
          (key) => current[key] !== draftFilters[key]
        );

        if (!hasChanged) return current;

        setPageFrom(0);
        return { ...current, ...draftFilters };
      });
    }, 500);

    return () => clearTimeout(timer);
  }, [draftFilters]);

  const stats = useMemo(() => {
    const successCount = records.filter((record) => record.status === 'success').length;
    const riskCount = records.filter(
      (record) => record.status && record.status !== 'success'
    ).length;
    const apiKeyCount = records.filter((record) => record.auth_type === 'api_key').length;

    return [
      {
        label: t('auditCenter.totalEvents'),
        value: total,
        detail: t('auditCenter.matchingRecords'),
        Icon: FileSearch,
        tone: 'bg-sky-500/10 text-sky-600',
      },
      {
        label: t('auditCenter.successEvents'),
        value: successCount,
        detail: t('auditCenter.currentPage'),
        Icon: ShieldCheck,
        tone: 'bg-emerald-500/10 text-emerald-600',
      },
      {
        label: t('auditCenter.riskEvents'),
        value: riskCount,
        detail: t('auditCenter.currentPage'),
        Icon: CircleAlert,
        tone: 'bg-rose-500/10 text-rose-600',
      },
      {
        label: t('auditCenter.apiKeyEvents'),
        value: apiKeyCount,
        detail: t('auditCenter.currentPage'),
        Icon: KeyRound,
        tone: 'bg-violet-500/10 text-violet-600',
      },
    ];
  }, [records, t, total]);

  const setTextFilter = (key: AuditTextFilterKey, value: string) => {
    setDraftFilters((current) => ({ ...current, [key]: value }));
  };

  const setFilter = (key: Exclude<AuditFilterKey, AuditTextFilterKey>, value?: string | number) => {
    setFilters((current) => ({ ...current, [key]: String(value || '') }));
    setPageFrom(0);
  };

  const setMultiFilter = (key: AuditMultiFilterKey, value: string[]) => {
    setFilters((current) => ({ ...current, [key]: value }));
    setPageFrom(0);
  };

  const resetFilters = () => {
    setDraftFilters(defaultTextFilters);
    setFilters(defaultFilters);
    setPageFrom(0);
  };

  const getOptionLabel = (prefix: string, value: string | undefined, knownValues: string[]) => {
    if (!value) return t(`${prefix}.unknown`);
    return knownValues.includes(value) ? t(`${prefix}.${value}`) : value;
  };

  const hasFilters = Object.values(filters).some((value) =>
    Array.isArray(value) ? value.length > 0 : Boolean(value)
  );
  const hasDraftFilters = Object.values(draftFilters).some(Boolean);
  const maxAccessibleTotal = Math.min(total, 10000);
  const currentPage = Math.floor(pageFrom / AUDIT_PAGE_SIZE) + 1;
  const totalPages = Math.ceil(maxAccessibleTotal / AUDIT_PAGE_SIZE) || 1;

  return (
    <PageContainer
      title={t('menu.auditCenter')}
      subTitle={t('auditCenter.pageDescription')}
      extraContent={
        <div className="flex items-center gap-2">
          <TimeRangePicker
            value={timeRange}
            onChange={(value) => {
              setTimeRange(value);
              setPageFrom(0);
            }}
          />
          <Button variant="outline" onClick={fetchAuditRecords} disabled={loading}>
            <RefreshCw className={cn('mr-2 h-4 w-4', loading && 'animate-spin')} />
            {t('auditCenter.refresh')}
          </Button>
        </div>
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

      <div className="mt-6 rounded-lg border bg-card">
        <div className="border-b p-5">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 font-semibold">
              <Filter className="h-4 w-4 text-muted-foreground" />
              {t('auditCenter.filters')}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={resetFilters}
              disabled={!hasFilters && !hasDraftFilters}
            >
              <RotateCcw className="mr-2 h-4 w-4" />
              {t('auditCenter.resetFilters')}
            </Button>
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
            <FilterInput
              label={t('auditCenter.user')}
              value={draftFilters.user}
              placeholder={t('auditCenter.userPlaceholder')}
              onChange={(value) => setTextFilter('user', value)}
            />
            <FilterInput
              label={t('auditCenter.apiKeyName')}
              value={draftFilters.apiKeyName}
              placeholder={t('auditCenter.apiKeyNamePlaceholder')}
              onChange={(value) => setTextFilter('apiKeyName', value)}
            />
            <FilterInput
              label={t('auditCenter.modelId')}
              value={draftFilters.modelId}
              placeholder={t('auditCenter.modelIdPlaceholder')}
              onChange={(value) => setTextFilter('modelId', value)}
            />
            <FilterInput
              label={t('auditCenter.modelName')}
              value={draftFilters.modelName}
              placeholder={t('auditCenter.modelNamePlaceholder')}
              onChange={(value) => setTextFilter('modelName', value)}
            />
            <FilterInput
              label={t('auditCenter.clientIp')}
              value={draftFilters.clientIp}
              placeholder={t('auditCenter.clientIpPlaceholder')}
              onChange={(value) => setTextFilter('clientIp', value)}
            />
            <FilterSelect
              label={t('auditCenter.authType')}
              value={filters.authType}
              values={authTypeValues}
              labelPrefix="auditCenter.authTypeOptions"
              placeholder={t('auditCenter.allAuthTypes')}
              onChange={(value) => setFilter('authType', value)}
            />
            <FilterMultiSelect
              label={t('auditCenter.status')}
              value={filters.status}
              values={statusValues}
              labelPrefix="auditCenter.statusOptions"
              placeholder={t('auditCenter.allStatuses')}
              onChange={(value) => setMultiFilter('status', value)}
            />
            <FilterMultiSelect
              label={t('auditCenter.category')}
              value={filters.category}
              values={categoryValues}
              labelPrefix="auditCenter.categoryOptions"
              placeholder={t('auditCenter.allCategories')}
              onChange={(value) => setMultiFilter('category', value)}
            />
            <FilterMultiSelect
              label={t('auditCenter.modelType')}
              value={filters.modelType}
              values={modelTypeValues}
              labelPrefix="auditCenter.modelTypeOptions"
              placeholder={t('auditCenter.allModelTypes')}
              onChange={(value) => setMultiFilter('modelType', value)}
            />
          </div>
        </div>

        <div className="overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-48">{t('auditCenter.time')}</TableHead>
                <TableHead className="w-28">{t('auditCenter.category')}</TableHead>
                <TableHead className="w-44">{t('auditCenter.identity')}</TableHead>
                <TableHead>{t('auditCenter.endpoint')}</TableHead>
                <TableHead className="w-44">{t('auditCenter.model')}</TableHead>
                <TableHead className="w-32">{t('auditCenter.status')}</TableHead>
                <TableHead className="w-28 text-right">{t('auditCenter.latency')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={7} className="py-20 text-center text-muted-foreground">
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {t('auditCenter.loading')}
                    </div>
                  </TableCell>
                </TableRow>
              ) : records.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="py-20 text-center text-muted-foreground">
                    {t('auditCenter.noRecords')}
                  </TableCell>
                </TableRow>
              ) : (
                records.map((record, index) => (
                  <TableRow
                    key={(record['@timestamp'] || '') + '-' + (record.endpoint || '') + '-' + index}
                    className="cursor-pointer"
                    onClick={() => setSelectedRecord(record)}
                  >
                    <TableCell className="whitespace-nowrap text-sm">
                      {formatAuditTime(record['@timestamp'])}
                    </TableCell>
                    <TableCell>
                      <Badge className={categoryTone(record.category)}>
                        {getOptionLabel(
                          'auditCenter.categoryOptions',
                          record.category,
                          categoryValues
                        )}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex min-w-0 items-center gap-2">
                        {record.auth_type === 'api_key' ? (
                          <KeyRound className="h-4 w-4 shrink-0 text-muted-foreground" />
                        ) : (
                          <UserRound className="h-4 w-4 shrink-0 text-muted-foreground" />
                        )}
                        <div className="min-w-0">
                          <div className="truncate font-medium">
                            {record.user || record.api_key_name || '-'}
                          </div>
                          <div className="truncate text-xs text-muted-foreground">
                            {record.auth_type || '-'}
                            {record.client_ip ? ` · ${record.client_ip}` : ''}
                          </div>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell className="max-w-0">
                      <div className="truncate font-mono text-xs">{toDash(record.endpoint)}</div>
                      <div className="truncate text-xs text-muted-foreground">
                        {toDash(record.node || record.address)}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="truncate">{record.model_name || record.model_id || '-'}</div>
                      <div className="truncate text-xs text-muted-foreground">
                        {record.model_type || '-'}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge className={statusTone(record.status)}>
                        {getOptionLabel('auditCenter.statusOptions', record.status, statusValues)}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {typeof record.latency_ms === 'number'
                        ? t('auditCenter.latencyMs', { value: Math.round(record.latency_ms) })
                        : '-'}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        <div className="flex min-h-12 flex-wrap items-center justify-end gap-3 border-t px-4 py-2">
          <span className="text-sm text-muted-foreground">
            {t('auditCenter.totalHits', { count: total })}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={pageFrom === 0}
            onClick={() => setPageFrom(Math.max(0, pageFrom - AUDIT_PAGE_SIZE))}
          >
            {t('auditCenter.prevPage')}
          </Button>
          <span className="text-sm">
            {currentPage} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={pageFrom + AUDIT_PAGE_SIZE >= maxAccessibleTotal}
            onClick={() => setPageFrom(pageFrom + AUDIT_PAGE_SIZE)}
          >
            {t('auditCenter.nextPage')}
          </Button>
        </div>
      </div>

      <Dialog
        open={selectedRecord !== null}
        onOpenChange={(open) => !open && setSelectedRecord(null)}
      >
        <DialogContent className="sm:max-w-4xl" onOpenAutoFocus={(event) => event.preventDefault()}>
          <DialogHeader>
            <DialogTitle>{t('auditCenter.detailTitle')}</DialogTitle>
          </DialogHeader>
          {selectedRecord && (
            <div className="space-y-4">
              <div className="grid gap-3 md:grid-cols-3">
                <DetailItem
                  label={t('auditCenter.time')}
                  value={formatAuditTime(selectedRecord['@timestamp'])}
                />
                <DetailItem label={t('auditCenter.user')} value={toDash(selectedRecord.user)} />
                <DetailItem
                  label={t('auditCenter.clientIp')}
                  value={toDash(selectedRecord.client_ip)}
                />
                <DetailItem
                  label={t('auditCenter.category')}
                  value={getOptionLabel(
                    'auditCenter.categoryOptions',
                    selectedRecord.category,
                    categoryValues
                  )}
                />
                <DetailItem
                  label={t('auditCenter.authType')}
                  value={toDash(selectedRecord.auth_type)}
                />
                <DetailItem label={t('auditCenter.status')} value={toDash(selectedRecord.status)} />
                <DetailItem
                  label={t('auditCenter.latency')}
                  value={toDash(selectedRecord.latency_ms)}
                />
                <DetailItem
                  label={t('auditCenter.endpoint')}
                  value={toDash(selectedRecord.endpoint)}
                  wide
                />
              </div>
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <div className="text-sm font-medium">{t('auditCenter.rawJson')}</div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => copyToClipboard(JSON.stringify(selectedRecord, null, 2))}
                  >
                    <Copy className="mr-2 h-4 w-4" />
                    {t('auditCenter.copyJson')}
                  </Button>
                </div>
                <JSONSyntaxHighlighter data={selectedRecord} className="max-h-[28rem]" />
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </PageContainer>
  );
}

interface FilterInputProps {
  label: string;
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
}

function FilterInput({ label, value, placeholder, onChange }: FilterInputProps) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <Input
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}

interface FilterSelectProps {
  label: string;
  value: string;
  values: string[];
  labelPrefix: string;
  placeholder: string;
  onChange: (value?: string | number) => void;
}

interface FilterMultiSelectProps {
  label: string;
  value: string[];
  values: string[];
  labelPrefix: string;
  placeholder: string;
  onChange: (value: string[]) => void;
}

function FilterMultiSelect({
  label,
  value,
  values,
  labelPrefix,
  placeholder,
  onChange,
}: FilterMultiSelectProps) {
  const { t } = useI18n();

  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <MultiSelect
        value={value}
        placeholder={placeholder}
        onChange={onChange}
        options={values.map((item) => ({
          value: item,
          label: t(`${labelPrefix}.${item}`),
        }))}
      />
    </div>
  );
}

function FilterSelect({
  label,
  value,
  values,
  labelPrefix,
  placeholder,
  onChange,
}: FilterSelectProps) {
  const { t } = useI18n();

  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <Select
        value={value}
        placeholder={placeholder}
        onChange={onChange}
        options={values.map((item) => ({
          value: item,
          label: t(`${labelPrefix}.${item}`),
        }))}
      />
    </div>
  );
}

function DetailItem({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={cn('rounded-md border bg-card p-3', wide && 'md:col-span-2')}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 break-all text-sm font-medium">{value}</div>
    </div>
  );
}
