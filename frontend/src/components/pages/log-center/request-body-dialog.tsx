'use client';

import axios from 'axios';
import { Loader2, ShieldAlert } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';

import type { ModelRequestBodyResponse } from './types';

export function RequestBodyDialog({
  requestId,
  open,
  onOpenChange,
}: {
  requestId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const [data, setData] = useState<ModelRequestBodyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorKey, setErrorKey] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !requestId) return;
    let active = true;
    setLoading(true);
    setErrorKey(null);
    setData(null);
    request
      .get<ModelRequestBodyResponse>(
        `/v1/cluster/model-requests/${encodeURIComponent(requestId)}/body`
      )
      .then((body) => active && setData(body))
      .catch((error: unknown) => {
        if (!active) return;
        const status = axios.isAxiosError(error) ? error.response?.status : undefined;
        if (status === 403) {
          setErrorKey('logCenter.detail.requestBodyForbidden');
        } else if (status === 404) {
          setErrorKey('logCenter.detail.requestBodyNotFound');
        } else if (status === 502) {
          setErrorKey('logCenter.detail.requestBodyBackendError');
        } else if (status === 503) {
          setErrorKey('logCenter.detail.requestBodyNotConfigured');
        } else {
          setErrorKey('logCenter.detail.requestBodyFetchError');
        }
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [open, requestId]);

  const displayBody = useMemo(() => {
    if (!data) return '';
    if (data.request_body !== undefined) return JSON.stringify(data.request_body, null, 2);
    if (data.request_body_raw !== undefined) return data.request_body_raw;
    if (data.request_body_omitted !== undefined)
      return JSON.stringify(data.request_body_omitted, null, 2);
    return t('logCenter.detail.bodyUnavailable');
  }, [data, t]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="!max-w-4xl gap-0 p-0" maskClosable>
        <DialogHeader className="border-b px-5 py-4">
          <DialogTitle>{t('logCenter.detail.requestBodyTitle')}</DialogTitle>
        </DialogHeader>
        <div className="flex items-center gap-2 border-b bg-amber-50 px-5 py-3 text-xs text-amber-900 dark:bg-amber-950/20 dark:text-amber-100">
          <ShieldAlert className="size-4 shrink-0" />
          {t('logCenter.detail.requestBodyWarning')}
        </div>
        {loading && <Loader2 className="mx-auto my-16 size-7 animate-spin" />}
        {!loading && errorKey && (
          <div className="py-16 text-center text-destructive">{t(errorKey)}</div>
        )}
        {!loading && !errorKey && data && (
          <pre className="m-0 max-h-[65vh] overflow-auto whitespace-pre-wrap break-all p-5 font-mono text-xs">
            {displayBody}
          </pre>
        )}
      </DialogContent>
    </Dialog>
  );
}
