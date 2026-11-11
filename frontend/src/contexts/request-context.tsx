'use client';

import { PropsWithChildren, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { eventBus } from '@/lib/event-bus';
import { RequestEvents } from '@/constants';
import { requestManager } from '@/lib/request-manager';
import { removeAuthTokens } from '@/lib/auth-storage';
import type { AuthBanNotification } from '@/lib/request';
import { useI18n } from '@/contexts/i18n-context';

import { requestErrorExtensions } from '@/lib/request-extensions';
import type { RequestErrorEvent, RequestErrorExtension } from '@/lib/request-extension-types';

function ExtensionListener({ extension }: { extension: RequestErrorExtension }) {
  const [error, setError] = useState<RequestErrorEvent | null>(null);
  const active = useRef<RequestErrorEvent | null>(null);

  useEffect(() => {
    const handle = (payload?: RequestErrorEvent) => {
      if (!payload) return;
      const current = {
        ...payload,
        release: () => {
          payload.release();
          if (active.current === current) {
            active.current = null;
            setError(null);
          }
        },
      };
      active.current = current;
      setError(current);
    };
    eventBus.on(extension.event, handle);
    return () => {
      eventBus.off(extension.event, handle);
      const pending = active.current;
      active.current = null;
      pending?.release();
    };
  }, [extension.event]);

  const Handler = extension.Handler;
  return <Handler error={error} />;
}

export default function RequestProvider({ children }: PropsWithChildren) {
  const router = useRouter();
  const { t } = useI18n();
  useEffect(() => {
    /** 401 */
    const handleUnauthorized = async (message?: string) => {
      if (message) toast.error(message);

      // clear token;
      removeAuthTokens();
      router.replace('/login');
      // restore lock
      setTimeout(() => {
        requestManager.reset401();
      }, 1000);
    };

    /** 403 */
    const handleForbidden = () => {
      toast.error('Server error: 403 No auth');
      // restore lock
      setTimeout(() => {
        // router.back();
        requestManager.reset403();
      }, 1000);
    };

    /** Server Error */
    const handleServerError = (message?: string) => {
      toast.error(message);
    };

    const handleAuthBanned = (ban?: AuthBanNotification) => {
      if (!ban) return;
      const message = t(
        ban.kind === 'ip' ? 'securitySettings.ipBannedMessage' : 'securitySettings.keyBannedMessage'
      );
      const remaining = ban.remainingSeconds
        ? ` ${t('securitySettings.banRetryAfter', { seconds: ban.remainingSeconds })}`
        : '';
      toast.error(message + remaining, { id: `auth-ban-${ban.kind}` });
    };

    eventBus.on(RequestEvents.UNAUTHORIZED, handleUnauthorized);

    eventBus.on(RequestEvents.FORBIDDEN, handleForbidden);

    eventBus.on(RequestEvents.SERVER_ERROR, handleServerError);
    eventBus.on(RequestEvents.AUTH_BANNED, handleAuthBanned);

    return () => {
      eventBus.off(RequestEvents.UNAUTHORIZED, handleUnauthorized);

      eventBus.off(RequestEvents.FORBIDDEN, handleForbidden);

      eventBus.off(RequestEvents.SERVER_ERROR, handleServerError);
      eventBus.off(RequestEvents.AUTH_BANNED, handleAuthBanned);
    };
  }, [router, t]);

  return (
    <>
      {requestErrorExtensions.map((extension) => (
        <ExtensionListener key={extension.event} extension={extension} />
      ))}
      {children}
    </>
  );
}
