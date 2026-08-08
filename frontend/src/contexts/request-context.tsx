'use client';

import { PropsWithChildren, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { eventBus } from '@/lib/event-bus';
import { RequestEvents } from '@/constants';
import { requestManager } from '@/lib/request-manager';
import { removeAuthTokens } from '@/lib/auth-storage';
import { translate as t } from '@/contexts/i18n-context';
import { copyToClipboard } from '@/lib/utils';
import type { ServerErrorPayload } from '@/lib/request';

export default function RequestProvider({ children }: PropsWithChildren) {
  const router = useRouter();
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
    const handleForbidden = (message?: string) => {
      toast.error(message || t('common.noAuth'));
      // restore lock
      setTimeout(() => {
        // router.back();
        requestManager.reset403();
      }, 1000);
    };

    /** Server Error */
    const handleServerError = (message?: string, payload?: ServerErrorPayload) => {
      const traceback = payload?.traceback;
      if (!traceback) {
        toast.error(message);
        return;
      }
      // A traceback is far too long for the default 4s auto-dismiss: keep it
      // up until dismissed and give the user a way to copy it.
      toast.error(message, {
        duration: Infinity,
        closeButton: true,
        description: (
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-xs">
            {traceback}
          </pre>
        ),
        action: {
          label: t('launchModel.copyError'),
          onClick: () => {
            void copyToClipboard(`${message ?? ''}\n\n${traceback}`);
          },
        },
      });
    };

    eventBus.on(RequestEvents.UNAUTHORIZED, handleUnauthorized);

    eventBus.on(RequestEvents.FORBIDDEN, handleForbidden);

    eventBus.on(RequestEvents.SERVER_ERROR, handleServerError);

    return () => {
      eventBus.off(RequestEvents.UNAUTHORIZED, handleUnauthorized);

      eventBus.off(RequestEvents.FORBIDDEN, handleForbidden);

      eventBus.off(RequestEvents.SERVER_ERROR, handleServerError);
    };
  }, [router]);

  return <>{children}</>;
}
