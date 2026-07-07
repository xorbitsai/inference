'use client';

import { PropsWithChildren, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { eventBus } from '@/lib/event-bus';
import { RequestEvents } from '@/constants';
import { requestManager } from '@/lib/request-manager';
import { removeAuthTokens } from '@/lib/auth-storage';

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
