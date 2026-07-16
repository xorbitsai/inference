'use client';

import { SquareArrowRightExit } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import { InfoTooltip } from '@/components/ui/tooltip';
import { LOGIN_PATH } from '@/constants';
import { useI18n } from '@/contexts/i18n-context';
import { getRefreshToken, removeAuthTokens } from '@/lib/auth-storage';
import request from '@/lib/request';
import { cn } from '@/lib/utils';

interface LoginOutProps {
  className?: string;
}

const LoginOut = ({ className }: LoginOutProps) => {
  const router = useRouter();
  const { t } = useI18n();
  const loginout = async () => {
    const refreshToken = getRefreshToken();
    removeAuthTokens();
    if (refreshToken) {
      try {
        await request.post('/v1/auth/logout', { refresh_token: refreshToken });
      } catch {
        // Local logout should still complete if server-side token cleanup fails.
      }
    }
    toast.success(t('common.loginOutSuccess'));
    router.replace(LOGIN_PATH);
  };

  return (
    <InfoTooltip content={t('common.loginOut')}>
      <button
        type="button"
        onClick={loginout}
        className={cn('text-muted-foreground hover:text-foreground transition-colors', className)}
        aria-label={t('common.loginOut')}
      >
        <SquareArrowRightExit className="h-5 w-5" />
      </button>
    </InfoTooltip>
  );
};
export default LoginOut;
