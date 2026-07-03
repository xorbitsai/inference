'use client';

import { SquareArrowRightExit } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import { InfoTooltip } from '@/components/ui/tooltip';
import { LOGIN_PATH } from '@/constants';
import { useI18n } from '@/contexts/i18n-context';
import { removeTokenValue } from '@/lib/auth-token';
import { cn } from '@/lib/utils';

interface LoginOutProps {
  className?: string;
}

const LoginOut = ({ className }: LoginOutProps) => {
  const router = useRouter();
  const { t } = useI18n();
  const loginout = () => {
    removeTokenValue();
    sessionStorage.removeItem('refresh_token');
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
