'use client';

import { SquareArrowRightExit } from 'lucide-react';
import { useRouter } from 'next/navigation';
import Cookies from 'js-cookie';
import { toast } from 'sonner';

import { LOGIN_PATH } from '@/constants';
import { useI18n } from '@/contexts/i18n-context';

const LoginOut = () => {
  const router = useRouter();
  const { t } = useI18n();
  const loginout = () => {
    Cookies.remove('token');
    toast.success(t('common.loginOutSuccess'))
    router.replace(LOGIN_PATH);
  };
  return (
    <SquareArrowRightExit
      onClick={loginout}
      className="h-5 w-5 text-muted-foreground hover:text-foreground cursor-pointer"
    />
  );
};
export default LoginOut;
