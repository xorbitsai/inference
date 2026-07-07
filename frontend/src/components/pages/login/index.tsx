'use client';

import { useState } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { FormField } from '@/components/ui/form-field';
import { Form } from '@/components/ui/form';
import { useForm } from '@/hooks/use-form';
import request from '@/lib/request';
import { decodeJwtPayload } from '@/lib/utils';
import { setAccessToken, setRefreshToken } from '@/lib/auth-storage';
import { getBrandingFromEnv } from '@/lib/branding';
import { useI18n } from '@/contexts/i18n-context';
import { useGlobal } from '@/contexts/global-context';
import { ChangePasswordDialog } from './change-password-dialog';

interface TokenResponse {
  access_token?: string;
  refresh_token?: string;
  token_type?: string;
  must_change_password?: boolean;
}

export default function LoginPage() {
  const router = useRouter();
  const [form] = useForm();
  const { fetchGlobalAfterAuth } = useGlobal();

  const { t } = useI18n();
  const branding = getBrandingFromEnv();

  const [loading, setLoading] = useState(false);
  const [changePasswordOpen, setChangePasswordOpen] = useState(false);
  const [changePasswordUserId, setChangePasswordUserId] = useState<number | null>(null);

  const goHomeAfterAuth = () => {
    fetchGlobalAfterAuth();
    setTimeout(() => router.push('/'), 0);
  };

  const onSubmit = async (values: Record<string, unknown>) => {
    setLoading(true);
    try {
      const res = await request.post<TokenResponse>('/token', values);
      if (res?.access_token) {
        toast.success(t('common.loginSuccess'));

        setAccessToken(res.access_token);

        if (res.refresh_token) {
          setRefreshToken(res.refresh_token);
        }

        if (res.must_change_password) {
          const payload = decodeJwtPayload(res.access_token);
          const userId = Number(payload?.user_id);

          if (!Number.isFinite(userId)) {
            toast.error(t('login.changePasswordUserMissing'));
            return;
          }
          setChangePasswordUserId(userId);
          setChangePasswordOpen(true);
          return;
        }
        goHomeAfterAuth();
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordChanged = () => {
    setChangePasswordOpen(false);
    setChangePasswordUserId(null);
    goHomeAfterAuth();
  };

  return (
    <div className="min-h-screen grid lg:grid-cols-2 bg-background">
      {/* Left brand section */}
      <div className="hidden lg:flex flex-col justify-between p-12 bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 text-white relative overflow-hidden">
        <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute -bottom-32 -left-16 w-96 h-96 rounded-full bg-indigo-400/20 blur-3xl" />
        <div className="relative flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/95 shadow-sm ring-1 ring-white/40">
            <Image
              src={branding.logoPath}
              width={28}
              height={28}
              alt={branding.logoAlt}
              className="h-7 w-7"
            />
          </span>
          <span className="text-2xl font-semibold tracking-tight">Xinference</span>
        </div>

        <div className="relative space-y-4">
          <h1 className="text-4xl font-bold leading-tight">{t('login.welcome')}</h1>
          <p className="text-blue-100/90 text-lg">{t('login.welcomeDesc')}</p>
        </div>

        <p className="relative text-sm text-blue-100/70">
          © {new Date().getFullYear()} Xinference. All rights reserved.
        </p>
      </div>

      {/* Right form panel */}
      <div className="flex items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-md space-y-8">
          <h2 className="text-3xl font-bold text-primary">{t('login.login')}</h2>
          <Form onFinish={onSubmit} form={form}>
            <FormField name="username" label={t('login.username')} rules={[{ required: true }]}>
              <Input />
            </FormField>
            <FormField name="password" label={t('login.password')} rules={[{ required: true }]}>
              <Input type="password" />
            </FormField>
            <Button block className="!mt-5" type="submit" loading={loading}>
              {t('login.login')}
            </Button>
          </Form>
        </div>
      </div>

      <ChangePasswordDialog
        open={changePasswordOpen}
        userId={changePasswordUserId}
        onChanged={handlePasswordChanged}
      />
    </div>
  );
}
