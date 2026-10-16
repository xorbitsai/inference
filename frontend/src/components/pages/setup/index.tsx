'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { CheckCircle2, Rocket, Network, Boxes } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { FormField } from '@/components/ui/form-field';
import { Form } from '@/components/ui/form';
import { useForm } from '@/hooks/use-form';
import { getApiUrl } from '@/lib/utils';
import { getBrandingFromEnv } from '@/lib/branding';
import { useI18n } from '@/contexts/i18n-context';
import { SETUP_COMPLETE_FLAG } from '@/constants';

// Real Xinference capabilities (see README.md "Key Features" / branding
// tagline), framed for someone who has just deployed the service and is
// about to create the first account.
const FEATURES = [
  {
    icon: Rocket,
    titleKey: 'setup.featureServingTitle',
    descKey: 'setup.featureServingDesc',
  },
  {
    icon: Network,
    titleKey: 'setup.featureApiTitle',
    descKey: 'setup.featureApiDesc',
  },
  {
    icon: Boxes,
    titleKey: 'setup.featureDistributedTitle',
    descKey: 'setup.featureDistributedDesc',
  },
];

const DEFAULT_PASSWORD_MIN_LENGTH = 8;

export default function Setup() {
  const router = useRouter();
  const [form] = useForm();
  const { t } = useI18n();
  const branding = getBrandingFromEnv();

  const [loading, setLoading] = useState(false);
  const [passwordMinLength, setPasswordMinLength] = useState(DEFAULT_PASSWORD_MIN_LENGTH);

  useEffect(() => {
    let cancelled = false;
    fetch(getApiUrl() + '/v1/admin/setup/status')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && typeof data?.password_min_length === 'number') {
          setPasswordMinLength(data.password_min_length);
        }
      })
      .catch(() => {
        // Keep the default; the backend will still enforce its own minimum.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const goToLogin = (setupComplete: boolean) => {
    if (setupComplete) {
      sessionStorage.setItem(SETUP_COMPLETE_FLAG, '1');
    }
    router.replace('/login');
  };

  const onSubmit = async (values: Record<string, unknown>) => {
    setLoading(true);
    try {
      // Plain fetch rather than the shared `request` client: this call
      // happens before any login, and the client's response interceptor
      // would otherwise surface a generic global "403 No auth" toast on
      // the race-loss path below instead of the specific message we want.
      const res = await fetch(getApiUrl() + '/v1/admin/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: values.username,
          password: values.password,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        if (res.status === 403 && data?.detail?.includes('already completed')) {
          // Someone else completed setup first; send this browser to the
          // normal login page instead of leaving it stuck here.
          goToLogin(false);
          return;
        }
        // Any other failure (e.g. password too short): show the specific
        // message and let the user retry on this page.
        toast.error(data?.detail || t('setup.createFailed'));
        return;
      }
      toast.success(t('setup.createSuccess'));
      goToLogin(true);
    } catch {
      toast.error(t('setup.createFailed'));
    } finally {
      setLoading(false);
    }
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
          <span className="text-2xl font-semibold tracking-tight">{branding.appName}</span>
        </div>

        <div className="relative space-y-6">
          <div className="space-y-4">
            <h1 className="text-4xl font-bold leading-tight">{t('setup.welcome')}</h1>
            <p className="text-blue-100/90 text-lg">{t('setup.welcomeDesc')}</p>
          </div>
          <div className="space-y-5">
            {FEATURES.map(({ icon: Icon, titleKey, descKey }) => (
              <div key={titleKey} className="flex gap-3">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-white/15">
                  <Icon className="h-4 w-4" />
                </span>
                <div>
                  <p className="font-semibold">{t(titleKey)}</p>
                  <p className="text-sm text-blue-100/80">{t(descKey)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <p className="relative text-sm text-blue-100/70">
          © {new Date().getFullYear()} {branding.appName}. All rights reserved.
        </p>
      </div>

      {/* Right form panel */}
      <div className="flex items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-md space-y-8">
          <div className="space-y-2">
            <h2 className="text-3xl font-bold text-primary">{t('setup.title')}</h2>
            <p className="text-sm text-muted-foreground">{t('setup.description')}</p>
          </div>
          <Form onFinish={onSubmit} form={form} initialValues={{ username: 'admin' }}>
            <FormField name="username" label={t('login.username')} rules={[{ required: true }]}>
              <Input />
            </FormField>
            <FormField
              name="password"
              label={t('login.password')}
              placeholder={t('userManagement.passwordPlaceholder')}
              rules={[
                { required: true, message: t('userManagement.passwordRequired') },
                {
                  validator: (val: unknown) =>
                    typeof val === 'string' && val.length >= passwordMinLength,
                  message: t('setup.passwordTooShort', { min: passwordMinLength }),
                },
              ]}
            >
              <Input type="password" />
            </FormField>
            <FormField
              name="confirm_password"
              label={t('userManagement.confirmPassword')}
              placeholder={t('userManagement.confirmPasswordPlaceholder')}
              rules={[
                { required: true, message: t('userManagement.passwordRequired') },
                {
                  validator: (val: unknown) => val === form.getFieldValue('password'),
                  message: t('userManagement.passwordMismatch'),
                },
              ]}
            >
              <Input type="password" />
            </FormField>
            <Button block className="!mt-5" type="submit" loading={loading}>
              {t('setup.submit')}
            </Button>
          </Form>
          <div className="flex items-center gap-2 text-sm text-muted-foreground lg:hidden">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <span>{t('setup.adminAccessNote')}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
