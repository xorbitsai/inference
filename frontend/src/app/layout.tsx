import type { Metadata } from 'next';
import { cookies, headers } from 'next/headers';
import { getBrandingFromEnv } from '@/lib/branding';
import { I18nProvider } from '@/contexts/i18n-context';
import RequestProvider from '@/contexts/request-context';
import { GlobalProvider } from '@/contexts/global-context';
import ThemeProvider from '@/contexts/theme-context';
import AppInit from '@/contexts/app-init';
import { LayoutContent } from '@/components/layout/layout-content';
import { Toaster } from '@/components/ui/sonner';
import type { Locale } from '@/types/common';
import { getApiUrl } from '@/lib/utils';
import './globals.css';

const branding = getBrandingFromEnv();

export const metadata: Metadata = {
  title: branding.appName,
  description: branding.description,
  icons: {
    icon: branding.logoPath,
    apple: branding.logoPath,
  },
};
const resolveInitialLocale = async (): Promise<Locale> => {
  try {
    const cookieStore = await cookies();
    const cookieLocale = cookieStore.get('app_locale')?.value;
    if (cookieLocale === 'en' || cookieLocale === 'zh') {
      return cookieLocale;
    }

    const headerStore = await headers();
    const acceptLanguage = headerStore.get('accept-language')?.toLowerCase() || '';
    return acceptLanguage.includes('zh') ? 'zh' : 'en';
  } catch {
    return 'en';
  }
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const initialLocale = await resolveInitialLocale();
  const apiUrl = getApiUrl();
  let clusterAuth = null;
  let clusterAuthError: string | null = null;
  try {
    const res = await fetch(apiUrl + '/v1/cluster/auth', {
      cache: 'no-store',
    });
    
    let data: any = null;
    try {
      data = await res.json();
    } catch {
      data = null;
    }
    if (!res.ok) {
      throw new Error(
        `Server error: ${res.status} - ${
          data?.detail || 'Unknown error'
        }`
      );
    }
    clusterAuth = data;
  } catch (error) {
    clusterAuthError = error instanceof Error ? error.message : 'Cluster auth failed';
  }

  return (
    <html lang={initialLocale} suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground" suppressHydrationWarning>
        <RequestProvider>
          <GlobalProvider
            initClusterAuth={clusterAuth}
          >
            <I18nProvider initialLocale={initialLocale}>
              <ThemeProvider>
                <AppInit clusterAuth={clusterAuth} clusterAuthError={clusterAuthError} />
                <LayoutContent>{children}</LayoutContent>
                <Toaster />
              </ThemeProvider>
            </I18nProvider>
          </GlobalProvider>
        </RequestProvider>
      </body>
    </html>
  );
}
