import type { Metadata } from 'next';
import { getBrandingFromEnv } from '@/lib/branding';
import { I18nProvider } from '@/contexts/i18n-context';
import RequestProvider from '@/contexts/request-context';
import { GlobalProvider } from '@/contexts/global-context';
import ThemeProvider from '@/contexts/theme-context';
import AppInit from '@/contexts/app-init';
import { LayoutContent } from '@/components/layout/layout-content';
import { Toaster } from '@/components/ui/sonner';
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

// No request-time cookies()/headers() here: the app is shipped as a static
// export served by the Xinference backend. The client restores the locale
// (localStorage/navigator) in I18nProvider and fetches the cluster auth mode
// in AppInit.
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground" suppressHydrationWarning>
        <RequestProvider>
          <GlobalProvider>
            <I18nProvider>
              <ThemeProvider>
                <AppInit />
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
