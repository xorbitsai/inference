import type { Metadata } from 'next';
import { cookies, headers } from 'next/headers';
import { getBrandingFromEnv } from '@/lib/branding';
import { I18nProvider } from '@/contexts/i18n-context';
import { ThemeProvider } from '@/contexts/theme-context';
import { LayoutContent } from '@/components/layout/layout-content';
import { Toaster } from '@/components/ui/sonner';
import { getThemeFromEnv, buildThemeStyle, themes } from '@/lib/theme';
import type { Locale } from '@/types/common';
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
  const themeName = getThemeFromEnv();
  const theme = themes[themeName] || themes.dark;
  const themeMode = theme.mode || 'light';
  const themeStyle = buildThemeStyle(themeName);

  return (
    <html lang={initialLocale} className={themeMode} style={themeStyle} suppressHydrationWarning>
      <body
        className={`antialiased bg-background text-foreground theme-${themeName}`}
        suppressHydrationWarning
      >
        <I18nProvider initialLocale={initialLocale}>
          <ThemeProvider>
            <LayoutContent>{children}</LayoutContent>
            <Toaster />
          </ThemeProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
