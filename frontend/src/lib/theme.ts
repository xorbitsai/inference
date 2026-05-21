export type ThemeMode = 'light' | 'dark';

export interface ThemeColors {
  [key: string]: string;
}

export interface Theme {
  name: string;
  mode: ThemeMode;
  colors: ThemeColors;
}

export const themes: Record<ThemeMode, Theme> = {
  dark: {
    name: 'Dark',
    mode: 'dark',
    colors: {
      background: '222.2 84% 4.9%',
      foreground: '210 40% 98%',
      card: '222.2 84% 4.9%',
      cardForeground: '210 40% 98%',
      popover: '222.2 84% 4.9%',
      popoverForeground: '210 40% 98%',
      primary: '217.2 91.2% 59.8%',
      primaryForeground: '222.2 84% 4.9%',
      secondary: '217.2 32.6% 17.5%',
      secondaryForeground: '210 40% 98%',
      muted: '217.2 32.6% 17.5%',
      mutedForeground: '215 20.2% 65.1%',
      accent: '217.2 32.6% 17.5%',
      accentForeground: '210 40% 98%',
      destructive: '0 84.2% 60.2%',
      destructiveForeground: '210 40% 98%',
      border: '217.2 32.6% 17.5%',
      input: '217.2 32.6% 17.5%',
      ring: '224.3 76.3% 94.1%',
      gradientFrom: '217.2 91.2% 59.8%',
      gradientTo: '262 83% 58%',
      sidebarActiveBgFrom: '217.2 32.6% 17.5%',
      sidebarActiveBgTo: '217.2 32.6% 17.5%',
      sidebarActiveText: '210 40% 98%',
      sidebarActiveBorder: '217.2 91.2% 59.8%',
    },
  },

  light: {
    name: 'Light',
    mode: 'light',
    colors: {
      background: '0 0% 100%',
      foreground: '222.2 84% 4.9%',
      card: '0 0% 100%',
      cardForeground: '222.2 84% 4.9%',
      popover: '0 0% 100%',
      popoverForeground: '222.2 84% 4.9%',
      primary: '221.2 83.2% 53.3%',
      primaryForeground: '210 40% 98%',
      secondary: '210 40% 96%',
      secondaryForeground: '222.2 84% 4.9%',
      muted: '210 30% 95%',
      mutedForeground: '215.4 25% 40%',
      accent: '210 40% 96%',
      accentForeground: '222.2 84% 4.9%',
      destructive: '0 84.2% 60.2%',
      destructiveForeground: '210 40% 98%',
      border: '214.3 31.8% 91.4%',
      input: '214.3 31.8% 91.4%',
      ring: '221.2 83.2% 53.3%',
      gradientFrom: '252 100% 67%',
      gradientTo: '320 85% 60%',
      sidebarActiveBgFrom: '270 95% 95%',
      sidebarActiveBgTo: '270 50% 98%',
      sidebarActiveText: '270 95% 45%',
      sidebarActiveBorder: '270 95% 60%',
    },
  },
};
function camelToKebab(str: string) {
  return str.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
}
/** 
 * Currently only supports dark/light modes.
 * Will extend to other themes like blue/purple in the future, generated using applyTheme.
  */
export function applyTheme(themeName: ThemeMode) {
  const theme = themes[themeName];

  if (!theme) return;

  const root = document.documentElement;

  Object.entries(theme.colors).forEach(([key, value]) => {
    const cssVar = `--${camelToKebab(key)}`;

    root.style.setProperty(cssVar, value);
  });
}
