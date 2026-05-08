import type { CSSProperties } from "react";

export type ThemeMode = 'dark' | 'light' | 'system';

export interface ThemeColors {
  background: string;
  foreground: string;
  card: string;
  cardForeground: string;
  popover: string;
  popoverForeground: string;
  primary: string;
  primaryForeground: string;
  secondary: string;
  secondaryForeground: string;
  muted: string;
  mutedForeground: string;
  accent: string;
  accentForeground: string;
  destructive: string;
  destructiveForeground: string;
  border: string;
  input: string;
  ring: string;
  // Extended colors for Cyber theme
  cardHover?: string;
  borderHighlight?: string;
  accentBg?: string;
  accentBorder?: string;
  shadowColor?: string;
  // Gradient text colors
  gradientFrom?: string;
  gradientTo?: string;
  // Sidebar active state colors
  sidebarActiveBgFrom?: string;
  sidebarActiveBgTo?: string;
  sidebarActiveText?: string;
  sidebarActiveBorder?: string;
}

export interface Theme {
  name: string;
  mode: ThemeMode;
  colors: ThemeColors;
}

export const themes: Record<string, Theme> = {
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
      // Extended colors
      gradientFrom: '217.2 91.2% 59.8%', // Primary blue
      gradientTo: '262 83% 58%', // Purple
      sidebarActiveBgFrom: '217.2 32.6% 17.5%', // muted/accent
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
      // Extended colors
      gradientFrom: '252 100% 67%', // Purple-400
      gradientTo: '320 85% 60%', // Pink/Magenta
      sidebarActiveBgFrom: '270 95% 95%', // Purple-50/100 equivalent
      sidebarActiveBgTo: '270 50% 98%', // Purple-50/0 equivalent
      sidebarActiveText: '270 95% 45%', // Purple-700
      sidebarActiveBorder: '270 95% 60%', // Purple-600
    },
  },
  blue: {
    name: 'Blue',
    mode: 'dark',
    colors: {
      background: '215 27% 17%', // Dark blue background
      foreground: '210 40% 98%',
      card: '215 25% 12%', // Slightly lighter blue card
      cardForeground: '210 40% 98%',
      popover: '215 27% 17%',
      popoverForeground: '210 40% 98%',
      primary: '217 91% 60%', // Bright blue primary color
      primaryForeground: '222.2 84% 4.9%',
      secondary: '215 25% 20%', // Blue secondary color
      secondaryForeground: '210 40% 98%',
      muted: '215 20% 25%', // Blue muted color
      mutedForeground: '215 20% 70%',
      accent: '215 25% 30%', // Blue accent color
      accentForeground: '210 40% 98%',
      destructive: '0 84.2% 60.2%',
      destructiveForeground: '210 40% 98%',
      border: '215 25% 25%', // Blue border
      input: '215 25% 25%',
      ring: '217 91% 60%',
      // Gradient text colors
      gradientFrom: '217 91% 60%', // Primary Blue
      gradientTo: '190 90% 50%', // Cyan
      // Sidebar active state colors
      sidebarActiveBgFrom: '217 91% 60% / 0.2',
      sidebarActiveBgTo: '217 91% 60% / 0',
      sidebarActiveText: '217 91% 60%',
      sidebarActiveBorder: '217 91% 60%',
    },
  },
  green: {
    name: 'Green',
    mode: 'dark',
    colors: {
      background: '142 28% 15%', // Dark green background
      foreground: '210 40% 98%',
      card: '142 25% 10%', // Slightly lighter green card
      cardForeground: '210 40% 98%',
      popover: '142 28% 15%',
      popoverForeground: '210 40% 98%',
      primary: '142 76% 36%', // Green primary color
      primaryForeground: '355.7 100% 97.3%',
      secondary: '142 25% 18%', // Green secondary color
      secondaryForeground: '210 40% 98%',
      muted: '142 20% 22%', // Green muted color
      mutedForeground: '142 20% 75%',
      accent: '142 25% 25%', // Green accent color
      accentForeground: '210 40% 98%',
      destructive: '0 84.2% 60.2%',
      destructiveForeground: '210 40% 98%',
      border: '142 25% 22%', // Green border
      input: '142 25% 22%',
      ring: '142 76% 36%',
      // Gradient text colors
      gradientFrom: '142 76% 36%', // Primary Green
      gradientTo: '100 80% 50%', // Lime
      // Sidebar active state colors
      sidebarActiveBgFrom: '142 76% 36% / 0.2',
      sidebarActiveBgTo: '142 76% 36% / 0',
      sidebarActiveText: '142 76% 36%',
      sidebarActiveBorder: '142 76% 36%',
    },
  },
  purple: {
    name: 'Purple',
    mode: 'dark',
    colors: {
      background: '262 28% 17%', // Dark purple background
      foreground: '210 40% 98%',
      card: '262 25% 12%', // Slightly lighter purple card
      cardForeground: '210 40% 98%',
      popover: '262 28% 17%',
      popoverForeground: '210 40% 98%',
      primary: '262 83% 58%', // Purple primary color
      primaryForeground: '210 40% 98%',
      secondary: '262 25% 20%', // Purple secondary color
      secondaryForeground: '210 40% 98%',
      muted: '262 20% 25%', // Purple muted color
      mutedForeground: '262 20% 70%',
      accent: '262 25% 30%', // Purple accent color
      accentForeground: '210 40% 98%',
      destructive: '0 84.2% 60.2%',
      destructiveForeground: '210 40% 98%',
      border: '262 25% 25%', // Purple border
      input: '262 25% 25%',
      ring: '262 83% 58%',
      // Gradient text colors
      gradientFrom: '262 83% 58%', // Primary Purple
      gradientTo: '300 80% 60%', // Pink
      // Sidebar active state colors
      sidebarActiveBgFrom: '262 83% 58% / 0.2',
      sidebarActiveBgTo: '262 83% 58% / 0',
      sidebarActiveText: '262 83% 58%',
      sidebarActiveBorder: '262 83% 58%',
    },
  },
  cyber: {
    name: 'Cyber',
    mode: 'dark',
    colors: {
      background: '222 47% 11%', // #0F172A - Dark blue-black background
      foreground: '0 0% 100%', // #FFFFFF - White text
      card: '217 33% 17%', // #1E293B - Card background
      cardForeground: '0 0% 100%',
      popover: '222 47% 11%',
      popoverForeground: '0 0% 100%',
      primary: '180 100% 50%', // #00F0FF - Cyan primary color (Cyber cyan)
      primaryForeground: '222 47% 11%',
      secondary: '217 33% 17%', // #1E293B - Secondary color
      secondaryForeground: '210 40% 98%',
      muted: '215 25% 27%', // #334155 - Muted color
      mutedForeground: '215 16% 47%', // #64748B
      accent: '180 100% 50%', // #00F0FF - Cyan accent color
      accentForeground: '222 47% 11%',
      destructive: '0 84% 60%', // Red
      destructiveForeground: '0 0% 100%',
      border: '217 33% 17%', // #1E293B - Border color
      input: '222 47% 7%', // #020617 - Input background
      ring: '180 100% 50%', // #00F0FF - Focus ring
      // Extended colors
      cardHover: '215 25% 27%', // #334155 - Card hover color
      borderHighlight: '215 25% 41%', // #CBD5E1 - Highlight border
      accentBg: '180 100% 50% / 0.1', // Cyan background (transparent)
      accentBorder: '180 100% 50% / 0.2', // Cyan border (transparent)
      shadowColor: '0 0% 0% / 0.5', // Shadow color
      // Gradient text colors
      gradientFrom: '180 100% 50%', // Cyan
      gradientTo: '280 100% 70%', // Neon Purple
      // Sidebar active state colors
      sidebarActiveBgFrom: '180 100% 50% / 0.2',
      sidebarActiveBgTo: '180 100% 50% / 0',
      sidebarActiveText: '180 100% 50%',
      sidebarActiveBorder: '180 100% 50%',
    },
  },
  cyberLight: {
    name: 'Cyber Light',
    mode: 'light',
    colors: {
      background: '210 40% 98%', // #F8FAFC - Light gray background
      foreground: '222 47% 11%', // #0F172A - Dark text
      card: '0 0% 100%', // #FFFFFF - White card
      cardForeground: '222 47% 11%',
      popover: '210 40% 98%',
      popoverForeground: '222 47% 11%',
      primary: '200 98% 39%', // #0284C7 - Blue primary color (Sky-600)
      primaryForeground: '0 0% 100%',
      secondary: '210 40% 96%', // Light gray secondary color
      secondaryForeground: '222 84% 4.9%',
      muted: '210 30% 95%', // Light gray muted color
      mutedForeground: '215 16% 47%', // #64748B
      accent: '200 98% 39%', // #0284C7 - Blue accent color
      accentForeground: '0 0% 100%',
      destructive: '0 84% 60%',
      destructiveForeground: '210 40% 98%',
      border: '214 31% 91%', // #E2E8F0 - Light border
      input: '210 40% 96%', // #F1F5F9 - Input background
      ring: '200 98% 39%', // Blue focus ring
      // Extended colors
      cardHover: '210 40% 96%', // #F1F5F9 - Card hover color
      borderHighlight: '213 27% 84%', // #CBD5E1 - Highlight border
      accentBg: '200 98% 39% / 0.1', // Blue background (transparent)
      accentBorder: '200 98% 39% / 0.2', // Blue border (transparent)
      shadowColor: '0 0% 0% / 0.05', // Shadow color
      // Gradient text colors
      gradientFrom: '200 98% 39%', // Sky-600
      gradientTo: '180 100% 50%', // Cyan/Teal
      // Sidebar active state colors
      sidebarActiveBgFrom: '200 98% 39% / 0.15', // Sky-600 with opacity
      sidebarActiveBgTo: '200 98% 39% / 0',
      sidebarActiveText: '200 98% 39%', // Sky-600
      sidebarActiveBorder: '200 98% 39%', // Sky-600
    },
  },
};

export function getThemeFromEnv(): string {
  return process.env.NEXT_PUBLIC_THEME || 'light';
}

export function applyTheme(themeName: string): void {
  const theme = themes[themeName] || themes.dark;
  const root = document.documentElement;

  // Apply CSS custom properties
  root.style.setProperty('--background', theme.colors.background);
  root.style.setProperty('--foreground', theme.colors.foreground);
  root.style.setProperty('--card', theme.colors.card);
  root.style.setProperty('--card-foreground', theme.colors.cardForeground);
  root.style.setProperty('--popover', theme.colors.popover);
  root.style.setProperty('--popover-foreground', theme.colors.popoverForeground);
  root.style.setProperty('--primary', theme.colors.primary);
  root.style.setProperty('--primary-foreground', theme.colors.primaryForeground);
  root.style.setProperty('--secondary', theme.colors.secondary);
  root.style.setProperty('--secondary-foreground', theme.colors.secondaryForeground);
  root.style.setProperty('--muted', theme.colors.muted);
  root.style.setProperty('--muted-foreground', theme.colors.mutedForeground);
  root.style.setProperty('--accent', theme.colors.accent);
  root.style.setProperty('--accent-foreground', theme.colors.accentForeground);
  root.style.setProperty('--destructive', theme.colors.destructive);
  root.style.setProperty('--destructive-foreground', theme.colors.destructiveForeground);
  root.style.setProperty('--border', theme.colors.border);
  root.style.setProperty('--input', theme.colors.input);
  root.style.setProperty('--ring', theme.colors.ring);

  // Apply extended color properties if available
  if (theme.colors.cardHover) {
    root.style.setProperty('--card-hover', theme.colors.cardHover);
  }
  if (theme.colors.borderHighlight) {
    root.style.setProperty('--border-highlight', theme.colors.borderHighlight);
  }
  if (theme.colors.accentBg) {
    root.style.setProperty('--accent-bg', theme.colors.accentBg);
  }
  if (theme.colors.accentBorder) {
    root.style.setProperty('--accent-border', theme.colors.accentBorder);
  }
  if (theme.colors.shadowColor) {
    root.style.setProperty('--shadow-color', theme.colors.shadowColor);
  }

  // Apply gradient text colors
  if (theme.colors.gradientFrom) {
    root.style.setProperty('--gradient-from', theme.colors.gradientFrom);
  }
  if (theme.colors.gradientTo) {
    root.style.setProperty('--gradient-to', theme.colors.gradientTo);
  }

  // Apply sidebar active state colors
  if (theme.colors.sidebarActiveBgFrom) {
    root.style.setProperty('--sidebar-active-bg-from', theme.colors.sidebarActiveBgFrom);
  }
  if (theme.colors.sidebarActiveBgTo) {
    root.style.setProperty('--sidebar-active-bg-to', theme.colors.sidebarActiveBgTo);
  }
  if (theme.colors.sidebarActiveText) {
    root.style.setProperty('--sidebar-active-text', theme.colors.sidebarActiveText);
  }
  if (theme.colors.sidebarActiveBorder) {
    root.style.setProperty('--sidebar-active-border', theme.colors.sidebarActiveBorder);
  }

  // Apply theme class to body
  const body = document.body;
  body.className = body.className.replace(/\b(theme-\w+)\b/g, '');
  body.classList.add(`theme-${themeName}`);

  // Apply dark/light mode
  if (theme.mode === 'dark') {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
}

export function buildThemeStyle (themeName: string): CSSProperties {
  const theme = themes[themeName] || themes.dark;
  const styleVars: Record<string, string> = {
    "--background": theme.colors.background,
    "--foreground": theme.colors.foreground,
    "--card": theme.colors.card,
    "--card-foreground": theme.colors.cardForeground,
    "--popover": theme.colors.popover,
    "--popover-foreground": theme.colors.popoverForeground,
    "--primary": theme.colors.primary,
    "--primary-foreground": theme.colors.primaryForeground,
    "--secondary": theme.colors.secondary,
    "--secondary-foreground": theme.colors.secondaryForeground,
    "--muted": theme.colors.muted,
    "--muted-foreground": theme.colors.mutedForeground,
    "--accent": theme.colors.accent,
    "--accent-foreground": theme.colors.accentForeground,
    "--destructive": theme.colors.destructive,
    "--destructive-foreground": theme.colors.destructiveForeground,
    "--border": theme.colors.border,
    "--input": theme.colors.input,
    "--ring": theme.colors.ring,
  };

  if (theme.colors.cardHover) {
    styleVars["--card-hover"] = theme.colors.cardHover;
  }
  if (theme.colors.borderHighlight) {
    styleVars["--border-highlight"] = theme.colors.borderHighlight;
  }
  if (theme.colors.accentBg) {
    styleVars["--accent-bg"] = theme.colors.accentBg;
  }
  if (theme.colors.accentBorder) {
    styleVars["--accent-border"] = theme.colors.accentBorder;
  }
  if (theme.colors.shadowColor) {
    styleVars["--shadow-color"] = theme.colors.shadowColor;
  }
  if (theme.colors.gradientFrom) {
    styleVars["--gradient-from"] = theme.colors.gradientFrom;
  }
  if (theme.colors.gradientTo) {
    styleVars["--gradient-to"] = theme.colors.gradientTo;
  }
  if (theme.colors.sidebarActiveBgFrom) {
    styleVars["--sidebar-active-bg-from"] = theme.colors.sidebarActiveBgFrom;
  }
  if (theme.colors.sidebarActiveBgTo) {
    styleVars["--sidebar-active-bg-to"] = theme.colors.sidebarActiveBgTo;
  }
  if (theme.colors.sidebarActiveText) {
    styleVars["--sidebar-active-text"] = theme.colors.sidebarActiveText;
  }
  if (theme.colors.sidebarActiveBorder) {
    styleVars["--sidebar-active-border"] = theme.colors.sidebarActiveBorder;
  }

  return styleVars as CSSProperties;
};