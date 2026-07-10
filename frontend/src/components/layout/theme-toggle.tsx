'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { cn } from '@/lib/utils';

interface ThemeToggleProps {
  className?: string;
  iconClassName?: string;
}

export default function ThemeToggle({ className, iconClassName }: ThemeToggleProps) {
  const { resolvedTheme, setTheme } = useTheme();

  const isDark = resolvedTheme === 'dark';

  const toggleTheme = () => {
    setTheme(isDark ? 'light' : 'dark');
  };

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className={cn('text-muted-foreground hover:text-foreground transition-colors', className)}
    >
      {isDark ? (
        <Sun className={cn('h-5 w-5', iconClassName)} />
      ) : (
        <Moon className={cn('h-5 w-5', iconClassName)} />
      )}
    </button>
  );
}
