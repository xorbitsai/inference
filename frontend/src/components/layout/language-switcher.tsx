import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Languages } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';
import { LANGUAGES } from '@/constants';
import { cn } from '@/lib/utils';

interface LanguageSwitcherProps {
  className?: string;
  iconClassName?: string;
}

export default function LanguageSwitcher({ className, iconClassName }: LanguageSwitcherProps) {
  const { locale, setLocale } = useI18n();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className={cn('text-muted-foreground hover:text-foreground transition-colors', className)}
        >
          <Languages className={cn('h-5 w-5', iconClassName)} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {LANGUAGES.map((lang) => (
          <DropdownMenuItem
            key={lang.value}
            onClick={() => setLocale(lang.value)}
            className={locale === lang.value ? 'bg-accent' : ''}
          >
            {lang.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
