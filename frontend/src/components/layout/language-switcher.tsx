import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Languages } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';
import { LANGUAGES } from '@/constants';

export default function LanguageSwitcher() {
  const { locale, setLocale } = useI18n();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild >
        <Languages className="h-5 w-5 text-muted-foreground hover:text-foreground cursor-pointer" />
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
