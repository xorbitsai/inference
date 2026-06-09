'use client';

import { FC, useState, useMemo } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { FaGithub } from 'react-icons/fa';
import {
  Database,
  Box,
  Layers,
  ChevronRight,
  Cpu,
  FileTextIcon,
  SquareArrowOutUpRight,
  Globe,
  BotIcon,
  Moon,
  Rocket,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { usePathname } from 'next/navigation';

import { useI18n } from '@/contexts/i18n-context';
import { useGlobal } from '@/contexts/global-context';
import { cn } from '@/lib/utils';
import { getBrandingFromEnv } from '@/lib/branding';
import {
  XINFERENCE_DOCS_URL,
  XINFERENCE_BASE_URL,
  XINFERENCE_CN_URL,
  XINFERENCE_GITHUB,
} from '@/constants';
import ThemeToggle from '@/components/layout/theme-toggle';
import LanguageSwitcher from '@/components/layout/language-switcher';
import LoginOut from '@/components/layout/login-out';

interface NavItem {
  path: string;
  name: React.ReactNode;
  target?: '_blank';
  Icon: LucideIcon;
  Extra: LucideIcon;
}
const Nav: FC<NavItem> = ({ path, name, target, Icon, Extra }) => {
  const pathname = usePathname();

  const isActive = useMemo(() => {
    if (path === '/') {
      return pathname === '/';
    }
    return pathname === path || pathname.startsWith(`${path}/`);
  }, [pathname, path]);
  return (
    <Link
      href={path}
      target={target || ''}
      className={cn(
        'flex items-center justify-between p-2 rounded-lg transition-colors rounded-lg',
        isActive
          ? 'bg-primary/10 text-primary'
          : 'text-muted-foreground hover:bg-accent hover:text-foreground'
      )}
    >
      <div className="flex items-center gap-2.5">
        <Icon className="w-5 h-5" />
        {name}
      </div>
      <Extra className="w-3.5 h-3.5" />
    </Link>
  );
};
export function Sidebar() {
  const { t, locale } = useI18n();
  const [isExpanded] = useState(false);
  const branding = getBrandingFromEnv();
  const { clusterVersion } = useGlobal();
  const links: NavItem[] = [
    {
      path: '/launch-model',
      name: t('menu.launchModel'),
      Icon: Rocket,
      Extra: ChevronRight,
    },
    {
      path: '/running-model',
      name: t('menu.runningModels'),
      Icon: Layers,
      Extra: ChevronRight,
    },
    {
      path: '/register-model',
      name: t('menu.registerModel'),
      Icon: Box,
      Extra: ChevronRight,
    },
    {
      path: '/cluster-information',
      name: t('menu.clusterInfo'),
      Icon: Cpu,
      Extra: ChevronRight,
    },
  ];
  const externalLinks: NavItem[] = [
    {
      path: `${XINFERENCE_DOCS_URL}/${locale === 'zh' ? 'zh-cn' : ''}`,
      name: t('menu.documentation'),
      target: '_blank',
      Icon: FileTextIcon,
      Extra: SquareArrowOutUpRight,
    },
    {
      path: `${XINFERENCE_GITHUB}/inference`,
      name: t('menu.contactUs'),
      Icon: FaGithub as any,
      Extra: SquareArrowOutUpRight,
    },
    {
      path: locale === 'zh' ? XINFERENCE_CN_URL : XINFERENCE_BASE_URL,
      name: t('menu.website'),
      Icon: Globe,
      Extra: SquareArrowOutUpRight,
    },
    {
      path: `${XINFERENCE_GITHUB}/xagent`,
      name: t('menu.xagent'),
      Icon: BotIcon,
      Extra: SquareArrowOutUpRight,
    },
  ];
  return (
    <div
      className={cn(
        'flex flex-col bg-card border-r border-border transition-all duration-300 shrink-0 overflow-y-auto',
        isExpanded ? 'w-0' : 'w-60'
      )}
    >
      <div className="flex h-16 items-center justify-between px-6 mt-2 mb-4 relative">
        <Link href="/" className="flex items-center justify-center gap-2">
          <Image src={branding.logoPath} alt={branding.logoAlt} width={32} height={32} className="rounded-lg" />
          <h1 className="text-xl font-bold text-foreground">{branding.appName}</h1>
        </Link>
      </div>
      <nav className="flex-1 min-h-0 px-3 pb-4 flex flex-col gap-1">
        {links.map((item) => (
          <Nav {...item} key={item.path} />
        ))}
      </nav>
      <nav className="px-3 border-t border-border py-2 flex flex-col gap-1">
        {externalLinks.map((item) => (
          <Nav {...item} key={item.path} />
        ))}
      </nav>
      <div className="py-3 px-5 border-t border-border flex justify-between items-center gap-3">
        <div className="flex gap-4 shrink-0">
          <ThemeToggle />
          <LanguageSwitcher />
          <LoginOut />
        </div>
        {clusterVersion?.version && (
          <div className="text-slate-400 truncate">v:{clusterVersion.version}</div>
        )}
      </div>
    </div>
  );
}
