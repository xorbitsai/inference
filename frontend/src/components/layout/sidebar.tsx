'use client';

import { ComponentType, FC, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { FaGithub } from 'react-icons/fa';
import {
  Box,
  Layers,
  ChevronLeft,
  ChevronRight,
  Cpu,
  FileSearch,
  FileTextIcon,
  SquareArrowOutUpRight,
  Globe,
  BotIcon,
  Rocket,
  Monitor,
  ScrollText,
  ShieldCheck,
  Users,
  UserRound,
  KeyRound,
} from 'lucide-react';
import { usePathname } from 'next/navigation';

import { useI18n } from '@/contexts/i18n-context';
import { useGlobal } from '@/contexts/global-context';
import { getAccessToken } from '@/lib/auth-storage';
import { cn, decodeJwtPayload } from '@/lib/utils';
import { getBrandingFromEnv } from '@/lib/branding';
import {
  XINFERENCE_DOCS_URL,
  XINFERENCE_BASE_URL,
  XINFERENCE_CN_URL,
  XINFERENCE_GITHUB,
  NO_AUTH,
} from '@/constants';
import ThemeToggle from '@/components/layout/theme-toggle';
import LanguageSwitcher from '@/components/layout/language-switcher';
import LoginOut from '@/components/layout/login-out';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useMenuAuth } from '@/hooks/use-menu-auth';

type IconComponent = ComponentType<{ className?: string }>;

interface NavItem {
  path: string;
  name: React.ReactNode;
  target?: '_blank';
  Icon: IconComponent;
  Extra?: IconComponent;
  show?: boolean;
}

interface NavGroup {
  name: string;
  items: NavItem[];
}

const Nav: FC<NavItem & { collapsed: boolean }> = ({
  path,
  name,
  target,
  Icon,
  Extra,
  collapsed,
}) => {
  const pathname = usePathname();

  const isActive = useMemo(() => {
    if (path === '/') {
      return pathname === '/';
    }
    return pathname === path || pathname.startsWith(`${path}/`);
  }, [pathname, path]);
  const link = (
    <Link
      href={path}
      target={target}
      rel={target === '_blank' ? 'noreferrer' : undefined}
      className={cn(
        'flex h-10 items-center rounded-lg transition-colors',
        collapsed ? 'justify-center px-0' : 'justify-between px-2',
        isActive
          ? 'bg-primary/10 text-primary'
          : 'text-muted-foreground hover:bg-accent hover:text-foreground'
      )}
      aria-label={typeof name === 'string' ? name : undefined}
    >
      <div className={cn('flex min-w-0 items-center', collapsed ? 'justify-center' : 'gap-2.5')}>
        <Icon className="h-5 w-5 shrink-0" />
        {!collapsed && <span className="truncate">{name}</span>}
      </div>
      {!collapsed && Extra && <Extra className="h-3.5 w-3.5 shrink-0" />}
    </Link>
  );

  if (!collapsed) {
    return link;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>{link}</TooltipTrigger>
      <TooltipContent side="right">
        <p>{name}</p>
      </TooltipContent>
    </Tooltip>
  );
};

const NavGroupSection: FC<NavGroup & { collapsed: boolean; showDivider: boolean }> = ({
  name,
  items,
  collapsed,
  showDivider,
}) => {
  if (items.length === 0) {
    return null;
  }

  return (
    <section className="flex flex-col gap-1">
      {collapsed ? (
        showDivider && <div className="mx-auto my-2 h-px w-8 bg-border" />
      ) : (
        <div
          className={cn(
            'px-2 pb-1 text-xs font-medium text-muted-foreground',
            showDivider && 'pt-3'
          )}
        >
          {name}
        </div>
      )}
      {items.map((item) => (
        <Nav {...item} collapsed={collapsed} key={item.path} />
      ))}
    </section>
  );
};

export function Sidebar() {
  const { t, locale } = useI18n();
  const [collapsed, setCollapsed] = useState(false);
  const branding = getBrandingFromEnv();
  const { clusterVersion, clusterAuth, clusterUIConfig } = useGlobal();
  const { isAdmin, usersManagePage, canAccessKeysPage } = useMenuAuth();
  const [token, setToken] = useState<string | undefined>();
  const showLoginOut = useMemo(
    () => Boolean(clusterAuth?.auth && token && token !== NO_AUTH),
    [clusterAuth, token]
  );
  const username = useMemo(() => {
    if (!showLoginOut || !token || token === NO_AUTH) {
      return '';
    }

    const payload = decodeJwtPayload(token);
    if (!payload) {
      return '';
    }

    const usernameKeys = ['username', 'preferred_username', 'name', 'sub', 'user_name'];
    const usernameValue = usernameKeys
      .map((key) => payload[key])
      .find((value) => typeof value === 'string' && value.trim());

    if (typeof usernameValue === 'string') {
      return usernameValue;
    }

    const userId = payload.user_id;
    return typeof userId === 'string' || typeof userId === 'number' ? String(userId) : '';
  }, [showLoginOut, token]);

  useEffect(() => {
    setToken(getAccessToken());
  }, []);

  const navGroups = useMemo<NavGroup[]>(() => {
    const groups: NavGroup[] = [
      {
        name: t('menu.modelManagement'),
        items: [
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
        ],
      },
      {
        name: t('menu.monitoringManagement'),
        items: [
          {
            path: '/cluster-information',
            name: t('menu.clusterInfo'),
            Icon: Cpu,
            Extra: ChevronRight,
          },
          {
            path: '/monitor-center',
            name: t('menu.monitorCenter'),
            Icon: Monitor,
            Extra: ChevronRight,
          },
          {
            path: '/log-center',
            name: t('menu.logCenter'),
            Icon: ScrollText,
            Extra: ChevronRight,
            show: Boolean(clusterUIConfig?.es_enabled),
          },
        ],
      },
      {
        name: t('menu.systemManagement'),
        items: [
          {
            path: '/user-management',
            name: t('menu.userManagement'),
            Icon: Users,
            Extra: ChevronRight,
            show: Boolean(clusterUIConfig?.auth_advanced) && usersManagePage,
          },
          {
            path: '/api-key-management',
            name: t('menu.apiKeyManagement'),
            Icon: KeyRound,
            Extra: ChevronRight,
            show: Boolean(clusterUIConfig?.auth_advanced) && canAccessKeysPage,
          },
          {
            path: '/security-settings',
            name: t('menu.securitySettings'),
            Icon: ShieldCheck,
            Extra: ChevronRight,
            show: Boolean(clusterUIConfig?.auth_advanced) && isAdmin,
          },
          {
            path: '/audit-center',
            name: t('menu.auditCenter'),
            Icon: FileSearch,
            Extra: ChevronRight,
            show: Boolean(clusterUIConfig?.auth_advanced) && isAdmin,
          },
        ],
      },
      {
        name: t('menu.resourcesAndSupport'),
        items: [
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
            target: '_blank',
            Icon: FaGithub as IconComponent,
            Extra: SquareArrowOutUpRight,
          },
          {
            path: locale === 'zh' ? XINFERENCE_CN_URL : XINFERENCE_BASE_URL,
            name: t('menu.website'),
            target: '_blank',
            Icon: Globe,
            Extra: SquareArrowOutUpRight,
          },
          {
            path: `${XINFERENCE_GITHUB}/xagent`,
            name: t('menu.xagent'),
            target: '_blank',
            Icon: BotIcon,
            Extra: SquareArrowOutUpRight,
          },
        ],
      },
    ];

    return groups
      .map((group) => ({
        ...group,
        items: group.items.filter(({ show = true }) => show),
      }))
      .filter(({ items }) => items.length > 0);
  }, [clusterUIConfig, locale, t, usersManagePage, canAccessKeysPage, isAdmin]);

  return (
    <div
      className={cn(
        'relative flex flex-col bg-card border-r border-border transition-all duration-300 shrink-0 overflow-visible',
        collapsed ? 'w-16' : 'w-60'
      )}
    >
      <button
        type="button"
        aria-label={collapsed ? t('common.unfold') : t('common.packUp')}
        title={collapsed ? t('common.unfold') : t('common.packUp')}
        className="absolute -right-3 top-20 z-20 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-sm transition-colors hover:bg-accent hover:text-foreground"
        onClick={() => setCollapsed((value) => !value)}
      >
        {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
      </button>

      <div
        className={cn(
          'flex h-16 items-center px-4 mt-2 mb-4',
          collapsed ? 'justify-center' : 'justify-between px-6'
        )}
      >
        <Link href="/" className="flex min-w-0 items-center justify-center gap-2">
          <Image
            src={branding.logoPath}
            alt={branding.logoAlt}
            width={32}
            height={32}
            className="shrink-0 rounded-lg"
          />
          {!collapsed && (
            <h1 className="truncate text-xl font-bold text-foreground">{branding.appName}</h1>
          )}
        </Link>
      </div>

      <TooltipProvider delayDuration={300}>
        <nav className="flex-1 min-h-0 overflow-y-auto px-3 pb-4 flex flex-col gap-1">
          {navGroups.map((group, index) => (
            <NavGroupSection
              {...group}
              collapsed={collapsed}
              showDivider={index > 0}
              key={group.name}
            />
          ))}
        </nav>
      </TooltipProvider>

      {!collapsed ? (
        <div className="border-t border-border px-4 py-3">
          <div className="flex h-8 items-center justify-between gap-3">
            <div className="flex shrink-0 items-center gap-1">
              <ThemeToggle className="flex h-8 w-8 items-center justify-center rounded-md" />
              <LanguageSwitcher className="flex h-8 w-8 items-center justify-center rounded-md" />
            </div>
            {clusterVersion?.version && (
              <div className="min-w-0 max-w-[132px] truncate text-right text-xs text-slate-400">
                v:{clusterVersion.version}
              </div>
            )}
          </div>
          {showLoginOut && (
            <div className="mt-2 flex h-8 min-w-0 items-center justify-between gap-2">
              <div className="flex min-w-0 flex-1 items-center gap-1 text-sm text-muted-foreground">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center">
                  <UserRound className="h-5 w-5" />
                </span>
                <span className="truncate" title={username}>
                  {username}
                </span>
              </div>
              <LoginOut className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md" />
            </div>
          )}
        </div>
      ) : (
        showLoginOut && (
          <div className="flex justify-center border-t border-border py-3">
            <LoginOut className="flex h-8 w-8 items-center justify-center rounded-md" />
          </div>
        )
      )}
    </div>
  );
}
