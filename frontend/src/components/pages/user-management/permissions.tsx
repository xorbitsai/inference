'use client';

import { ChevronDown, ChevronUp } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox-group';
import { Label } from '@/components/ui/label';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

export const ALL_PERMISSIONS = [
  'admin',
  'models:list',
  'models:read',
  'models:write',
  'keys:create',
  'keys:manage',
  'users:manage',
  'cache:list',
  'cache:delete',
  'virtualenv:list',
  'virtualenv:delete',
];

const PERMISSION_GROUPS = [
  {
    key: 'admin',
    permissions: ['admin'],
  },
  {
    key: 'models',
    permissions: ['models:list', 'models:read', 'models:write'],
  },
  {
    key: 'keys',
    permissions: ['keys:create', 'keys:manage'],
  },
  {
    key: 'users',
    permissions: ['users:manage'],
  },
  {
    key: 'cache',
    permissions: ['cache:list', 'cache:delete'],
  },
  {
    key: 'virtualenv',
    permissions: ['virtualenv:list', 'virtualenv:delete'],
  },
];

interface PermissionSelectorProps {
  selected: string[];
  onChange: (permissions: string[]) => void;
}

export function getPermissionLabel(t: (key: string) => string, permission: string) {
  const label = t(`userManagement.permissionLabels.${permission}`);

  return label === `userManagement.permissionLabels.${permission}` ? permission : label;
}

export function PermissionSelector({ selected, onChange }: PermissionSelectorProps) {
  const { t } = useI18n();
  const normalizedSelected = selected.filter((permission) => ALL_PERMISSIONS.includes(permission));
  const isAllSelected = ALL_PERMISSIONS.every((permission) =>
    normalizedSelected.includes(permission)
  );
  const isPartialSelected = !isAllSelected && normalizedSelected.length > 0;

  const setPermission = (permission: string, checked: boolean) => {
    if (checked) {
      onChange([...new Set([...selected, permission])]);
      return;
    }

    onChange(selected.filter((item) => item !== permission));
  };

  const toggleAll = () => {
    if (isAllSelected) {
      onChange(selected.filter((permission) => !ALL_PERMISSIONS.includes(permission)));
    } else {
      onChange([...new Set([...selected, ...ALL_PERMISSIONS])]);
    }
  };

  return (
    <div className="rounded-lg border border-border">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <Label className="text-sm font-semibold">{t('userManagement.permissions')}</Label>
        <Checkbox
          checked={isAllSelected ? true : isPartialSelected ? 'indeterminate' : false}
          onCheckedChange={toggleAll}
          label={<span className="text-sm font-medium">{t('userManagement.selectAll')}</span>}
        />
      </div>

      <div className="space-y-3 p-3">
        {PERMISSION_GROUPS.map((group) => (
          <section key={group.key} className="flex flex-col gap-2">
            <h4 className="text-sm font-semibold text-muted-foreground">
              {t(`userManagement.permissionGroups.${group.key}`)}
            </h4>
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              {group.permissions.map((permission) => (
                <Checkbox
                  key={permission}
                  checked={normalizedSelected.includes(permission)}
                  onCheckedChange={(checked) => setPermission(permission, checked === true)}
                  label={
                    <span className="text-sm font-medium">{getPermissionLabel(t, permission)}</span>
                  }
                />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

interface PermissionBadgesProps {
  permissions: string[];
  expanded: boolean;
  onToggleExpanded: () => void;
}

export function PermissionBadges({
  permissions,
  expanded,
  onToggleExpanded,
}: PermissionBadgesProps) {
  const { t } = useI18n();
  const normalizedPermissions = permissions || [];

  if (normalizedPermissions.length === 0) {
    return <span className="text-xs text-muted-foreground">-</span>;
  }

  return (
    <div className="flex max-w-[420px] items-start gap-1.5">
      <div
        className={cn(
          'flex min-w-0 flex-1 gap-1.5',
          expanded ? 'flex-wrap' : 'overflow-hidden whitespace-nowrap'
        )}
      >
        {normalizedPermissions.map((permission) => {
          const isAdmin = permission === 'admin';

          return (
            <Badge
              key={permission}
              variant={isAdmin ? 'default' : 'outline'}
              title={permission}
              className={cn(
                'max-w-none',
                isAdmin && 'border-primary/30 bg-primary/15 text-primary'
              )}
            >
              {getPermissionLabel(t, permission)}
            </Badge>
          );
        })}
      </div>
      {normalizedPermissions.length > 1 && (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0 text-muted-foreground"
          title={
            expanded
              ? t('userManagement.collapsePermissions')
              : t('userManagement.expandPermissions')
          }
          onClick={onToggleExpanded}
        >
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>
      )}
    </div>
  );
}
