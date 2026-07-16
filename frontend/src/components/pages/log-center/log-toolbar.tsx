'use client';

import { Search } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { LOG_LEVELS, LOG_TYPES } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

interface LogToolbarProps {
  nodes: string[];
  selectedNode: string;
  onSelectedNodeChange: (value: string) => void;
  searchText: string;
  onSearchTextChange: (value: string) => void;
  onSearchCommit: () => void;
  selectedLevels: string[];
  onToggleLevel: (value: string) => void;
  selectedLogType: string;
  onSelectedLogTypeChange: (value: string) => void;
}

export function LogToolbar({
  nodes,
  selectedNode,
  onSelectedNodeChange,
  searchText,
  onSearchTextChange,
  onSearchCommit,
  selectedLevels,
  onToggleLevel,
  selectedLogType,
  onSelectedLogTypeChange,
}: LogToolbarProps) {
  const { t } = useI18n();

  return (
    <div className="flex flex-col gap-3 border-b bg-background px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        {nodes.length > 0 && (
          <Select
            value={selectedNode}
            onChange={(value) => onSelectedNodeChange(String(value || ''))}
            options={[
              { value: '', label: t('logCenter.allNodes') },
              ...nodes.map((node) => ({ value: node, label: node })),
            ]}
            allowClear={false}
            className="w-56"
          />
        )}
        <div className="relative w-72 max-w-full">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={searchText}
            onChange={(event) => onSearchTextChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') onSearchCommit();
            }}
            placeholder={t('logCenter.searchPlaceholder')}
            className="pl-9"
          />
        </div>
        <div className="min-w-0 flex-1" />
      </div>
      <div className="flex flex-wrap items-center gap-5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="w-16 text-sm text-muted-foreground">{t('logCenter.logLevel')}</span>
          <div className="flex flex-wrap gap-1.5">
            {LOG_LEVELS.map((level) => (
              <button
                key={level}
                type="button"
                className={cn(
                  'h-7 rounded-md border px-2 text-xs font-medium transition-colors hover:bg-accent',
                  selectedLevels.includes(level)
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'bg-background text-muted-foreground'
                )}
                onClick={() => onToggleLevel(level)}
              >
                {level}
              </button>
            ))}
          </div>
        </div>
        <div className="flex min-w-0 items-center gap-2">
          <span className="w-16 text-sm text-muted-foreground">{t('logCenter.nodeType')}</span>
          <div className="flex flex-wrap gap-1.5">
            {LOG_TYPES.map((logType) => (
              <button
                key={logType}
                type="button"
                className={cn(
                  'h-7 rounded-md border px-2 text-xs font-medium transition-colors hover:bg-accent',
                  selectedLogType === logType
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'bg-background text-muted-foreground'
                )}
                onClick={() => onSelectedLogTypeChange(selectedLogType === logType ? '' : logType)}
              >
                {logType}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
