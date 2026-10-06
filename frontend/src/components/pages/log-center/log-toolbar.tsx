'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, RefreshCw, Search, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { LOG_LEVEL_BADGE_CLASSES, LOG_LEVELS, LOG_TYPES } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

import type { LogNodeOption } from './types';
import { NodeRoleBadge } from './utils';

interface LogToolbarProps {
  nodeOptions: LogNodeOption[];
  selectedNodes: string[];
  onSelectedNodesChange: (values: string[]) => void;
  searchText: string;
  onSearchTextChange: (value: string) => void;
  onSearchCommit?: () => void;
  selectedLevels: string[];
  onToggleLevel: (value: string) => void;
  selectedLogType?: string;
  onSelectedLogTypeChange?: (value: string) => void;
  emptySelectionMeansAll?: boolean;
  maxSelectedNodes?: number;
  onReset?: () => void;
  actionLabel?: string;
  onAction?: () => void;
  actionDisabled?: boolean;
}

export function LogToolbar({
  nodeOptions,
  selectedNodes,
  onSelectedNodesChange,
  searchText,
  onSearchTextChange,
  onSearchCommit,
  selectedLevels,
  onToggleLevel,
  selectedLogType = '',
  onSelectedLogTypeChange,
  emptySelectionMeansAll = false,
  maxSelectedNodes,
  onReset,
  actionLabel,
  onAction,
  actionDisabled = false,
}: LogToolbarProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [nodeSearch, setNodeSearch] = useState('');
  const [selectionLimitReached, setSelectionLimitReached] = useState(false);
  const listboxId = useId();
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedSet = useMemo(() => new Set(selectedNodes), [selectedNodes]);
  const optionMap = useMemo(
    () => new Map(nodeOptions.map((option) => [option.value, option])),
    [nodeOptions]
  );
  const filteredOptions = useMemo(() => {
    const needle = nodeSearch.trim().toLowerCase();
    if (!needle) return nodeOptions;
    return nodeOptions.filter((option) =>
      [
        option.label,
        option.roleLabel,
        option.description,
        option.searchText,
        option.fullAddress,
        option.value,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
        .includes(needle)
    );
  }, [nodeOptions, nodeSearch]);

  const close = useCallback(() => {
    setOpen(false);
    setNodeSearch('');
    setSelectionLimitReached(false);
  }, []);

  useEffect(() => {
    if (!open) return;
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) close();
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [close, open]);

  const handleToggleNode = (node: string) => {
    if (selectedSet.has(node)) {
      onSelectedNodesChange(selectedNodes.filter((value) => value !== node));
      setSelectionLimitReached(false);
      return;
    }
    if (maxSelectedNodes && selectedNodes.length >= maxSelectedNodes) {
      setSelectionLimitReached(true);
      return;
    }
    onSelectedNodesChange([...selectedNodes, node]);
    setSelectionLimitReached(false);
  };

  const firstSelected = selectedNodes[0] ? optionMap.get(selectedNodes[0]) : undefined;
  const triggerLabel =
    selectedNodes.length === 0
      ? emptySelectionMeansAll
        ? t('logCenter.allNodes')
        : t('logCenter.selectNodes')
      : selectedNodes.length === 1
        ? firstSelected?.label || selectedNodes[0]
        : `${firstSelected?.label || selectedNodes[0]} +${selectedNodes.length - 1}`;

  const canSelectAll =
    !emptySelectionMeansAll &&
    nodeOptions.length > 0 &&
    (!maxSelectedNodes || nodeOptions.length <= maxSelectedNodes);

  return (
    <div className="flex flex-col gap-3 border-b bg-background px-4 py-3">
      <div className="flex flex-wrap items-start gap-3">
        <div ref={containerRef} className="relative min-w-64 max-w-full">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">
            {t('logCenter.node')}
          </label>
          <button
            type="button"
            role="combobox"
            aria-controls={listboxId}
            aria-expanded={open}
            aria-haspopup="listbox"
            className={cn(
              'flex h-9 w-80 max-w-full items-center justify-between rounded-md border border-input bg-background px-3 text-sm',
              'hover:bg-accent hover:text-accent-foreground',
              open && 'ring-2 ring-ring ring-offset-2'
            )}
            onClick={() => setOpen((value) => !value)}
            onKeyDown={(event) => {
              if (event.key === 'Escape') close();
            }}
          >
            <span className="truncate text-left">{triggerLabel}</span>
            <ChevronDown className="ml-2 size-4 shrink-0 opacity-50" />
          </button>
          {open && (
            <div className="absolute left-0 z-50 mt-1 w-96 max-w-[calc(100vw-2rem)] rounded-md border bg-popover p-2 text-popover-foreground shadow-md">
              <div className="relative mb-2">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  autoFocus
                  value={nodeSearch}
                  onChange={(event) => setNodeSearch(event.target.value)}
                  placeholder={t('logCenter.searchNodes')}
                  className="h-8 pl-8"
                />
              </div>
              <div
                id={listboxId}
                role="listbox"
                aria-multiselectable="true"
                className="max-h-60 overflow-auto"
              >
                {emptySelectionMeansAll && !nodeSearch && (
                  <button
                    type="button"
                    role="option"
                    aria-selected={selectedNodes.length === 0}
                    className={cn(
                      'flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm',
                      selectedNodes.length === 0 ? 'bg-accent' : 'hover:bg-accent'
                    )}
                    onClick={() => {
                      onSelectedNodesChange([]);
                      close();
                    }}
                  >
                    <Check
                      className={cn(
                        'size-4 shrink-0',
                        selectedNodes.length === 0 ? 'opacity-100' : 'opacity-0'
                      )}
                    />
                    <span>{t('logCenter.allNodes')}</span>
                  </button>
                )}
                {filteredOptions.map((option) => {
                  const selected = selectedSet.has(option.value);
                  return (
                    <button
                      key={option.value}
                      type="button"
                      role="option"
                      aria-selected={selected}
                      className={cn(
                        'flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm',
                        selected ? 'bg-accent text-accent-foreground' : 'hover:bg-accent'
                      )}
                      onClick={() => handleToggleNode(option.value)}
                    >
                      <Check
                        className={cn('size-4 shrink-0', selected ? 'opacity-100' : 'opacity-0')}
                      />
                      <NodeRoleBadge role={option.role} className="w-20 shrink-0 justify-center">
                        {option.roleLabel}
                      </NodeRoleBadge>
                      <span className="min-w-0 flex-1">
                        <span
                          className="block truncate font-mono text-xs"
                          title={option.fullAddress || option.label}
                        >
                          {option.label}
                        </span>
                        {option.description && option.description !== option.label && (
                          <span className="block truncate text-[11px] text-muted-foreground">
                            {option.description}
                          </span>
                        )}
                      </span>
                    </button>
                  );
                })}
                {filteredOptions.length === 0 && (
                  <p className="px-2 py-3 text-center text-sm text-muted-foreground">
                    {t('logCenter.noNodes')}
                  </p>
                )}
              </div>
              {!emptySelectionMeansAll && (
                <div className="mt-2 flex items-center justify-between border-t pt-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={!canSelectAll}
                    onClick={() => onSelectedNodesChange(nodeOptions.map((option) => option.value))}
                  >
                    {t('logCenter.selectAllNodes')}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={selectedNodes.length === 0}
                    onClick={() => onSelectedNodesChange([])}
                  >
                    {t('logCenter.clearSelection')}
                  </Button>
                </div>
              )}
              {!canSelectAll && !emptySelectionMeansAll && nodeOptions.length > 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {t('logCenter.selectAllDisabled', { count: maxSelectedNodes })}
                </p>
              )}
              {selectionLimitReached && (
                <p className="mt-1 text-xs text-destructive">
                  {t('logCenter.selectionLimit', { count: maxSelectedNodes })}
                </p>
              )}
            </div>
          )}
        </div>

        <div className="w-80 max-w-full">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">
            {t('logCenter.search')}
          </label>
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={searchText}
              onChange={(event) => onSearchTextChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') onSearchCommit?.();
              }}
              placeholder={t('logCenter.searchPlaceholder')}
              className="pl-9"
            />
          </div>
        </div>
      </div>

      {selectedNodes.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {selectedNodes.map((node) => {
            const option = optionMap.get(node);
            return (
              <span
                key={node}
                className="inline-flex max-w-full items-center gap-1.5 overflow-hidden rounded-md border bg-background p-1 text-xs"
                title={[option?.roleLabel, option?.fullAddress || option?.label || node]
                  .filter(Boolean)
                  .join(' ')}
              >
                {option && (
                  <NodeRoleBadge role={option.role} className="shrink-0">
                    {option.roleLabel}
                  </NodeRoleBadge>
                )}
                <span className="max-w-64 truncate px-1 font-mono">{option?.label || node}</span>
                <button
                  type="button"
                  className="mr-1 rounded-sm p-0.5 hover:bg-muted-foreground/20"
                  aria-label={t('logCenter.removeNode', { node: option?.label || node })}
                  onClick={() =>
                    onSelectedNodesChange(selectedNodes.filter((value) => value !== node))
                  }
                >
                  <X className="size-3" />
                </button>
              </span>
            );
          })}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-sm text-muted-foreground">{t('logCenter.logLevel')}</span>
          <div className="flex flex-wrap gap-1.5">
            {LOG_LEVELS.map((level) => (
              <button
                key={level}
                type="button"
                className={cn(
                  'h-7 rounded-md border px-2 text-xs font-medium transition-all',
                  LOG_LEVEL_BADGE_CLASSES[level],
                  selectedLevels.includes(level)
                    ? 'opacity-100 ring-2 ring-current ring-offset-1'
                    : 'opacity-60 hover:opacity-100'
                )}
                onClick={() => onToggleLevel(level)}
              >
                {level}
              </button>
            ))}
          </div>
        </div>
        {onSelectedLogTypeChange && (
          <div className="flex min-w-0 items-center gap-2">
            <span className="text-sm text-muted-foreground">{t('logCenter.logType')}</span>
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
                  onClick={() =>
                    onSelectedLogTypeChange(selectedLogType === logType ? '' : logType)
                  }
                >
                  {logType}
                </button>
              ))}
            </div>
          </div>
        )}
        {(onReset || onAction) && (
          <div className="ml-auto flex items-center gap-2">
            {onReset && (
              <Button type="button" variant="ghost" size="sm" onClick={onReset}>
                {t('logCenter.reset')}
              </Button>
            )}
            {onAction && actionLabel && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={actionDisabled}
                onClick={onAction}
              >
                <RefreshCw className="size-4" />
                {actionLabel}
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
