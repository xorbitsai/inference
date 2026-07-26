'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, Search, X } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { LOG_LEVELS, LOG_TYPES } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

interface LogToolbarProps {
  nodes: string[];
  selectedNodes: string[];
  onSelectedNodesChange: (values: string[]) => void;
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
  selectedNodes,
  onSelectedNodesChange,
  searchText,
  onSearchTextChange,
  onSearchCommit,
  selectedLevels,
  onToggleLevel,
  selectedLogType,
  onSelectedLogTypeChange,
}: LogToolbarProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Build option list: first "All nodes" (empty string), then each node
  const options: string[] = ['', ...nodes];

  const close = useCallback(() => {
    setOpen(false);
    setFocusedIndex(-1);
  }, []);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        close();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open, close]);

  // Scroll focused option into view
  useEffect(() => {
    if (!open || focusedIndex < 0 || !listRef.current) return;
    const items = listRef.current.querySelectorAll('[role="option"]');
    if (items[focusedIndex]) {
      items[focusedIndex].scrollIntoView({ block: 'nearest' });
    }
  }, [open, focusedIndex]);

  const handleToggleNode = (node: string) => {
    if (node === '') {
      onSelectedNodesChange([]);
      close();
    } else {
      const next = selectedNodes.includes(node)
        ? selectedNodes.filter((n) => n !== node)
        : [...selectedNodes, node];
      onSelectedNodesChange(next);
    }
  };

  const handleRemoveNode = (node: string, e: React.MouseEvent) => {
    e.stopPropagation();
    onSelectedNodesChange(selectedNodes.filter((n) => n !== node));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!open) {
          setOpen(true);
          setFocusedIndex(0);
        } else {
          setFocusedIndex((prev) => (prev + 1) % options.length);
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (!open) {
          setOpen(true);
          setFocusedIndex(options.length - 1);
        } else {
          setFocusedIndex((prev) => (prev - 1 + options.length) % options.length);
        }
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (!open) {
          setOpen(true);
          setFocusedIndex(0);
        } else if (focusedIndex >= 0 && focusedIndex < options.length) {
          handleToggleNode(options[focusedIndex]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        close();
        break;
    }
  };

  const triggerLabel =
    selectedNodes.length === 0
      ? t('logCenter.allNodes')
      : selectedNodes.length === 1
        ? selectedNodes[0]
        : `${selectedNodes[0]} +${selectedNodes.length - 1}`;

  return (
    <div className="flex flex-col gap-3 border-b bg-background px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        {nodes.length > 0 && (
          <div ref={containerRef} className="relative w-56">
            <button
              type="button"
              role="combobox"
              aria-expanded={open}
              aria-haspopup="listbox"
              className={cn(
                'flex h-9 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-1 text-sm',
                'hover:bg-accent hover:text-accent-foreground',
                open && 'ring-2 ring-ring ring-offset-2'
              )}
              onClick={() => {
                setOpen(!open);
                if (!open) setFocusedIndex(-1);
              }}
              onKeyDown={handleKeyDown}
            >
              <span className="truncate text-left">{triggerLabel}</span>
              <ChevronDown className="ml-2 size-4 shrink-0 opacity-50" />
            </button>
            {open && (
              <div
                ref={listRef}
                role="listbox"
                aria-multiselectable="true"
                className="absolute z-50 mt-1 w-full rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
              >
                <div className="max-h-60 overflow-auto">
                  {options.map((node, index) => {
                    const isSelected = node === '' ? selectedNodes.length === 0 : selectedNodes.includes(node);
                    const isFocused = index === focusedIndex;

                    return (
                      <button
                        key={node || '__all__'}
                        type="button"
                        role="option"
                        aria-selected={isSelected}
                        className={cn(
                          'flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm',
                          isFocused && 'bg-accent text-accent-foreground',
                          !isFocused && isSelected && 'bg-accent/50 text-accent-foreground',
                          !isFocused && !isSelected && 'hover:bg-accent hover:text-accent-foreground'
                        )}
                        onClick={() => handleToggleNode(node)}
                        onMouseEnter={() => setFocusedIndex(index)}
                      >
                        <Check
                          className={cn('size-4', isSelected ? 'opacity-100' : 'opacity-0')}
                        />
                        <span className="truncate">
                          {node === '' ? t('logCenter.allNodes') : node}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
            {selectedNodes.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {selectedNodes.slice(0, 3).map((node) => (
                  <span
                    key={node}
                    className="inline-flex items-center gap-0.5 rounded-md border bg-muted px-1.5 py-0.5 text-xs"
                  >
                    <span className="max-w-[120px] truncate">{node}</span>
                    <button
                      type="button"
                      className="ml-0.5 rounded-sm hover:bg-muted-foreground/20"
                      onClick={(e) => handleRemoveNode(node, e)}
                      aria-label={`Remove ${node}`}
                    >
                      <X className="size-3" />
                    </button>
                  </span>
                ))}
                {selectedNodes.length > 3 && (
                  <span className="inline-flex items-center rounded-md border bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                    +{selectedNodes.length - 3}
                  </span>
                )}
              </div>
            )}
          </div>
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
