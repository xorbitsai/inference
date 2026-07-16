'use client';

import { X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

import type { FieldFilter } from './types';

interface FilterChipBarProps {
  filters: FieldFilter[];
  clearLabel: string;
  onRemove: (index: number) => void;
  onClear: () => void;
  className?: string;
}

export function FilterChipBar({
  filters,
  clearLabel,
  onRemove,
  onClear,
  className,
}: FilterChipBarProps) {
  if (!filters.length) return null;

  return (
    <div className={cn('flex flex-wrap items-center gap-2 border-b bg-muted/20 px-4 py-2', className)}>
      {filters.map((filter, index) => (
        <span
          key={`${filter.op}${filter.key}:${filter.value}-${index}`}
          className={cn(
            'inline-flex max-w-full items-center gap-1 rounded-md border px-2 py-1 text-xs',
            filter.op === '-' ? 'border-destructive/40 text-destructive' : 'bg-background'
          )}
        >
          <span className="max-w-[18rem] truncate">
            {filter.op === '-' ? 'NOT ' : ''}
            {filter.key}: {filter.value}
          </span>
          <button
            type="button"
            aria-label={`${clearLabel} ${filter.key}`}
            className="rounded-sm text-muted-foreground hover:text-foreground"
            onClick={() => onRemove(index)}
          >
            <X className="size-3.5" />
          </button>
        </span>
      ))}
      <Button variant="outline" size="sm" className="h-7 px-2 text-xs" onClick={onClear}>
        {clearLabel}
      </Button>
    </div>
  );
}
