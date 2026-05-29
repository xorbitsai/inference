'use client';

import * as React from 'react';
import { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { ChevronDown, Check, X } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
  description?: string;
  suffix?: React.ReactNode;
}

interface SelectProps {
  value?: string;
  onChange?: (value: string) => void;

  options?: SelectOption[];
  placeholder?: string;
  className?: string;
  disabled?: boolean;

  error?: boolean;

  showSearch?: boolean;
  searchPlaceholder?: string;
  allowClear?: boolean;
  allowCustom?: boolean;
  customPlaceholder?: string;
  customButtonText?: string;
  onCustomAdd?: (value: string) => void;
}

export function Select({
  value,
  onChange,
  options = [],
  placeholder,
  className,
  disabled,
  error,

  showSearch,
  searchPlaceholder,
  allowClear = true,
  allowCustom,
  customPlaceholder,
  customButtonText,
  onCustomAdd,
}: SelectProps) {
  const { t } = useI18n();

  const _customPlaceholder = customPlaceholder || 'Select...';

  const _customButtonText = customButtonText || t('common.add');

  const [open, setOpen] = useState(false);

  const [dropdownDirection, setDropdownDirection] = useState<'down' | 'up'>('down');

  const [customValue, setCustomValue] = useState('');

  const [searchValue, setSearchValue] = useState('');

  const buttonRef = useRef<HTMLDivElement>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  // Handle clicking outside to close the dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
        setSearchValue('');
      }
    };

    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [open]);

  // Check if the dropdown menu should expand up or down
  useEffect(() => {
    if (open && buttonRef.current) {
      const buttonRect = buttonRef.current.getBoundingClientRect();

      const spaceBelow = window.innerHeight - buttonRect.bottom - 50;

      const spaceAbove = buttonRect.top - 50;

      if (spaceBelow < 200 && spaceAbove > spaceBelow) {
        setDropdownDirection('up');
      } else {
        setDropdownDirection('down');
      }
    }
  }, [open]);

  const selectedOption = options.find((opt) => opt.value === value);

  const filteredOptions = options.filter((option) => {
    if (!showSearch || !searchValue.trim()) {
      return true;
    }

    const keyword = searchValue.toLowerCase();

    return (
      option.label.toLowerCase().includes(keyword) ||
      option.value.toLowerCase().includes(keyword) ||
      option.description?.toLowerCase().includes(keyword)
    );
  });

  const handleOptionClick = (option: SelectOption) => {
    if (disabled || option.disabled) return;

    onChange?.(option.value);

    setSearchValue('');
    setOpen(false);
  };
  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();

    if (disabled) return;

    onChange?.('');

    setSearchValue('');
    setOpen(false);
  };
  return (
    <div ref={containerRef} className={cn('relative', className)}>
      <div
        ref={buttonRef}
        onClick={() => {
          if (disabled) return;

          if (!showSearch) {
            setOpen(!open);
          }
        }}
        className={cn(
          'border-input flex h-9 w-full items-center justify-between rounded-md border bg-transparent px-3 py-1 text-sm outline-none transition-all',
          'focus-within:border-ring focus-within:ring-ring/50 focus-within:ring-[3px]',

          error &&
            'border-destructive focus-within:border-destructive focus-within:ring-destructive/40',

          !disabled && 'hover:text-accent-foreground',

          disabled && 'cursor-not-allowed opacity-50'
        )}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {showSearch ? (
            <input
              value={open ? searchValue : selectedOption?.label || ''}
              onChange={(e) => {
                if (disabled) return;

                setSearchValue(e.target.value);

                if (!open) {
                  setOpen(true);
                }
              }}
              onFocus={() => {
                if (disabled) return;

                setOpen(true);
              }}
              onBlur={() => {
                setTimeout(() => {
                  setSearchValue('');
                }, 100);
              }}
              placeholder={selectedOption?.label || placeholder || searchPlaceholder || 'Search...'}
              disabled={disabled}
              className="w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
              onClick={(e) => {
                e.stopPropagation();
              }}
            />
          ) : selectedOption ? (
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className="font-medium truncate">{selectedOption.label}</span>
            </div>
          ) : (
            <span className="text-muted-foreground">{placeholder || 'Select...'}</span>
          )}
        </div>

        {allowClear && value ? (
          <button
            type="button"
            onClick={handleClear}
            className={cn(
              'flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors',
              !disabled && 'hover:bg-muted hover:text-foreground'
            )}
            disabled={disabled}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        ) : (
          <ChevronDown
            className={cn(
              'h-4 w-4 flex-shrink-0 text-muted-foreground transition-transform',
              open && 'rotate-180'
            )}
          />
        )}
      </div>

      {open && (
        <div
          className={cn(
            'absolute left-0 right-0 z-[9999] mt-1 flex flex-col rounded-md border border-border bg-popover shadow-lg',
            dropdownDirection === 'down' ? 'top-full' : 'bottom-full mb-1 mt-0'
          )}
        >
          <div className="max-h-60 overflow-auto">
            {filteredOptions.length === 0 ? (
              <div className="py-10 text-center text-sm text-muted-foreground">
                {t('common.noOptions')}
              </div>
            ) : (
              filteredOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  disabled={option.disabled}
                  onClick={() => handleOptionClick(option)}
                  className={cn(
                    'w-full border-b border-border px-3 py-2 text-left text-sm transition-colors last:border-b-0',
                    !option.disabled && 'hover:bg-accent hover:text-accent-foreground',
                    value === option.value && 'bg-accent text-accent-foreground',
                    option.disabled && 'cursor-not-allowed text-muted-foreground opacity-50'
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 flex-1 items-center gap-2">
                      <span className="truncate font-medium">{option.label}</span>
                    </div>

                    {option.suffix && (
                      <div className="flex-shrink-0 text-muted-foreground">{option.suffix}</div>
                    )}
                    <span className="flex h-4 w-4 flex-shrink-0 items-center justify-center">
                      {value === option.value && <Check className="h-4 w-4 text-primary" />}
                    </span>
                  </div>

                  {option.description && (
                    <div className="mt-1 truncate text-xs text-muted-foreground">
                      {option.description}
                    </div>
                  )}
                </button>
              ))
            )}
          </div>

          {allowCustom && (
            <div className="flex shrink-0 gap-2 border-t border-border bg-muted/10 p-2">
              <input
                type="text"
                value={customValue}
                onChange={(e) => setCustomValue(e.target.value)}
                placeholder={_customPlaceholder}
                className="border-input flex h-8 w-full rounded-md border bg-background px-3 py-1 text-sm shadow-sm outline-none transition-all placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => {
                  e.stopPropagation();

                  if (e.key === 'Enter' && customValue.trim()) {
                    e.preventDefault();

                    onCustomAdd?.(customValue.trim());

                    setCustomValue('');
                    setOpen(false);
                  }
                }}
              />

              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();

                  if (customValue.trim()) {
                    onCustomAdd?.(customValue.trim());

                    setCustomValue('');
                    setOpen(false);
                  }
                }}
                disabled={!customValue.trim()}
                className="inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
              >
                {_customButtonText}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export {
  Select as SelectRadix,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
} from './select-radix';
