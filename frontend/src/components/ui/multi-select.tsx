'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/lib/utils';
import { ChevronDown, X, Check } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';

interface MultiSelectOption {
  value: string;
  label: string;
  description?: string;
}

interface MultiSelectProps {
  value?: string[];
  onChange?: (value: string[]) => void;

  options: MultiSelectOption[];

  placeholder?: string;
  className?: string;

  creatable?: boolean;
  searchable?: boolean;
  disabled?: boolean;

  error?: boolean;
}

export function MultiSelect({
  value,
  onChange,

  options,

  placeholder,
  className,

  creatable,
  searchable,
  disabled,

  error,
}: MultiSelectProps) {
  const { t } = useI18n();

  const [open, setOpen] = useState(false);
  const [dropdownDirection, setDropdownDirection] = useState<'down' | 'up'>('down');
  const [dropdownStyle, setDropdownStyle] = useState<React.CSSProperties>();

  const [inputValue, setInputValue] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedValues = Array.isArray(value) ? value : [];
  const updateDropdownPosition = useCallback(() => {
    if (!triggerRef.current) return;

    const rect = triggerRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom - 50;
    const spaceAbove = rect.top - 50;
    const direction = spaceBelow < 200 && spaceAbove > spaceBelow ? 'up' : 'down';

    setDropdownDirection(direction);
    setDropdownStyle({
      left: rect.left,
      top: direction === 'down' ? rect.bottom + 4 : rect.top - 4,
      width: rect.width,
      transform: direction === 'up' ? 'translateY(-100%)' : undefined,
    });
  }, []);

  // click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;

      if (
        containerRef.current &&
        !containerRef.current.contains(target) &&
        !dropdownRef.current?.contains(target)
      ) {
        setOpen(false);
      }
    };

    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [open]);

  // dropdown direction
  useEffect(() => {
    if (!open) return;

    updateDropdownPosition();
    window.addEventListener('resize', updateDropdownPosition);
    window.addEventListener('scroll', updateDropdownPosition, true);

    return () => {
      window.removeEventListener('resize', updateDropdownPosition);
      window.removeEventListener('scroll', updateDropdownPosition, true);
    };
  }, [open, updateDropdownPosition]);

  useEffect(() => {
    const dropdown = dropdownRef.current;

    if (!dropdown) return;

    const stopScrollPropagation = (event: Event) => event.stopPropagation();

    dropdown.addEventListener('wheel', stopScrollPropagation);
    dropdown.addEventListener('touchmove', stopScrollPropagation);

    return () => {
      dropdown.removeEventListener('wheel', stopScrollPropagation);
      dropdown.removeEventListener('touchmove', stopScrollPropagation);
    };
  }, [open, dropdownStyle]);

  // selected options
  const selectedOptions = selectedValues.map((v) => {
    const option = options.find((o) => o.value === v);

    return option || { value: v, label: v };
  });

  // filter
  const filteredOptions = options.filter((option) => {
    const query = searchQuery.toLowerCase();

    return option.label.toLowerCase().includes(query) || option.value.toLowerCase().includes(query);
  });

  const toggleOpen = () => {
    if (disabled) return;

    setOpen((prev) => !prev);
  };

  const handleSelect = (nextValue: string) => {
    if (disabled) return;

    const newValues = selectedValues.includes(nextValue)
      ? selectedValues.filter((v) => v !== nextValue)
      : [...selectedValues, nextValue];

    onChange?.(newValues);
  };

  const handleRemove = (removeValue: string, e: React.MouseEvent) => {
    e.stopPropagation();

    if (disabled) return;

    const newValues = selectedValues.filter((v) => v !== removeValue);

    onChange?.(newValues);
  };

  const handleCreate = () => {
    if (disabled) return;

    const newValue = inputValue.trim();

    if (!newValue) return;

    if (!selectedValues.includes(newValue)) {
      onChange?.([...selectedValues, newValue]);
    }

    setInputValue('');
  };

  return (
    <div ref={containerRef} className={cn('relative w-full', className)}>
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        onClick={toggleOpen}
        className={cn(
          'border-input flex min-h-9 w-full items-center justify-between rounded-md border bg-transparent px-3 py-1 text-sm outline-none transition-all',

          'focus-within:border-ring focus-within:ring-ring/50 focus-within:ring-[3px]',

          error &&
            'border-destructive focus-within:border-destructive focus-within:ring-destructive/40',

          disabled && 'cursor-not-allowed opacity-50'
        )}
      >
        <div className="flex flex-1 items-center gap-1 overflow-hidden flex-wrap">
          {selectedOptions.length > 0 ? (
            selectedOptions.map((option) => (
              <div
                key={option.value}
                className="flex items-center gap-1 rounded-md bg-secondary px-2 py-0.5 text-xs text-secondary-foreground"
              >
                <span className="truncate max-w-[120px]">{option.label}</span>

                <button
                  type="button"
                  tabIndex={-1}
                  onClick={(e) => handleRemove(option.value, e)}
                  className="hover:text-destructive transition-colors"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))
          ) : (
            <span className="text-muted-foreground">{placeholder || 'Select...'}</span>
          )}
        </div>

        <ChevronDown
          className={cn(
            'ml-2 h-4 w-4 flex-shrink-0 text-muted-foreground transition-transform',
            open && 'rotate-180'
          )}
        />
      </button>

      {open &&
        dropdownStyle &&
        createPortal(
          <div
            ref={dropdownRef}
            data-slot="select-dropdown"
            style={dropdownStyle}
            className={cn(
              'pointer-events-auto fixed z-[9999] overflow-hidden rounded-md border border-border bg-popover text-popover-foreground shadow-md',
              dropdownDirection === 'up' && 'origin-bottom'
            )}
          >
            {searchable && (
              <div className="border-b p-2">
                <input
                  className="border-input focus-visible:border-ring focus-visible:ring-ring/50 flex h-8 w-full rounded-md border bg-transparent px-2 py-1 text-sm outline-none transition-all focus-visible:ring-[3px]"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
            )}

            <div className="max-h-60 overflow-auto">
              {filteredOptions.length === 0 && !creatable ? (
                <div className="py-10 text-center text-sm text-muted-foreground">
                  {t('common.noOptions')}
                </div>
              ) : (
                filteredOptions.map((option) => {
                  const active = selectedValues.includes(option.value);

                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => handleSelect(option.value)}
                      className={cn(
                        'w-full border-b border-border px-3 py-2 text-left text-sm transition-colors last:border-b-0',

                        'hover:bg-accent hover:text-accent-foreground',

                        active && 'bg-accent text-accent-foreground'
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{option.label}</span>
                        </div>
                        {active && <Check className="h-4 w-4 flex-shrink-0 text-primary" />}
                      </div>

                      {option?.description && (
                        <div className="mt-1 ml-2 text-xs text-muted-foreground">
                          {option.description}
                        </div>
                      )}
                    </button>
                  );
                })
              )}

              {creatable && (
                <div
                  className="sticky bottom-0 border-t bg-background p-2"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex gap-2">
                    <input
                      className="border-input focus-visible:border-ring focus-visible:ring-ring/50 flex h-8 flex-1 rounded-md border bg-transparent px-2 py-1 text-sm outline-none transition-all focus-visible:ring-[3px]"
                      placeholder="Custom..."
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          handleCreate();
                        }
                      }}
                    />

                    <button
                      type="button"
                      onClick={handleCreate}
                      className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-md px-3 text-xs transition-colors"
                    >
                      Add
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>,
          document.body
        )}
    </div>
  );
}
