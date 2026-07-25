'use client';

import * as React from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { Check, ChevronDown, X } from 'lucide-react';

import { cn } from '@/lib/utils';
import { useI18n } from '@/contexts/i18n-context';

export interface AutoCompleteOption {
  label: string;
  value: string;
  description?: string;
}

interface AutoCompleteProps {
  value?: string;
  onChange?: (value?: string) => void;

  options?: AutoCompleteOption[];

  optionsTips?: React.ReactNode;

  placeholder?: string;

  className?: string;

  disabled?: boolean;

  error?: boolean;

  allowClear?: boolean;

  /**
   * Whether to allow entering values not in options
   */
  allowCustomValue?: boolean;

  /**
   * Text to display when there is no data
   */
  emptyText?: string;
}

export function AutoComplete({
  value,
  options = [],
  optionsTips,
  placeholder,
  className,
  disabled,
  error,
  allowClear = true,
  allowCustomValue = true,
  emptyText,
  onChange,
}: AutoCompleteProps) {
  const { t } = useI18n();

  const containerRef = useRef<HTMLDivElement>(null);

  const triggerRef = useRef<HTMLDivElement>(null);

  const dropdownRef = useRef<HTMLDivElement>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  const [open, setOpen] = useState(false);

  const [dropdownDirection, setDropdownDirection] = useState<'down' | 'up'>('down');

  const [dropdownStyle, setDropdownStyle] = useState<React.CSSProperties>();

  const [inputValue, setInputValue] = useState('');

  /**
   * Whether the user is currently typing
   */
  const [typing, setTyping] = useState(false);

  const selectedOption = useMemo(() => {
    return options.find((option) => option.value === value);
  }, [options, value]);

  /**
   * Sync input when value changes
   */
  useEffect(() => {
    if (!typing) {
      setInputValue(selectedOption?.label || value || '');
    }
  }, [selectedOption, value, typing]);

  const updateDropdownPosition = React.useCallback(() => {
    if (!triggerRef.current) return;

    const triggerRect = triggerRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - triggerRect.bottom - 50;
    const spaceAbove = triggerRect.top - 50;
    const direction = spaceBelow < 200 && spaceAbove > spaceBelow ? 'up' : 'down';

    setDropdownDirection(direction);
    setDropdownStyle({
      left: triggerRect.left,
      top: direction === 'down' ? triggerRect.bottom + 4 : triggerRect.top - 4,
      width: triggerRect.width,
      transform: direction === 'up' ? 'translateY(-100%)' : undefined,
    });
  }, []);

  const handleBlurBehavior = React.useCallback(() => {
    setOpen(false)
    setTyping(false)

    /**
     * Allow free text input
     */
    if (allowCustomValue) {
      onChange?.(inputValue || undefined)
      return
    }

    /**
     * Do not allow free text input
     * Restore the selected value
     */
    setInputValue(
      selectedOption?.label || ""
    )
  }, [
    allowCustomValue,
    inputValue,
    onChange,
    selectedOption?.label,
  ])

  /**
   * Close when clicking outside
   */
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;

      if (
        containerRef.current &&
        !containerRef.current.contains(target) &&
        !dropdownRef.current?.contains(target)
      ) {
        if (inputRef.current && document.activeElement === inputRef.current) {
          inputRef.current.blur();
        } else {
          handleBlurBehavior();
        }
      }
    };

    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handleBlurBehavior, open]);

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

  const filteredOptions = useMemo(() => {
    const keyword = inputValue.trim().toLowerCase();

    if (!keyword) {
      return options;
    }

    return options.filter((option) => {
      return (
        option.label.toLowerCase().includes(keyword) ||
        option.value.toLowerCase().includes(keyword) ||
        option.description?.toLowerCase().includes(keyword)
      );
    });
  }, [options, inputValue]);

  const handleSelect = (option: AutoCompleteOption) => {
    onChange?.(option.value);

    setInputValue(option.label);

    setTyping(false);
    setOpen(false);
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();

    if (disabled) return;

    setInputValue('');
    setTyping(false);

    onChange?.(undefined);

    inputRef.current?.focus();
  };

  return (
    <div ref={containerRef} className={cn('relative', className)}>
      <div
        ref={triggerRef}
        className={cn(
          'border-input flex h-9 w-full items-center rounded-md border bg-transparent px-3 py-1 text-sm outline-none transition-all',

          'focus-within:border-ring focus-within:ring-ring/50 focus-within:ring-[3px]',

          error &&
            'border-destructive focus-within:border-destructive focus-within:ring-destructive/40',

          disabled && 'cursor-not-allowed opacity-50'
        )}
        onClick={() => {
          if (disabled) return;

          inputRef.current?.focus();
        }}
      >
        <input
          ref={inputRef}
          value={inputValue}
          disabled={disabled}
          placeholder={placeholder}
          className={cn(
            'flex-1 bg-transparent outline-none placeholder:text-muted-foreground',
            disabled && 'cursor-not-allowed'
          )}
          onFocus={() => {
            if (disabled) return;

            setOpen(true);
          }}
          onBlur={handleBlurBehavior}
          onChange={(e) => {
            if (disabled) return;

            const nextValue = e.target.value;

            setTyping(true);
            setInputValue(nextValue);

            if (allowCustomValue) {
              onChange?.(nextValue || undefined);
            }

            if (!open) {
              setOpen(true);
            }
          }}
          onKeyDown={(e) => {
            if (disabled) return;
            /**
             * Tab
             */
            if (e.key === 'Tab') {
              setOpen(false);
            }

            /**
             * Enter:
             * allowCustomValue=true submit value
             */
            if (e.key === 'Enter' && allowCustomValue) {
              e.preventDefault();

              inputRef.current?.blur();
            }

            /**
             * ESC
             */
            if (e.key === 'Escape') {
              setOpen(false);
              inputRef.current?.blur();
            }
          }}
        />

        {allowClear && value ? (
          <button
            type="button"
            disabled={disabled}
            onClick={handleClear}
            className={cn(
              'ml-2 flex h-4 w-4 shrink-0 items-center justify-center rounded-sm text-muted-foreground transition-colors',

              !disabled && 'hover:bg-muted hover:text-foreground'
            )}
          >
            <X className="h-4 w-4" />
          </button>
        ) : !allowCustomValue ? (
          <ChevronDown
            className={cn(
              'ml-2 h-4 w-4 shrink-0 text-muted-foreground transition-transform',
              open && 'rotate-180'
            )}
          />
        ) : null}
      </div>

      {open &&
        dropdownStyle &&
        createPortal(
          <div
            ref={dropdownRef}
            data-slot="auto-complete-dropdown"
            style={dropdownStyle}
            className={cn(
              'pointer-events-auto fixed z-[9999] overflow-hidden rounded-md border border-border bg-popover shadow-lg',
              dropdownDirection === 'up' && 'origin-bottom'
            )}
          >
            <div className="max-h-60 overflow-auto p-1">
              {filteredOptions.length === 0 ? (
                <div className="py-10 text-center text-sm text-muted-foreground">
                  {emptyText || t('common.noOptions')}
                </div>
              ) : (
                <>
                  {!!optionsTips && (
                    <div className="px-3 py-2 text-muted-foreground">{optionsTips}</div>
                  )}
                  {filteredOptions.map((option) => {
                    const active = value === option.value;

                    return (
                      <button
                        key={option.value}
                        type="button"
                        className={cn(
                          'flex w-full items-start justify-between gap-2 rounded-[4px] px-3 py-2 text-left text-sm transition-colors',

                          active
                            ? 'hover:bg-primary/10 hover:text-primary'
                            : 'hover:bg-accent hover:text-accent-foreground',

                          active && 'bg-primary/10 text-primary'
                        )}
                        onMouseDown={(e) => {
                          e.preventDefault();
                        }}
                        onClick={() => handleSelect(option)}
                      >
                        <div className="min-w-0 flex-1">
                          <div className="truncate font-medium">{option.label}</div>

                          {option.description && (
                            <div className="mt-1 truncate text-xs text-muted-foreground">
                              {option.description}
                            </div>
                          )}
                        </div>

                        {active && <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" />}
                      </button>
                    );
                  })}
                </>
              )}
            </div>
          </div>,
          document.body
        )}
    </div>
  );
}
