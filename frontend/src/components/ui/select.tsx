'use client';

import * as React from 'react';
import { useState, useRef, useEffect, useCallback, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/lib/utils';
import { ChevronDown, Check, X } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';

const DROPDOWN_VIEWPORT_PADDING = 8;

export type SelectValue = string | number;

export interface SelectOption<T extends SelectValue = SelectValue> {
  value: T;
  label: string;
  disabled?: boolean;
  description?: string;
  prefix?: React.ReactNode;
  suffix?: React.ReactNode;
}

interface SelectProps<T extends SelectValue = SelectValue> {
  value?: T;
  onChange?: (value: T | undefined) => void;

  options?: SelectOption<T>[];
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
  dropdownAutoWidth?: boolean;
}

export function Select<T extends SelectValue = SelectValue>({
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
  dropdownAutoWidth = false,
}: SelectProps<T>) {
  const { t } = useI18n();

  const _customPlaceholder = customPlaceholder || 'Select...';

  const _customButtonText = customButtonText || t('common.add');

  const [open, setOpen] = useState(false);

  const [dropdownDirection, setDropdownDirection] = useState<'down' | 'up'>('down');

  const [customValue, setCustomValue] = useState('');

  const [searchValue, setSearchValue] = useState('');

  const buttonRef = useRef<HTMLDivElement>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  const dropdownRef = useRef<HTMLDivElement>(null);

  const optionsContainerRef = useRef<HTMLDivElement>(null);

  const autoDropdownWidthRef = useRef<number | undefined>(undefined);

  const [dropdownStyle, setDropdownStyle] = useState<React.CSSProperties>();

  const hasDropdownStyle = dropdownStyle !== undefined;

  const optionLayoutKey = JSON.stringify(
    options.map((option) => [
      option.value,
      option.label,
      option.description,
      Boolean(option.prefix),
      Boolean(option.suffix),
    ])
  );

  const updateDropdownPosition = useCallback(() => {
    if (!buttonRef.current) return;

    const buttonRect = buttonRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - buttonRect.bottom - 50;
    const spaceAbove = buttonRect.top - 50;
    const direction = spaceBelow < 200 && spaceAbove > spaceBelow ? 'up' : 'down';

    setDropdownDirection(direction);
    const dropdownWidth = dropdownAutoWidth
      ? Math.max(buttonRect.width, autoDropdownWidthRef.current ?? 0)
      : buttonRect.width;
    let dropdownLeft = buttonRect.left;

    if (dropdownAutoWidth) {
      dropdownLeft = Math.max(
        DROPDOWN_VIEWPORT_PADDING,
        Math.min(
          buttonRect.left,
          window.innerWidth - DROPDOWN_VIEWPORT_PADDING - dropdownWidth
        )
      );
    }

    setDropdownStyle({
      left: dropdownLeft,
      top: direction === 'down' ? buttonRect.bottom + 4 : buttonRect.top - 4,
      width: dropdownWidth,
      minWidth: buttonRect.width,
      transform: direction === 'up' ? 'translateY(-100%)' : undefined,
    });
  }, [dropdownAutoWidth]);

  // Handle clicking outside to close the dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;

      if (
        containerRef.current &&
        !containerRef.current.contains(target) &&
        !dropdownRef.current?.contains(target)
      ) {
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
    if (!open) return;

    updateDropdownPosition();
    window.addEventListener('resize', updateDropdownPosition);
    window.addEventListener('scroll', updateDropdownPosition, true);

    return () => {
      window.removeEventListener('resize', updateDropdownPosition);
      window.removeEventListener('scroll', updateDropdownPosition, true);
    };
  }, [open, updateDropdownPosition]);

  useLayoutEffect(() => {
    const dropdown = dropdownRef.current;
    const optionsContainer = optionsContainerRef.current;

    if (!open || !dropdownAutoWidth || !dropdown || !optionsContainer) return;

    const optionElements = Array.from(
      optionsContainer.querySelectorAll<HTMLElement>('[data-slot="select-option"]')
    );

    if (optionElements.length === 0) return;

    const originalWidths = optionElements.map((option) => option.style.width);
    let widestOptionWidth = 0;

    // Measure the complete rendered option structure without letting its
    // intrinsic width affect the visible dropdown layout.
    try {
      optionElements.forEach((option) => {
        option.style.width = 'max-content';
        widestOptionWidth = Math.max(widestOptionWidth, option.getBoundingClientRect().width);
      });
    } finally {
      optionElements.forEach((option, index) => {
        option.style.width = originalWidths[index];
      });
    }

    const optionsContainerStyle = window.getComputedStyle(optionsContainer);
    const optionsContainerPadding =
      Number.parseFloat(optionsContainerStyle.paddingLeft) +
      Number.parseFloat(optionsContainerStyle.paddingRight);
    const scrollbarWidth = optionsContainer.offsetWidth - optionsContainer.clientWidth;
    const dropdownBorderWidth = dropdown.offsetWidth - dropdown.clientWidth;
    const measuredWidth = Math.ceil(
      widestOptionWidth + optionsContainerPadding + scrollbarWidth + dropdownBorderWidth
    );

    if (autoDropdownWidthRef.current === measuredWidth) return;

    autoDropdownWidthRef.current = measuredWidth;
    updateDropdownPosition();
  }, [dropdownAutoWidth, hasDropdownStyle, open, optionLayoutKey, updateDropdownPosition]);

  useEffect(() => {
    const dropdown = dropdownRef.current;

    if (!dropdown) return;

    // Keep portal dropdown scrolling inside the menu instead of letting Dialog's
    // document-level scroll lock treat it as an outside interaction.
    const stopScrollPropagation = (event: Event) => event.stopPropagation();

    dropdown.addEventListener('wheel', stopScrollPropagation);
    dropdown.addEventListener('touchmove', stopScrollPropagation);

    return () => {
      dropdown.removeEventListener('wheel', stopScrollPropagation);
      dropdown.removeEventListener('touchmove', stopScrollPropagation);
    };
  }, [open, dropdownStyle]);

  const selectedOption = options.find((opt) => opt.value === value);
  const hasValue = value !== undefined && value !== null && String(value) !== '';

  const filteredOptions = options.filter((option) => {
    if (!showSearch || !searchValue.trim()) {
      return true;
    }

    const keyword = searchValue.toLowerCase();

    return (
      option.label.toLowerCase().includes(keyword) ||
      String(option.value).toLowerCase().includes(keyword) ||
      option.description?.toLowerCase().includes(keyword)
    );
  });

  const handleOptionClick = (option: SelectOption<T>) => {
    if (disabled || option.disabled) return;

    onChange?.(option.value);

    setSearchValue('');
    setOpen(false);
  };
  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();

    if (disabled) return;

    onChange?.(undefined);

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
            <div className="flex items-center gap-1 min-w-0 flex-1">
              {!!selectedOption.prefix && <span className='shrink-0'>{selectedOption.prefix}</span>}
              <span className="font-medium truncate">{selectedOption.label}</span>
            </div>
          ) : (
            <span className="text-muted-foreground truncate">{placeholder || 'Select...'}</span>
          )}
        </div>

        {allowClear && hasValue ? (
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

      {open &&
        dropdownStyle &&
        createPortal(
        <div
          ref={dropdownRef}
          data-slot="select-dropdown"
          style={dropdownStyle}
          className={cn(
            'pointer-events-auto fixed z-[9999] flex flex-col rounded-md border border-border bg-popover shadow-lg',
            dropdownDirection === 'up' && 'origin-bottom'
          )}
        >
          <div ref={optionsContainerRef} className="max-h-60 overflow-auto p-1">
            {filteredOptions.length === 0 ? (
              <div className="py-10 text-center text-sm text-muted-foreground">
                {t('common.noOptions')}
              </div>
            ) : (
              filteredOptions.map((option) => (
                <button
                  key={option.value}
                  data-slot="select-option"
                  type="button"
                  disabled={option.disabled}
                  onClick={() => handleOptionClick(option)}
                  className={cn(
                    'w-full rounded-[4px] px-3 py-2 text-left text-sm transition-colors',
                    !option.disabled &&
                      (value === option.value
                        ? 'hover:bg-primary/10 hover:text-primary'
                        : 'hover:bg-accent hover:text-accent-foreground'),
                    value === option.value && 'bg-primary/10 text-primary',
                    option.disabled && 'cursor-not-allowed text-muted-foreground opacity-50'
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 flex-1 items-center gap-1">
                      {!!option.prefix && <span className='shrink-0'>{option.prefix}</span>}
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
        </div>,
        document.body
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
