'use client';

import type { ReactNode } from 'react';
import * as RadioGroupPrimitive from '@radix-ui/react-radio-group';

import { cn } from '@/lib/utils';
import type { BaseFormFieldProps } from '@/types/form';

export interface SegmentedOption<T extends string = string> {
  label: ReactNode;
  value: T;
  disabled?: boolean;
}

export interface SegmentedProps<T extends string = string> extends Omit<
  BaseFormFieldProps<T>,
  'placeholder'
> {
  block?: boolean;
  options: SegmentedOption<T>[];
}

export function Segmented<T extends string = string>({
  block = false,
  className,
  disabled,
  error,
  onChange,
  options,
  value,
}: SegmentedProps<T>) {
  return (
    <RadioGroupPrimitive.Root
      value={value}
      disabled={disabled}
      onValueChange={(nextValue) => {
        const selected = options.find((option) => option.value === nextValue);
        if (selected) onChange?.(selected.value);
      }}
      className={cn(
        'inline-flex max-w-full self-start flex-wrap gap-1 rounded-lg bg-muted p-1',
        block && 'w-full self-stretch',
        error && 'ring-1 ring-destructive',
        disabled && 'opacity-50',
        className
      )}
    >
      {options.map((option) => {
        const selected = option.value === value;

        return (
          <RadioGroupPrimitive.Item
            key={option.value}
            disabled={disabled || option.disabled}
            value={option.value}
            className={cn(
              'min-w-20 rounded-md px-4 py-2 text-sm font-medium text-muted-foreground transition-colors',
              block && 'flex-1',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50',
              selected && 'bg-background text-foreground shadow-sm'
            )}
          >
            {option.label}
          </RadioGroupPrimitive.Item>
        );
      })}
    </RadioGroupPrimitive.Root>
  );
}
