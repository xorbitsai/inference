// components/ui/radio-group.tsx
'use client'

import * as React from 'react'
import * as RadioGroupPrimitive from '@radix-ui/react-radio-group'
import { Circle } from 'lucide-react'

import { cn } from '@/lib/utils'

export type RadioValue = string | boolean

export interface RadioOption {
  label: React.ReactNode
  value: RadioValue
  disabled?: boolean
}

export interface RadioGroupItemProps
  extends React.ComponentPropsWithoutRef<
    typeof RadioGroupPrimitive.Item
  > {
  label?: React.ReactNode
  error?: boolean
}

export interface RadioGroupProps
  extends Omit<
    React.ComponentPropsWithoutRef<
      typeof RadioGroupPrimitive.Root
    >,
    'onValueChange' | 'onChange' | 'value'
  > {
  options?: RadioOption[]
  value?: RadioValue
  error?: boolean
  direction?: 'horizontal' | 'vertical'
  onChange?: (value: RadioValue) => void
  renderLabel?: (
    option: RadioOption,
    checked: boolean,
  ) => React.ReactNode
}

export const RadioGroupItem = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Item>,
  RadioGroupItemProps
>(
  (
    {
      className,
      label,
      error,
      disabled,
      id,
      ...props
    },
    ref,
  ) => {
    return (
      <label
        htmlFor={id}
        className={cn(
          'inline-flex items-center gap-2 text-sm font-medium leading-none',
          disabled && 'cursor-not-allowed opacity-50',
        )}
      >
        <RadioGroupPrimitive.Item
          ref={ref}
          id={id}
          disabled={disabled}
          className={cn(
            'aspect-square size-4 shrink-0 rounded-full border outline-none transition-all',

            'border-input bg-background text-primary',

            'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',

            'disabled:cursor-not-allowed disabled:opacity-50',

            error &&
              'border-destructive focus-visible:border-destructive focus-visible:ring-destructive/40',

            className,
          )}
          {...props}
        >
          <RadioGroupPrimitive.Indicator className="flex items-center justify-center">
            <Circle className="size-2 fill-current text-current" />
          </RadioGroupPrimitive.Indicator>
        </RadioGroupPrimitive.Item>

        {label}
      </label>
    )
  },
)

RadioGroupItem.displayName =
  RadioGroupPrimitive.Item.displayName

export const RadioGroup = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Root>,
  RadioGroupProps
>(
  (
    {
      className,
      options = [],
      direction = 'horizontal',
      error,
      onChange,
      renderLabel,
      value,
      disabled,
      ...props
    },
    ref,
  ) => {
    return (
      <RadioGroupPrimitive.Root
        ref={ref}
        value={
          value === undefined
            ? undefined
            : String(value)
        }
        disabled={disabled}
        onValueChange={(nextValue) => {
          onChange?.(
            options.find(
              (option) =>
                String(option.value) ===
                nextValue,
            )?.value ?? nextValue,
          )
        }}
        className={cn(
          'flex gap-3',
          direction === 'vertical'
            ? 'flex-col'
            : 'flex-row flex-wrap',
          className,
        )}
        {...props}
      >
        {options.map((option) => {
          const checked = value === option.value

          return (
            <RadioGroupItem
              key={String(option.value)}
              id={String(option.value)}
              value={String(option.value)}
              error={error}
              disabled={disabled || option.disabled}
              label={
                renderLabel
                  ? renderLabel(option, checked)
                  : option.label
              }
            />
          )
        })}
      </RadioGroupPrimitive.Root>
    )
  },
)

RadioGroup.displayName =
  RadioGroupPrimitive.Root.displayName
