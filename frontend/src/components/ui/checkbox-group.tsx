// components/ui/checkbox-group.tsx
'use client'

import * as React from 'react'
import * as CheckboxPrimitive from '@radix-ui/react-checkbox'
import { Check } from 'lucide-react'

import { cn } from '@/lib/utils'

export interface CheckboxOption {
  label: React.ReactNode
  value: string
  disabled?: boolean
}

export interface CheckboxProps
  extends React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root> {
  label?: React.ReactNode
  error?: boolean
}

export interface CheckboxGroupChangeEvent {
  checked: boolean
  changedValue: string
}

export interface CheckboxGroupProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange'> {
  value?: string[]
  defaultValue?: string[]
  options?: CheckboxOption[]
  disabled?: boolean
  direction?: 'horizontal' | 'vertical'
  error?: boolean
  onChange?: (
    values: string[],
    event: CheckboxGroupChangeEvent,
  ) => void
  renderLabel?: (
    option: CheckboxOption,
    checked: boolean,
  ) => React.ReactNode
}

export const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  CheckboxProps
>(({ className, label, error, id, disabled, ...props }, ref) => {
  return (
    <label
      htmlFor={id}
      className={cn(
        'inline-flex items-center gap-2 text-sm font-medium leading-none',
        disabled && 'cursor-not-allowed opacity-50',
      )}
    >
      <CheckboxPrimitive.Root
        ref={ref}
        id={id}
        disabled={disabled}
        className={cn(
          'peer flex size-4 shrink-0 items-center justify-center rounded-md border outline-none transition-all',

          'border-input bg-background',

          'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',

          'data-[state=checked]:border-primary data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground',

          'disabled:cursor-not-allowed disabled:opacity-50',

          error &&
            'border-destructive focus-visible:border-destructive focus-visible:ring-destructive/40',

          className,
        )}
        {...props}
      >
        <CheckboxPrimitive.Indicator className="flex items-center justify-center text-current">
          <Check className="size-3.5" />
        </CheckboxPrimitive.Indicator>
      </CheckboxPrimitive.Root>

      {label}
    </label>
  )
})

Checkbox.displayName = CheckboxPrimitive.Root.displayName

export const CheckboxGroup = React.forwardRef<
  HTMLDivElement,
  CheckboxGroupProps
>(
  (
    {
      className,
      value,
      defaultValue = [],
      options = [],
      disabled,
      direction = 'horizontal',
      error,
      onChange,
      renderLabel,
      ...props
    },
    ref,
  ) => {
    const [internalValue, setInternalValue] =
      React.useState<string[]>(defaultValue)

    const isControlled = value !== undefined

    const currentValue = isControlled
      ? value
      : internalValue

    const handleCheckedChange = (
      checked: boolean | 'indeterminate',
      optionValue: string,
    ) => {
      const isChecked = checked === true

      const nextValue = isChecked
        ? [...currentValue, optionValue]
        : currentValue.filter(
            (item) => item !== optionValue,
          )

      const normalizedValue = [...new Set(nextValue)]

      if (!isControlled) {
        setInternalValue(normalizedValue)
      }

      onChange?.(normalizedValue, {
        checked: isChecked,
        changedValue: optionValue,
      })
    }

    return (
      <div
        ref={ref}
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
          const checked = currentValue.includes(
            option.value,
          )

          return (
            <Checkbox
              key={option.value}
              id={option.value}
              checked={checked}
              error={error}
              disabled={disabled || option.disabled}
              onCheckedChange={(checked) =>
                handleCheckedChange(
                  checked,
                  option.value,
                )
              }
              label={
                renderLabel
                  ? renderLabel(option, checked)
                  : option.label
              }
            />
          )
        })}
      </div>
    )
  },
)

CheckboxGroup.displayName = 'CheckboxGroup'