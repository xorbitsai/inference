'use client'

import * as React from 'react'
import * as SwitchPrimitive from '@radix-ui/react-switch'

import { cn } from '@/lib/utils'

export interface SwitchProps
  extends Omit<
    React.ComponentPropsWithoutRef<
      typeof SwitchPrimitive.Root
    >,
    'onCheckedChange' | 'onChange'
  > {
  error?: boolean
  onChange?: (checked: boolean) => void
}

export const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Root>,
  SwitchProps
>(
  (
    {
      className,
      error,
      onChange,
      disabled,
      ...props
    },
    ref,
  ) => {
    return (
      <SwitchPrimitive.Root
        ref={ref}
        disabled={disabled}
        onCheckedChange={onChange}
        className={cn(
          'peer inline-flex h-6 w-11 shrink-0 items-center rounded-full border-2 border-transparent transition-all outline-none',

          'cursor-pointer',

          'data-[state=checked]:bg-primary data-[state=unchecked]:bg-input',

          'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',

          'disabled:cursor-not-allowed disabled:opacity-50',

          error &&
            'focus-visible:border-destructive focus-visible:ring-destructive/40',

          className,
        )}
        {...props}
      >
        <SwitchPrimitive.Thumb
          className={cn(
            'pointer-events-none block size-5 rounded-full bg-background shadow-lg transition-transform',

            'data-[state=checked]:translate-x-5',
            'data-[state=unchecked]:translate-x-0',
          )}
        />
      </SwitchPrimitive.Root>
    )
  },
)

Switch.displayName = SwitchPrimitive.Root.displayName