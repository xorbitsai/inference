'use client';

import * as React from 'react';
import { Eye, EyeOff } from 'lucide-react';

import { cn } from '@/lib/utils';

interface InputProps extends React.ComponentProps<'input'> {
  error?: boolean;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, error, disabled, ...props }, ref) => {
    const [showPassword, setShowPassword] = React.useState(false);
    const isPassword = type === 'password';
    const inputClassName = cn(
      'border-input flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-sm outline-none transition-all',
      'placeholder:text-muted-foreground',
      'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
      error &&
        'border-destructive focus-visible:border-destructive focus-visible:ring-destructive/40',
      isPassword && 'pr-10',
      !isPassword && className
    );
    const input = (
      <input
        ref={ref}
        type={isPassword && showPassword ? 'text' : type}
        data-slot="input"
        disabled={disabled}
        className={inputClassName}
        placeholder="Please enter"
        {...props}
      />
    );

    if (isPassword) {
      const Icon = showPassword ? EyeOff : Eye;

      return (
        <div className={cn('relative w-full', className)}>
          {input}
          <button
            type="button"
            aria-label={showPassword ? 'Hide password' : 'Show password'}
            disabled={disabled}
            className="absolute right-2 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
            onClick={() => setShowPassword((prev) => !prev)}
          >
            <Icon className="h-4 w-4" />
          </button>
        </div>
      );
    }

    return input;
  }
);

Input.displayName = 'Input';

export { Input };
