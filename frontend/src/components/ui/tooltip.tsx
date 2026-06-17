'use client';

import * as React from 'react';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { CircleQuestionMark, type LucideIcon } from 'lucide-react';

import { cn } from '@/lib/utils';

const TooltipProvider = TooltipPrimitive.Provider;

const Tooltip = TooltipPrimitive.Root;

const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        'z-[10000] overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
        className
      )}
      {...props}
    />
  </TooltipPrimitive.Portal>
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

interface InfoTooltipProps {
  content: React.ReactNode;
  icon?: LucideIcon;
  iconClassName?: string;
  contentClassName?: string;
  children?: React.ReactNode;
}

export function InfoTooltip({
  content,
  icon: Icon = CircleQuestionMark,
  iconClassName,
  contentClassName,
  children,
}: InfoTooltipProps) {
  const normalizedContent = typeof content === 'string' ? content.replace(/\\n/g, '\n') : content;

  return (
    <TooltipProvider>
      <Tooltip delayDuration={300}>
        <TooltipTrigger asChild>
          {children || (
            <span className="inline-flex cursor-pointer items-center">
              <Icon
                className={cn(
                  'h-3.5 w-3.5 text-muted-foreground/70 hover:text-muted-foreground',
                  iconClassName
                )}
              />
            </span>
          )}
        </TooltipTrigger>
        <TooltipContent sideOffset={6} className={cn('max-w-72', contentClassName)}>
          <div className="whitespace-pre-line">{normalizedContent}</div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
